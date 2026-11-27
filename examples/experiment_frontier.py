"""Frontier experiment: four importance scores for block-wise KV eviction.

Controlled comparison — the *only* variable is the per-token importance score
fed to the SAME block-level policy (BlockKV). Four scores, attention fidelity /
recency varying:

  proxy    : ||V_i|| / ||K_i||            (PagedEviction; no query, no recency)
  exact    : sum over ALL queries         (single-pass Q K^T)
  window   : sum over last W queries      (SnapKV / KV-Compress observation window)
  recency  : exponential-decay over queries (smooth recency weighting; ours)

Quality (perplexity) is averaged over several held-out continuations to reduce
single-sample noise. Run: python examples/experiment_frontier.py
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from vllm_kv_squeeze.policy import BlockKVPolicy

MODEL = os.path.expanduser("~/models/Qwen2.5-7B-Instruct")
CORPUS = os.path.expanduser("~/wiki_zh/wikipedia-zh-cn-20250320.json")
BLOCK_SIZE = 16
OBS_WINDOW = 64
DECAY = 0.02  # exponential recency decay over queries
BUDGETS = [0.25, 0.35, 0.45, 0.55, 0.65]
N_CONTEXTS = 30


def load_contexts(tok, n, ctx_len=2000, cont_len=128):
    chunks = []
    ids = []
    with open(CORPUS) as f:
        for line in f:
            ids += tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            while len(ids) >= ctx_len + cont_len and len(chunks) < n:
                chunks.append((ids[:ctx_len], ids[ctx_len:ctx_len + cont_len]))
                ids = ids[ctx_len:]
            if len(chunks) >= n:
                break
    return [(torch.tensor([c]), torch.tensor([k])) for c, k in chunks]


def compute_scores(model, ctx):
    C = ctx.shape[1]
    attn = model.model.layers[-1].self_attn
    cap = {}

    def hook(module, args, kwargs, output):
        cap["h"] = kwargs["hidden_states"].detach()

    handle = attn.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        out = model(ctx, output_attentions=True)
    handle.remove()

    w = out.attentions[-1].float()                     # [1, H, C, C]
    exact = w.sum(dim=(0, 1, 2))
    window = w[:, :, -OBS_WINDOW:, :].sum(dim=(0, 1, 2))

    # exponential recency decay over the query dimension
    q = torch.arange(C, device=w.device, dtype=torch.float32)
    qw = torch.exp(DECAY * (q - (C - 1)))              # 1 at last query, decays back
    recency = (w * qw[None, None, :, None]).sum(dim=(0, 1, 2))

    h = cap["h"]
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)
    k = attn.k_proj(h).view(1, C, n_kv, head_dim)
    v = attn.v_proj(h).view(1, C, n_kv, head_dim)
    proxy = v.norm(dim=-1).mean(dim=(0, 2)) / (k.norm(dim=-1).mean(dim=(0, 2)) + 1e-8)

    return {"exact": exact, "window": window, "proxy": proxy, "recency": recency}


def count_freed_blocks(evict_indices, seq_len, block_size):
    ev = set(evict_indices.tolist())
    return sum(
        1 for b in range(0, seq_len - block_size + 1, block_size)
        if all((b + i) in ev for i in range(block_size))
    )


def perplexity(model, ctx, cont, evict_mask):
    ids = torch.cat([ctx, cont], dim=1)
    labels = ids.clone()
    labels[:, :ctx.shape[1]] = -100
    with torch.no_grad():
        out = model(ids, labels=labels, attention_mask=evict_mask)
    return torch.exp(out.loss).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--rescue", type=float, default=0.0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model = model.to("cuda")
    model.set_attn_implementation("eager")
    model.eval()

    contexts = load_contexts(tok, N_CONTEXTS)
    print(f"{len(contexts)} contexts loaded", flush=True)

    policy = BlockKVPolicy(window_size=args.window, block_size=BLOCK_SIZE,
                           rescue_ratio=args.rescue)

    # accumulate: score -> budget -> [ppl list across contexts]
    acc = {}
    full_ppls = []

    for ci, (ctx, cont) in enumerate(contexts):
        ctx, cont = ctx.to("cuda"), cont.to("cuda")
        C = ctx.shape[1]
        scores = compute_scores(model, ctx)

        full_mask = torch.ones(1, C + cont.shape[1], device="cuda")
        full_ppls.append(perplexity(model, ctx, cont, full_mask))

        for name, sc in scores.items():
            acc.setdefault(name, {br: [] for br in BUDGETS})
            for br in BUDGETS:
                budget = max(BLOCK_SIZE, int(C * br))
                res = policy.select(sc.cpu(), C, budget)
                mask = torch.ones(1, C + cont.shape[1], device="cuda")
                mask[:, res.evict_indices.to("cuda")] = 0
                acc[name][br].append(perplexity(model, ctx, cont, mask))
        print(f"context {ci + 1}/{len(contexts)} done", flush=True)

    # average (per-budget means)
    out = {"full_ppl": round(sum(full_ppls) / len(full_ppls), 3)}
    print(f"\nfull-context ppl: {out['full_ppl']}  (avg over {len(full_ppls)} contexts)\n")
    print("score     budget  avg_ppl")
    print("-" * 30)
    for name in ["proxy", "exact", "window", "recency"]:
        out[name] = []
        for br in BUDGETS:
            m = sum(acc[name][br]) / len(acc[name][br])
            out[name].append({"budget": br, "ppl": round(m, 3)})
            print(f"{name:8s} {br:.2f}   {m:.3f}", flush=True)
        print("-" * 30, flush=True)

    # ---- paired analysis: per-context average ppl per method ----
    methods = ["proxy", "exact", "window", "recency"]
    n = len(full_ppls)
    per_ctx = {m: [] for m in methods}
    for i in range(n):
        for m in methods:
            per_ctx[m].append(sum(acc[m][br][i] for br in BUDGETS) / len(BUDGETS))

    def paired(a, b):
        """mean difference (b - a), std err, t-stat, over contexts (paired)."""
        d = [per_ctx[b][i] - per_ctx[a][i] for i in range(n)]
        md = sum(d) / n
        sd = (sum((x - md) ** 2 for x in d) / (n - 1)) ** 0.5
        se = sd / (n ** 0.5)
        t = md / se if se > 0 else 0.0
        return md, se, t

    print(f"\n=== PAIRED ANALYSIS (n={n} contexts, mean diff / std err / t) ===")
    pairs = [("recency", "window"), ("recency", "proxy"),
             ("recency", "exact"), ("exact", "window")]
    paired_out = {}
    for a, b in pairs:
        md, se, t = paired(a, b)  # b - a; negative = a is better
        paired_out[f"{a}_vs_{b}"] = {"mean_diff": round(md, 3), "se": round(se, 3), "t": round(t, 2)}
        sig = "***" if abs(t) > 3 else "**" if abs(t) > 2.5 else "*" if abs(t) > 2 else "n.s."
        print(f"  {a} vs {b}: {md:+.3f} (se {se:.3f}, t={t:.2f}) {sig}", flush=True)

    print("\nEXPERIMENT_DONE")
    print(json.dumps({"full_ppl": out["full_ppl"], "paired": paired_out,
                      "results": {m: [{"budget": x["budget"], "ppl": x["ppl"]} for x in out[m]]
                                  for m in methods}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
