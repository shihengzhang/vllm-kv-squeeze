"""Quality experiment: blind vs ref_cnt-aware eviction at equal memory freed.

Simulates a prefix-caching scenario in transformers: a request's context is a
*shared* prefix (P tokens, ref_cnt=M — not freeable) + a *unique* suffix
(S tokens, ref_cnt=1 — freeable), followed by a held-out continuation.

Under a memory budget of F blocks (F*16 tokens), a ref_cnt-BLIND policy evicts
the lowest-attention tokens globally until F suffix blocks are freed — wasting
evictions on the shared prefix. A ref_cnt-AWARE policy evicts the F lowest-
attention suffix blocks directly. Both free the same memory; the blind policy
masks MORE tokens, so it should yield higher perplexity.

Run: python examples/prefix_eviction_quality.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from vllm_kv_squeeze.policy import BlockKVPolicy

MODEL = os.path.expanduser("~/models/Qwen2.5-7B-Instruct")
CORPUS = os.path.expanduser("~/wiki_zh/wikipedia-zh-cn-20250320.json")
BLOCK = 16
P, S, C = 512, 128, 64          # prefix (shared), suffix (unique), continuation
BUDGETS = [2, 4, 6]             # suffix blocks to free (memory budget)
N_CONTEXTS = 12


def load_context(tok):
    ids = []
    with open(CORPUS) as f:
        for line in f:
            ids += tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            if len(ids) >= P + S + C:
                break
    return torch.tensor([ids[:P + S]]), torch.tensor([ids[P + S:P + S + C]])


def load_many(tok, n):
    """Yield n independent (context, continuation) pairs from distinct articles."""
    out = []
    ids = []
    with open(CORPUS) as f:
        for line in f:
            ids += tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            while len(ids) >= P + S + C and len(out) < n:
                out.append((torch.tensor([ids[:P + S]]), torch.tensor([ids[P + S:P + S + C]])))
                ids = ids[P + S:]
            if len(out) >= n:
                break
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--contexts", type=int, default=N_CONTEXTS)
    args = ap.parse_args()

    print("loading ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model = model.to("cuda")
    model.set_attn_implementation("eager")
    model.eval()

    contexts = load_many(tok, args.contexts)
    print(f"{len(contexts)} contexts loaded (prefix={P}, suffix={S}, cont={C})", flush=True)

    # accumulate per-budget: masked tokens + ppl lists
    agg = {F: {"blind_masked": [], "blind_ppl": [], "aware_ppl": []} for F in BUDGETS}
    full_ppls = []

    for ci, (ctx, cont) in enumerate(contexts):
        ctx, cont = ctx.to("cuda"), cont.to("cuda")
        L = ctx.shape[1]
        with torch.no_grad():
            out = model(ctx, output_attentions=True)
        w = out.attentions[-1].float()
        scores = w.sum(dim=(0, 1, 2)).cpu()
        prefix_scores = scores[:P]
        suffix_scores = scores[P:P + S]

        def ppl(mask_tokens):
            ids = torch.cat([ctx, cont], dim=1)
            labels = ids.clone()
            labels[:, :L] = -100
            m = torch.ones(1, L + C, device="cuda")
            m[:, mask_tokens] = 0
            with torch.no_grad():
                o = model(ids, labels=labels, attention_mask=m)
            return torch.exp(o.loss).item()

        full_ppls.append(ppl([]))
        order = torch.argsort(scores)
        s_order = torch.argsort(suffix_scores)
        for F in BUDGETS:
            target = F * BLOCK
            blind_evict = []
            suffix_freed = 0
            for t in order.tolist():
                blind_evict.append(t)
                if t >= P:
                    suffix_freed += 1
                if suffix_freed >= target:
                    break
            aware_evict = (P + s_order[:target]).tolist()
            agg[F]["blind_masked"].append(len(blind_evict))
            agg[F]["blind_ppl"].append(ppl(blind_evict))
            agg[F]["aware_ppl"].append(ppl(aware_evict))
        print(f"context {ci + 1}/{len(contexts)} done", flush=True)

    n = len(contexts)
    print(f"\nfull ppl: {sum(full_ppls)/n:.3f}\n")
    print("budget  masked(blind vs aware)  ppl(blind vs aware)")
    print("-" * 56)
    results = {"full": round(sum(full_ppls) / n, 3), "blind": [], "aware": []}
    for F in BUDGETS:
        bm = sum(agg[F]["blind_masked"]) / n
        bp = sum(agg[F]["blind_ppl"]) / n
        ap = sum(agg[F]["aware_ppl"]) / n
        results["blind"].append({"F": F, "masked": round(bm, 1), "ppl": round(bp, 3)})
        results["aware"].append({"F": F, "ppl": round(ap, 3)})
        print(f"  {F:2d}   {bm:6.1f} / {F * BLOCK:4d}          {bp:.3f} / {ap:.3f}", flush=True)

    print("\nQUALITY_DONE")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
