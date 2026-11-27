"""LongBench QA eval: accuracy under blind vs ref_cnt-aware eviction.

Uses LongBench multifieldqa_zh (Chinese single-doc QA). Attention scores come
from the LAST layer only, computed via a forward hook + manual Q@K^T (single
layer, O(L^2) memory instead of O(L^2 * num_layers) from output_attentions).

  blind : evict the lowest-attention K tokens (masks ~3x more for equal memory)
  aware : evict the lowest-attention K/3 tokens (same memory, fewer masked)

Score = char-level F1 vs reference answers. Run on Qwen2.5-7B.

Run: python examples/prefix_eviction_longbench.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

MODEL = os.path.expanduser("~/models/Qwen2.5-7B-Instruct")
DATA = os.path.expanduser("~/LongBench/data/multifieldqa_zh.jsonl")
N_SAMPLES = 40
CTX_MAX = 4000
EVICT_FRAC = 0.5
AWARE_DIVISOR = 3


def last_layer_scores(model, ctx_ids):
    """Last-layer per-token attention received (single-pass, no output_attentions)."""
    L = ctx_ids.shape[1]
    last = model.model.layers[-1]
    attn = last.self_attn
    cap = {}

    def hook(module, args, kwargs, output):
        cap["h"] = kwargs["hidden_states"].detach()

    handle = attn.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(ctx_ids)
    handle.remove()

    h = cap["h"]                               # [1, L, hidden] (bf16)
    cfg = model.config
    H, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    q = attn.q_proj(h).view(1, L, H, d).transpose(1, 2).float()
    k = attn.k_proj(h).view(1, L, Hkv, d).transpose(1, 2).float()
    k = k.repeat_interleave(H // Hkv, dim=1)
    w = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    w = torch.softmax(w, dim=-1)
    return w.sum(dim=(0, 1, 2)).cpu()         # [L]


def f1_char(pred: str, refs: list[str]) -> float:
    p = set(pred.strip())
    best = 0.0
    for r in refs:
        r = set(r.strip())
        if not p or not r:
            continue
        inter = len(p & r)
        prec = inter / len(p)
        rec = inter / len(r)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best = max(best, f1)
    return best


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = [json.loads(l) for l in open(DATA)][:N_SAMPLES]
    print(f"{len(samples)} samples loaded", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model = model.to("cuda")
    model.set_attn_implementation("eager")
    model.eval()

    scores = {"full": [], "blind": [], "aware": []}

    for si, s in enumerate(samples):
        ctx = tok(s["context"], add_special_tokens=False)["input_ids"][:CTX_MAX]
        q = tok(s["input"], add_special_tokens=False)["input_ids"]
        refs = s["answers"]
        L = len(ctx)
        ctx_t = torch.tensor([ctx], device="cuda")
        q_t = torch.tensor([q], device="cuda")

        def gen(mask_ctx_idx):
            m = torch.ones(1, L + len(q), device="cuda")
            if mask_ctx_idx:
                m[:, mask_ctx_idx] = 0
            ids = torch.cat([ctx_t, q_t], dim=1)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=m, max_new_tokens=32,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][L + len(q):], skip_special_tokens=True)

        scores["full"].append(f1_char(gen(None), refs))

        sc = last_layer_scores(model, ctx_t)
        order = torch.argsort(sc).tolist()

        k_blind = int(L * EVICT_FRAC)
        scores["blind"].append(f1_char(gen(order[:k_blind]), refs))
        k_aware = k_blind // AWARE_DIVISOR
        scores["aware"].append(f1_char(gen(order[:k_aware]), refs))

        if (si + 1) % 10 == 0:
            print(f"{si + 1}/{len(samples)} done", flush=True)

    def avg(k):
        return sum(scores[k]) / len(scores[k]) if scores[k] else 0.0

    print("\n=== LongBench QA (multifieldqa_zh, char-F1) ===")
    print(f"full  (no eviction): {avg('full'):.3f}")
    print(f"blind (evict {EVICT_FRAC:.0%}):     {avg('blind'):.3f}")
    print(f"aware (evict {EVICT_FRAC / AWARE_DIVISOR:.0%}):     {avg('aware'):.3f}")
    print("\nLONGBENCH_DONE")
    print(json.dumps({k: round(avg(k), 3) for k in scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
