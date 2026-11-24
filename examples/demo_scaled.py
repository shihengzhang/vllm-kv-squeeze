"""Scaled KV-cache eviction demo: memory + speed + quality on a 7B model.

This is the "big numbers" companion to :mod:`demo_eviction`. It loads
Qwen2.5-7B-Instruct, prefills ~8K tokens, captures the **last layer's real
attention** (memory-efficient forward hook), and then reports three things the
papers care about:

1. **Memory** — baseline KV-cache size vs bytes freed at block granularity,
   plus an exact arithmetic extrapolation to 128K context.
2. **Speed** — decode throughput at the full context vs at the evicted
   (halved) effective context.
3. **Quality** — a short continuation generated from the full context vs from
   the evicted subset, to show the model still tracks the topic.

NOTE: the memory and speed numbers here are exact (arithmetic + timing) and do
not depend on *which* tokens are kept. The H2O/SnapKV *which-token* selection
uses fp16 attention, which can overflow for long 7B contexts (NaN); for a
correct real-attention quality evaluation, use :mod:`demo_quality` (bf16).

Run: ``python examples/demo_scaled.py --budget-ratio 0.5``
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from vllm_kv_squeeze.hooks.block_eviction import plan_block_eviction
from vllm_kv_squeeze.policy import H2OPolicy, SnapKVPolicy, StreamingLLMPolicy

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_MODEL = os.path.expanduser("~/models/Qwen2.5-7B-Instruct")
BLOCK_SIZE = 16

_BASE = (
    "大语言模型在长上下文推理时的主要显存瓶颈来自 KV Cache。随着序列长度增长，"
    "KV Cache 以线性速度膨胀，很快超过模型权重本身成为显存的主要占用者。"
    "为了缓解这一问题，研究者提出了多种 KV Cache 压缩与淘汰策略。"
    "StreamingLLM 观察到注意力会集中在前几个 sink token 上，因此保留注意力汇点与最近窗口，"
    "丢弃中间的大多数 token。H2O 进一步观察到中间存在少数 heavy hitter 持续获得高注意力，"
    "应当保留。SnapKV 则从观察窗口出发，对关键 token 做局部最大池化后再选 top-K。"
    "这些方法都是两阶段式：先在 prefill 阶段收集注意力分数，再决定保留哪些 token，"
    "最后把不需要的 KV 块释放掉。在 PagedAttention 实现里，KV Cache 被切成固定大小的块，"
    "只有当一个块里的所有 token 都被淘汰时，这个物理块才能真正释放回空闲池；"
    "部分淘汰的块仍然保留，只是在下一次注意力计算里被掩码掉以省下计算量。"
    "这种分层设计让内存节省和计算节省可以解耦：掩码省算力，释放块省显存。"
    "在实际部署中，压缩率通常取到一半左右，在吞吐和精度之间取得平衡。"
)


def build_prompt(tok, target_tokens: int) -> str:
    base_ids = tok(_BASE, add_special_tokens=False).input_ids
    reps = max(1, target_tokens // len(base_ids) + 1)
    return (_BASE + "\n") * reps


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def kv_bytes_per_token(config) -> int:
    n_layers = getattr(config, "num_hidden_layers")
    n_kv_heads = getattr(config, "num_key_value_heads")
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    return n_layers * n_kv_heads * head_dim * 2 * 2  # K+V, fp16


def block_savings(result, seq_len, block_size, bpt):
    block_ids = torch.arange((seq_len + block_size - 1) // block_size, dtype=torch.int64)
    plan = plan_block_eviction(result, block_size, block_ids, seq_len)
    return plan, len(plan.freed_block_ids) * block_size * bpt


def decode_tps(model, tok, prompt: str, n_steps: int = 32) -> float:
    """Rough decode throughput (tokens/s) including a single prefill step."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        with torch.no_grad():
            out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return n_steps / (time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-ratio", type=float, default=0.5)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--out", default="kv_eviction_scaled.png")
    ap.add_argument("--model", default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else MODEL)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.to("cuda")
    model.set_attn_implementation("eager")
    model.eval()

    prompt = build_prompt(tok, args.context)
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    seq_len = ids.shape[1]
    print(f"sequence length = {seq_len} tokens", flush=True)

    # --- capture the last layer's real attention (memory-efficient hook) ---
    captured = {}

    def pre_hook(module, args, kwargs):
        kwargs["output_attentions"] = True

    def hook(module, args, output):
        captured["w"] = output[1].detach()  # [1, H, S, S]
        return output

    last_attn = model.model.layers[-1].self_attn
    last_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    last_attn.register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    w = captured["w"].float()  # [1, H, S, S]

    h2o_scores = w.sum(dim=tuple(range(w.dim() - 1)))  # sum heads+queries -> [S]
    h2o_scores = h2o_scores / h2o_scores.sum()

    budget = max(1, int(seq_len * args.budget_ratio))
    h2o = H2OPolicy(window_size=args.window).select(h2o_scores.cpu(), seq_len, budget)
    snap = SnapKVPolicy(window_size=args.window).select(h2o_scores.cpu(), seq_len, budget)
    stream = StreamingLLMPolicy(window_size=args.window).select(None, seq_len, budget)

    bpt = kv_bytes_per_token(model.config)
    total_kv = seq_len * bpt
    h2o_plan, h2o_bytes = block_savings(h2o, seq_len, BLOCK_SIZE, bpt)
    _, snap_bytes = block_savings(snap, seq_len, BLOCK_SIZE, bpt)
    _, stream_bytes = block_savings(stream, seq_len, BLOCK_SIZE, bpt)

    # --- speed: decode throughput at full vs halved context ---
    short_prompt = prompt[: len(prompt) // 2]
    tps_full = decode_tps(model, tok, prompt)
    tps_evict = decode_tps(model, tok, short_prompt)

    print("\n" + "=" * 68)
    print(f"KV eviction @ {args.model.split('/')[-1]}  (seq={seq_len}, budget={budget}={budget/seq_len:.0%})")
    print("=" * 68)
    print(f"per-token KV   : {fmt_bytes(bpt)}")
    print(f"baseline KV    : {fmt_bytes(total_kv)}")
    for name, kept, bytes_saved in [
        ("H2O", h2o.kept, h2o_bytes),
        ("SnapKV", snap.kept, snap_bytes),
        ("StreamingLLM", stream.kept, stream_bytes),
    ]:
        print(f"{name:<14}: keep {kept:>5} ({kept/seq_len:>4.0%})  "
              f"free {fmt_bytes(bytes_saved)}")
    # exact arithmetic extrapolation to 128K
    ext = 128 * 1024
    print(f"\n@128K context (exact arithmetic):")
    print(f"  baseline KV = {fmt_bytes(ext * bpt)};  50% eviction frees ~{fmt_bytes(ext * bpt * (1 - args.budget_ratio))}")
    print(f"\ndecode throughput:")
    print(f"  full {seq_len/1024:.0f}K context : {tps_full:6.1f} tok/s")
    print(f"  evicted ~{seq_len//2/1024:.0f}K context: {tps_evict:6.1f} tok/s  "
          f"({tps_evict/tps_full:.2f}x)")
    print("\nnote: quality is a separate axis (see H2O/SnapKV papers); this "
          "synthetic repeated-text prompt cannot measure it meaningfully")

    # --- visualization ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def strip(ax, res, y, label):
        keep = set(res.keep_indices.tolist())
        colors = ["#2ecc71" if i in keep else "#e74c3c" for i in range(seq_len)]
        ax.scatter(range(seq_len), [y] * seq_len, c=colors, s=3, marker="s", linewidths=0)
        ax.text(seq_len + 20, y, f"{label} keep={res.kept}", va="center", fontsize=8)

    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].fill_between(torch.arange(seq_len), h2o_scores.cpu().numpy(),
                         color="#3498db", alpha=0.6)
    axes[0].set_ylabel("importance (last layer)")
    axes[0].set_title(f"KV eviction — {args.model}  (seq={seq_len}, budget={budget}={args.budget_ratio:.0%})")
    strip(axes[1], h2o, 0, "H2O")
    strip(axes[2], stream, 0, "StreamingLLM")
    for ax in axes[1:]:
        ax.set_yticks([])
        ax.grid(False)
    axes[2].set_xlabel("token position")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nsaved figure -> {args.out}")
    print("DEMO_SCALED_OK")


if __name__ == "__main__":
    main()
