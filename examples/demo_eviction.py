"""End-to-end KV-cache eviction demo (requires a CUDA GPU).

Pipeline
--------
1. Load a real LLM (Qwen2.5-0.5B-Instruct by default) on one GPU.
2. Prefill a long document and capture **real** attention weights.
3. Derive per-token importance scores (H2O / SnapKV style).
4. Run the ``vllm-kv-squeeze`` policies to produce keep/evict sets.
5. Map the token-level decision to PagedAttention blocks (block_size=16)
   and count how many blocks — and how many bytes of KV cache — get freed.
6. Free real ``vllm.v1.core.block_pool.BlockPool`` blocks with
   :func:`free_evicted_blocks` to prove the mutation path works.
7. Save a visualization PNG and print a summary.

Run
---
.. code-block:: bash

    python examples/demo_eviction.py --budget-ratio 0.5
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from vllm_kv_squeeze.hooks.block_eviction import plan_block_eviction
from vllm_kv_squeeze.hooks.v1 import free_evicted_blocks
from vllm_kv_squeeze.policy import H2OPolicy, SnapKVPolicy, StreamingLLMPolicy

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16

# A long-ish technical passage so the eviction has something to chew on.
PROMPT = (
    "大语言模型在长上下文推理时的显存瓶颈主要来自 KV Cache。随着序列长度增长，"
    "KV Cache 以线性速度膨胀，很快就超过模型权重本身成为显存的主要占用者。"
    "例如一个 7B 模型在 128K 上下文下，KV Cache 可能与权重相当甚至更大。"
    "为了缓解这个问题，研究者提出了多种 KV Cache 压缩与淘汰策略。"
    "StreamingLLM 观察到注意力分数会集中在前几个 sink token 上，因此保留注意力汇点、"
    "最近窗口，并丢弃中间的大多数 token。它实现简单，但会丢失中间的重要信息。"
    "H2O（Heavy-Hitter Oracle）进一步观察到，除了注意力汇点之外，还有少数中间 token "
    "会持续获得较高的累计注意力，这些 heavy hitter 应当被保留。H2O 保留注意力汇点、"
    "最近窗口，以及中间累计注意力最高的若干 token，在压缩率很高时仍能保持较好的质量。"
    "SnapKV 则从观察窗口的注意力出发，对关键 token 做局部最大池化以聚类相邻位置，"
    "再选取 top-K 的重要位置，从而在保持信息的同时避免选出过于密集的相邻 token。"
    "这些方法通常都是两阶段式的：先在 prefill 阶段收集注意力分数，再据此决定保留哪些 token，"
    "最后把不需要的 KV 块释放掉。"
    "在 PagedAttention 的实现里，KV Cache 被切成固定大小的 block（例如 16 个 token），"
    "每个 block 通过 block table 映射到物理块。因此 token 级别的淘汰最终要落到 block 级别："
    "只有当一个 block 里的所有 token 都被淘汰时，这个物理块才能真正被释放回空闲池；"
    "部分淘汰的 block 仍然保留，只是在下一次注意力计算里被掩码掉，以省下计算量。"
    "这种分层设计让内存节省和计算节省可以解耦：level 1 掩码省算力，level 2 释放块省显存。"
    "在实际部署中，压缩率通常取到一半左右，也就是把 KV Cache 砍掉一半，"
    "在吞吐和精度之间取得平衡。对于长文档问答、多轮对话和代码生成等场景，"
    "这类 KV 淘汰策略能够在几乎不损失质量的前提下显著降低显存占用，"
    "从而在同样的显卡上容纳更长的上下文或更大的并发。"
    "下面我们用真实的注意力分数来演示这一过程：先跑一次前向拿到注意力，"
    "再分别用 H2O、SnapKV 和 StreamingLLM 三种策略决定保留哪些 token，"
    "最后把它们映射到块级别，统计能释放多少个块、多少显存。"
    "从评测的角度看，衡量这些方法的好坏通常看两个维度：一个是显存或吞吐的提升，"
    "另一个是下游任务质量的保持程度。长文本问答、多文档摘要、代码库理解等任务对中间信息敏感，"
    "过度淘汰会导致关键事实丢失，因此注意力汇点和 heavy hitter 的保留尤为重要。"
    "在工程实现上，一个常见的做法是在 prefill 阶段顺便记录注意力分数，"
    "这一步几乎不增加额外开销，因为注意力矩阵本来就要计算；"
    "然后在 prefill 结束时做一次两阶段的淘汰决策，把分数交给策略函数，"
    "策略函数返回需要保留的 token 集合，再据此改写 block table 并释放物理块。"
    "释放块之后，空闲块会回到一个按淘汰优先级排序的空闲队列里，"
    "供后续的请求复用，这样整个服务的峰值显存就能被显著压低，"
    "同样的硬件可以容纳更大的 batch 或者更长的上下文。"
    "对于有前缀缓存的服务，淘汰时还需要注意块哈希和引用计数，"
    "只有当引用计数归零且不再被缓存命中时，块才能真正回收，"
    "否则共享块可能被其他请求继续使用，不能贸然释放。"
    "此外还有一类基于重要性的压缩方法，不是直接丢 token，"
    "而是对注意力分数低的 token 做有损量化或者合并，"
    "它们可以和淘汰策略结合，进一步压缩显存。"
    "总之，KV Cache 淘汰是一套在两阶段假设下工作的机制，"
    "它的核心组件包括分数采集、策略决策、块级映射和物理释放四个环节。"
    "下面这个演示就把这四个环节完整串起来，用真实模型的真实注意力分数，"
    "演示 H2O、SnapKV 和 StreamingLLM 三种策略在同一段文本上的不同保留结果。"
)


def build_h2o_scores(attentions, n_layers: int = 4) -> torch.Tensor:
    """Average attention received by each key token over the last ``n_layers``.

    ``attentions`` is the tuple returned by ``output_attentions=True``; each
    element is ``[..., seq_q, seq_k]``. We sum over every leading dim (batch,
    heads, and query positions), leaving a per-key-token importance ``[seq_k]``.
    """
    per_layer = []
    for t in attentions[-n_layers:]:
        t = t.float()
        per_layer.append(t.sum(dim=tuple(range(t.dim() - 1))))    # [S]
    scores = torch.stack(per_layer).mean(dim=0)                   # [S]
    return scores / scores.sum()


def build_snapkv_scores(attentions, window: int, n_layers: int = 4) -> torch.Tensor:
    """Attention received by each key from the last ``window`` queries.

    SnapKV derives importance from the observation window (recent queries);
    the pooling + top-K is done by :class:`SnapKVPolicy`.
    """
    per_layer = []
    for t in attentions[-n_layers:]:
        t = t.float()                                             # [..., S_q, S_k]
        obs = t[..., -window:, :]                                 # [..., window, S_k]
        per_layer.append(obs.sum(dim=tuple(range(obs.dim() - 1))))  # [S]
    scores = torch.stack(per_layer).mean(dim=0)                   # [S]
    return scores / scores.sum()


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
    bytes_per_elem = 2  # fp16
    return n_layers * n_kv_heads * head_dim * 2 * bytes_per_elem  # K + V


def block_savings(result, seq_len: int, block_size: int, bytes_per_token: int):
    """Map a token-level keep/evict set to freed blocks + bytes."""
    block_ids = torch.arange((seq_len + block_size - 1) // block_size, dtype=torch.int64)
    plan = plan_block_eviction(result, block_size, block_ids, seq_len)
    freed_blocks = len(plan.freed_block_ids)
    freed_tokens = freed_blocks * block_size
    return plan, freed_blocks, freed_tokens * bytes_per_token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-ratio", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--out", default="kv_eviction_demo.png")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model = model.to("cuda")
    model.set_attn_implementation("eager")
    model.eval()

    inputs = tok(PROMPT, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"sequence length = {seq_len} tokens", flush=True)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    attn = out.attentions  # tuple of [1,H,S,S]

    h2o_scores = build_h2o_scores(attn)
    snap_scores = build_snapkv_scores(attn, window=min(args.window, seq_len // 4))

    budget = max(1, int(seq_len * args.budget_ratio))
    h2o = H2OPolicy(window_size=args.window).select(h2o_scores.cpu(), seq_len, budget)
    snap = SnapKVPolicy(window_size=args.window).select(snap_scores.cpu(), seq_len, budget)
    stream = StreamingLLMPolicy(window_size=args.window).select(None, seq_len, budget)

    bpt = kv_bytes_per_token(model.config)
    h2o_plan, h2o_blocks, h2o_bytes = block_savings(h2o, seq_len, BLOCK_SIZE, bpt)
    snap_plan, snap_blocks, snap_bytes = block_savings(snap, seq_len, BLOCK_SIZE, bpt)
    _, stream_blocks, stream_bytes = block_savings(stream, seq_len, BLOCK_SIZE, bpt)

    # ---- real vLLM BlockPool: prove the mutation path frees blocks ----
    from vllm.v1.core.block_pool import BlockPool

    n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = BlockPool(num_gpu_blocks=n_blocks + 1, enable_caching=False, hash_block_size=BLOCK_SIZE)
    blocks = pool.get_new_blocks(n_blocks)
    free_before = pool.get_num_free_blocks()
    mgr = SimpleNamespace(req_to_blocks={"req-1": blocks},
                          _null_block=pool.null_block, block_pool=pool)
    kcm = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[mgr]))
    free_evicted_blocks(kcm, "req-1", h2o_plan.evicted_block_indices)
    free_after = pool.get_num_free_blocks()

    total_kv = seq_len * bpt
    print("\n" + "=" * 64)
    print(f"KV-cache eviction demo  (seq_len={seq_len}, budget={budget} = {budget/seq_len:.0%})")
    print("=" * 64)
    print(f"per-token KV   : {fmt_bytes(bpt)}")
    print(f"baseline KV    : {fmt_bytes(total_kv)}")
    for name, kept, freed_b, freed_bytes in [
        ("H2O", h2o.kept, h2o_blocks, h2o_bytes),
        ("SnapKV", snap.kept, snap_blocks, snap_bytes),
        ("StreamingLLM", stream.kept, stream_blocks, stream_bytes),
    ]:
        print(f"{name:<14}: keep {kept:>5} tokens ({kept/seq_len:>5.0%})  "
              f"free {freed_b:>4} blocks = {fmt_bytes(freed_bytes)} saved")
    print(f"real BlockPool: free {free_before} -> {free_after} "
          f"(freed {free_after - free_before} blocks)")
    print("note: evicted tokens in partially-kept blocks are masked (Level 1); "
          "only fully-evicted blocks are freed (Level 2)")

    # ---- visualization ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def strip(ax, res, y, label):
        keep = set(res.keep_indices.tolist())
        colors = ["#2ecc71" if i in keep else "#e74c3c" for i in range(seq_len)]
        ax.scatter(range(seq_len), [y] * seq_len, c=colors, s=6, marker="s", linewidths=0)
        ax.text(seq_len + 20, y, f"{label}  keep={res.kept}", va="center", fontsize=9)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    xs = torch.arange(seq_len)
    axes[0].fill_between(xs, h2o_scores.cpu().numpy(), color="#3498db", alpha=0.6, label="H2O importance")
    axes[0].set_ylabel("importance")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"KV eviction — {MODEL}  (seq={seq_len}, budget={budget} = {args.budget_ratio:.0%})")

    strip(axes[1], h2o, 0, "H2O")
    strip(axes[2], snap, 0, "SnapKV")
    for ax in axes[1:]:
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.grid(False)
    axes[2].set_xlabel("token position")

    fig.suptitle("green = keep, red = evict", fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nsaved figure -> {args.out}")
    print("DEMO_OK")


if __name__ == "__main__":
    main()
