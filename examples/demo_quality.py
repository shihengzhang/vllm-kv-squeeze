"""Quality axis: perplexity with full vs H2O-evicted KV cache.

Closes the loop left open by :mod:`demo_scaled` (memory + speed). We use a
coherent multi-topic Chinese document as context, derive the H2O keep/evict
decision from the last layer's real attention, then compare perplexity on a
held-out continuation with the FULL context vs with evicted tokens masked.

NOTE: the model must be loaded in bf16 (not fp16) — eager attention overflows
fp16 for long sequences (NaN logits); bf16 has fp32's exponent range and is safe.

Run: ``python examples/demo_quality.py --budget-ratio 0.5``
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from vllm_kv_squeeze.policy import H2OPolicy

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_MODEL = os.path.expanduser("~/models/Qwen2.5-7B-Instruct")

CONTEXT = (
    "大语言模型的推理部署是一个典型的系统工程问题，涉及显存、延迟、吞吐和成本之间的多重权衡。"
    "在实际服务中，模型权重往往以半精度或更低的精度加载，而动态生成的键值缓存则随着请求长度不断增长，"
    "很快成为显存的主要占用者。为了在高并发下保持稳定，服务系统需要精心设计内存管理策略。"
    "PagedAttention 借鉴了操作系统的分页思想，把键值缓存切分成固定大小的块，用块表把逻辑位置映射到物理块，"
    "从而避免了显存碎片化，并支持多个请求共享同一批物理块。这种设计显著提高了显存利用率。"
    "然而仅仅分页还不够，当上下文很长时，键值缓存的绝对大小仍然可观，于是出现了各类压缩与淘汰策略。"
    "研究者观察到，尽管注意力在理论上可以关注所有历史位置，但实际的注意力质量并不均匀，"
    "少数几个开头的 token 会吸引大量注意力，这一现象被称为注意力汇点。"
    "基于这一观察，StreamingLLM 提出只保留注意力汇点和最近的一段窗口，丢弃中间的大部分 token，"
    "从而把显存占用控制在一个与序列长度无关的常数级别。"
    "H2O 则进一步发现，在注意力汇点和最近窗口之外，中间还存在少数持续获得较高累计注意力的重要 token，"
    "它们承载着长程依赖中的关键信息，因此应当与汇点和窗口一起被保留。"
    "SnapKV 采用了另一种思路，它从观察窗口的注意力出发，对相邻位置做局部最大池化来聚类，"
    "再选取 top-K 的重要位置，从而在保留信息的同时避免选出过于密集的相邻 token。"
    "这些方法的共同点在于，它们都需要在 prefill 阶段先获得注意力信息，再据此做出保留与淘汰的决策，"
    "因此都属于两阶段式的方法。"
    "除了淘汰，量化也是降低推理成本的重要手段，将权重和激活从十六位浮点压缩到八位甚至四位整数，"
    "可以在精度损失可控的前提下大幅减少显存带宽需求，提升吞吐。"
    "批处理方面，continuous batching 允许新请求在旧请求尚未结束时动态加入，"
    "而不是等待整个批次完成，从而显著提高 GPU 的利用率。"
    "调度策略决定了哪些请求优先获得计算资源，常见的有先到先服务、抢占式调度和基于优先级的调度等。"
    "投机解码则利用一个小模型快速生成候选 token，再由大模型并行验证，"
    "在不改变输出分布的前提下降低了解码的延迟。"
    "前缀缓存把具有相同前缀的多个请求的键值缓存复用起来，避免了重复计算。"
    "面对超长上下文，研究者还提出了旋转位置编码的扩展方法以及各种稀疏注意力机制，"
    "试图在保持质量的同时把计算复杂度从平方级别降下来。"
    "综合来看，高效的推理系统通常不是依赖单一技术，而是把分页内存、键值淘汰、量化、批处理、"
    "投机解码和前缀缓存等手段组合起来，针对具体的工作负载进行调优。"
    "下面我们通过一个具体的例子，量化键值淘汰对语言模型困惑度的影响。"
)

CONTINUATION = (
    "困惑度是衡量语言模型质量的一个常用指标，它反映了模型对一段文本的预测能力。"
    "困惑度越低，说明模型对文本的建模越准确。"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-ratio", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--model", default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else MODEL)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    model.set_attn_implementation("eager")
    model.eval()

    ctx = tok(CONTEXT, return_tensors="pt").input_ids.to("cuda")
    cont = tok(CONTINUATION, return_tensors="pt").input_ids.to("cuda")
    C, N = ctx.shape[1], cont.shape[1]
    print(f"context={C} tokens, continuation={N} tokens", flush=True)

    # --- real last-layer attention -> H2O keep/evict on the context ---
    with torch.no_grad():
        out = model(ctx, output_attentions=True)
    w = out.attentions[-1].float()  # [1, H, C, C]
    scores = w.sum(dim=tuple(range(w.dim() - 1)))  # sum heads+queries -> [C]
    scores = scores / scores.sum()

    budget = max(1, int(C * args.budget_ratio))
    h2o = H2OPolicy(window_size=args.window).select(scores.cpu(), C, budget)
    evict = h2o.evict_indices.to("cuda")

    # --- perplexity on the continuation: full vs evicted (masked) ---
    ids = torch.cat([ctx, cont], dim=1)
    labels = ids.clone()
    labels[:, :C] = -100  # loss only on the continuation

    with torch.no_grad():
        out_full = model(ids, labels=labels)
    ppl_full = torch.exp(out_full.loss).item()

    mask = torch.ones_like(ids)
    mask[:, evict] = 0
    with torch.no_grad():
        out_evict = model(ids, labels=labels, attention_mask=mask)
    ppl_evict = torch.exp(out_evict.loss).item()

    print("\n" + "=" * 60)
    print(f"H2O quality @ {args.model.split('/')[-1]}  (context={C}, budget={budget}={budget/C:.0%})")
    print("=" * 60)
    print(f"evicted tokens   : {h2o.evicted} / {C}")
    print(f"full-context ppl : {ppl_full:.3f}")
    print(f"evicted-KV  ppl  : {ppl_evict:.3f}")
    print(f"relative gap     : {ppl_evict / ppl_full:.3f}x")
    ratio = ppl_evict / ppl_full
    verdict = "negligible" if ratio < 1.05 else "moderate" if ratio < 1.15 else "large"
    print(f"quality impact   : {verdict}")
    print("DEMO_QUALITY_OK")


if __name__ == "__main__":
    main()
