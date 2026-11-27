"""Prefix-cache-aware eviction: the eviction-efficiency gap (analysis).

Under vLLM prefix caching, a physical KV block can be shared by M requests
(ref_cnt = M). ``BlockPool.free_blocks`` only returns a block to the free pool
when its ref_cnt reaches 0. So:

  * evicting a unique block (ref_cnt=1) frees it immediately (1 eviction -> 1 free);
  * evicting a shared block (ref_cnt=M) only decrements ref_cnt; the block is NOT
    freed until ALL M requests release it.

A single-request importance policy ignores ref_cnt, so it "wastes" evictions on
shared blocks (no memory freed, but quality is still lost). A ref_cnt-aware
policy evicts ref_cnt=1 blocks first, so every eviction frees memory.

This script quantifies that gap analytically (expected values) and plots
"eviction efficiency = freed blocks / evicted blocks" vs the shared-prefix
fraction, for the two policies.
"""

from __future__ import annotations

import math


def efficiency_single(M: int, P: int, S: int, K: int) -> float:
    """Expected freed-blocks / evicted-blocks for a ref_cnt-blind policy.

    Each of M requests evicts K of its P+S blocks uniformly (importance is
    assumed uniform so the ONLY effect under study is ref_cnt awareness).
    """
    total = P + S
    p_evict = K / total
    freed_unique = M * S * p_evict                     # each unique block frees iff its owner evicts it
    freed_shared = P * (p_evict ** M)                  # shared block frees iff ALL M owners evict it
    return (freed_unique + freed_shared) / (M * K)


def efficiency_aware(M: int, P: int, S: int, K: int) -> float:
    """Expected efficiency for a ref_cnt-aware policy (evict ref_cnt=1 first)."""
    if K <= S:
        return 1.0                                     # all K evictions hit unique blocks
    # K > S: S unique blocks + (K-S) shared blocks per request
    unique = M * S
    shared_evicted = M * (K - S)
    # shared block frees iff all M requests also evict it among their (K-S)
    p_shared_each = (K - S) / P
    shared_freed = P * (p_shared_each ** M)
    return (unique + shared_freed) / (unique + shared_evicted)


def main() -> None:
    M, K = 8, 16                       # 8 concurrent requests, evict 16 blocks each
    shared_fractions = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    print(f"M={M} requests, K={K} blocks evicted each\n")
    print("shared_frac   single(re-blind)   aware(ref_cnt)")
    print("-" * 50)
    rows = []
    for f in shared_fractions:
        P = int(f * 100)               # shared prefix blocks
        S = 100 - P                    # unique suffix blocks
        es = efficiency_single(M, P, S, K)
        ea = efficiency_aware(M, P, S, K)
        rows.append((f, es, ea))
        print(f"  {f:5.2f}        {es:.3f}              {ea:.3f}")

    # save plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"]})
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fs = [r[0] for r in rows]
        ax.plot(fs, [r[1] for r in rows], "o-", color="#e07b39", lw=2, ms=7,
                label="ref_cnt-blind (single-request)")
        ax.plot(fs, [r[2] for r in rows], "s-", color="#2ecc71", lw=2, ms=7,
                label="ref_cnt-aware (ours)")
        ax.set_xlabel("Shared-prefix fraction of each request")
        ax.set_ylabel("Eviction efficiency (freed / evicted)")
        ax.set_title(f"Prefix-cache-aware eviction (M={M} requests, K={K} evictions each)", fontsize=11.5, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig("prefix_eviction_efficiency.png", dpi=130)
        print("\nsaved prefix_eviction_efficiency.png")
    except ImportError:
        pass

    print("\nANALYSIS_DONE")


if __name__ == "__main__":
    main()
