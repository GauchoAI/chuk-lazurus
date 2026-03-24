#!/usr/bin/env python3
"""
Experiment B — Novelty Density Across Document Types

Measures per-token surprise (cross-entropy rank) across 5 document types
to determine what fraction is parametric vs novel.

Uses the chuk-mlx KV-direct generator for efficient forward passes.

Usage:
    cd /Users/christopherhay/chris-source/apollo-demo
    uv run --project /Users/christopherhay/chris-source/chuk-mlx \
        python demo_b_novelty_density.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Document excerpts (~200-400 tokens each)
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "wikipedia": {
        "label": "Wikipedia (Apollo 11)",
        "text": (
            "Apollo 11 was the American spaceflight that first landed humans "
            "on the Moon. Commander Neil Armstrong and lunar module pilot Buzz "
            "Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at "
            "20:17 UTC, and Armstrong became the first person to step onto the "
            "Moon's surface six hours and 39 minutes later, on July 21 at "
            "02:56 UTC. Aldrin joined him 19 minutes later, and they spent "
            "about two and a quarter hours together exploring the site they "
            "had named Tranquility Base upon landing. Armstrong and Aldrin "
            "collected 47.5 pounds of lunar material to bring back to Earth "
            "as pilot Michael Collins flew the Command Module Columbia in "
            "lunar orbit, and were on the Moon's surface for 21 hours, 36 "
            "minutes before lifting off to rejoin Columbia."
        ),
        "expected_novel_pct": "5-10%",
    },
    "fiction": {
        "label": "Fiction (original passage)",
        "text": (
            "The old man sat in the corner of the bar, nursing a whiskey that "
            "had long gone warm. He watched the door, waiting for someone who "
            "might never come. Outside, the rain hammered against the windows "
            "of the Thornfield Arms, turning the cobblestones into mirrors. "
            "Elena arrived at half past nine, her red coat dripping, the "
            "crumpled letter still clutched in her left hand. She sat across "
            "from Marcus without a word. The bartender, a heavy woman named "
            "Gretel, brought two fresh glasses without being asked. 'I found "
            "the box,' Elena said. 'Fourteen paces from the stone wall, just "
            "like you said. But it was empty, Marcus. It was empty.' He "
            "closed his eyes. 'That's what I was afraid of.'"
        ),
        "expected_novel_pct": "40-55%",
    },
    "ml_paper": {
        "label": "ML Paper (abstract)",
        "text": (
            "We present a novel approach to long-context language modeling "
            "using sparse semantic indexing. Our method achieves comparable "
            "performance to full-attention transformers while reducing memory "
            "requirements by orders of magnitude. Experimental results on the "
            "PG-19 and SCROLLS benchmarks demonstrate that our sparse index "
            "retains 94.7% of retrieval accuracy while compressing the context "
            "window by a factor of 300. Previous work on efficient attention "
            "mechanisms, including Longformer and BigBird, addressed quadratic "
            "scaling but required architectural modifications. Our approach "
            "operates post-hoc on any pretrained model. We train a linear "
            "probe on the residual stream at layer 14 and find that entity "
            "identity is encoded with 100% accuracy in a 10-dimensional "
            "subspace. Table 3 shows the ablation results: removing the "
            "surprise threshold increases index size by 340% while improving "
            "recall by only 2.1 percentage points."
        ),
        "expected_novel_pct": "20-30%",
    },
    "transcript": {
        "label": "Transcript (Apollo 11 EVA)",
        "text": (
            "04 14 09 05 CC: Roger. How's it going? "
            "04 14 09 14 CMP: Great. "
            "04 14 15 06 LMP: It's hard saying what size pace might be. "
            "I think it's the one that I'm using now would get rather "
            "tiring after several hundred but this may be a function of "
            "this suit as well as lack of gravity forces. "
            "04 14 15 47 CC: Tranquility Base, this is Houston. Could we "
            "get both of you on the camera for a minute, please? "
            "04 14 16 09 CC: Neil and Buzz, the President of the United "
            "States is in his office now and would like to say a few "
            "words to you. Over. "
            "04 14 16 23 CDR: That would be an honor. "
            "04 14 20 06 LMP: Houston, it's very interesting to note that "
            "when I kick my foot with no atmosphere here, and this gravity "
            "they seem to leave, and most of them have about the same "
            "angle of departure and velocity."
        ),
        "expected_novel_pct": "25-35%",
    },
    "proprietary": {
        "label": "Proprietary Memo",
        "text": (
            "INTERNAL MEMO — Project Lighthouse Q3 Review. Attendees: Sarah "
            "Chen (VP Engineering), Marcus Webb (Lead Architect), Diana "
            "Okafor (QA Director). Decision: migrate all legacy services to "
            "Valkey by November 15th. The Lighthouse backend latency target "
            "was revised from 200ms to 145ms after the October 14th incident "
            "involving customer outages across the APAC region. Marcus "
            "proposed replacing the Redis cache with a Valkey cluster, "
            "estimating a cost saving of $47,200 per quarter. Diana flagged "
            "3 critical regressions in build 9.4.2-rc1: the auth token "
            "refresh loop, the CSV export truncation at 50,000 rows, and "
            "the timezone offset bug affecting scheduled reports in UTC+8 "
            "zones. Action item: Diana to schedule regression testing for "
            "build 9.4.3 with the new Valkey backend by October 28th."
        ),
        "expected_novel_pct": "60-80%",
    },
}


def compute_per_token_surprise(
    kv_gen, tokenizer, text: str, skip_first: int = 8
) -> dict:
    """Compute per-token surprise rank for a text.

    Returns dict with per-token ranks and summary statistics.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(token_ids)

    if n_tokens < 2:
        return {"error": "text too short"}

    ids = mx.array(token_ids)[None]  # (1, seq_len)
    logits, _kv = kv_gen.prefill(ids)
    mx.eval(logits)

    logits_f32 = logits[0].astype(mx.float32)  # (seq_len, vocab)
    mx.eval(logits_f32)

    # For each position i, logits[i] predicts token at position i+1.
    # Compute rank of actual next token (0 = top prediction, higher = more surprising).
    n_score = n_tokens - 1 - skip_first
    if n_score <= 0:
        return {"error": "text too short after skip"}

    actual_ids = mx.array(token_ids[skip_first + 1:])
    logits_slice = logits_f32[skip_first:skip_first + n_score]
    actual_logits = logits_slice[mx.arange(n_score), actual_ids]

    # Rank = number of tokens with higher logit
    ranks = mx.sum(logits_slice > actual_logits[:, None], axis=1)
    mx.eval(ranks)
    ranks_np = np.array(ranks.tolist(), dtype=np.int32)

    # Also get probabilities for the actual tokens
    probs = mx.softmax(logits_slice, axis=-1)
    actual_probs = probs[mx.arange(n_score), actual_ids]
    mx.eval(actual_probs)
    probs_np = np.array(actual_probs.tolist(), dtype=np.float32)

    # Decode tokens for display
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids[skip_first + 1:]]

    # Classify: rank 0-2 = parametric, 3-50 = semi-parametric, 50+ = novel
    n_parametric = int(np.sum(ranks_np <= 2))
    n_semi = int(np.sum((ranks_np > 2) & (ranks_np <= 50)))
    n_novel = int(np.sum(ranks_np > 50))

    return {
        "n_tokens": n_tokens,
        "n_scored": n_score,
        "ranks": ranks_np,
        "probs": probs_np,
        "decoded_tokens": decoded_tokens,
        "n_parametric": n_parametric,
        "n_semi": n_semi,
        "n_novel": n_novel,
        "pct_parametric": 100 * n_parametric / n_score,
        "pct_semi": 100 * n_semi / n_score,
        "pct_novel": 100 * n_novel / n_score,
        "median_rank": float(np.median(ranks_np)),
        "mean_rank": float(np.mean(ranks_np)),
        "max_rank": int(np.max(ranks_np)),
    }


def print_surprise_histogram(result: dict, label: str) -> None:
    """Print a visual histogram of surprise distribution."""
    ranks = result["ranks"]
    bins = [0, 1, 3, 10, 50, 200, 1000, 10000, 300000]
    labels = ["top-1", "top-3", "top-10", "top-50", "top-200", "top-1K", "top-10K", "10K+"]

    print(f"\n  {BOLD}{label}{RESET}")
    print(f"  Tokens: {result['n_scored']} scored "
          f"(median rank: {result['median_rank']:.0f})")

    for i in range(len(bins) - 1):
        count = int(np.sum((ranks >= bins[i]) & (ranks < bins[i + 1])))
        pct = 100 * count / result["n_scored"]
        bar = "█" * int(pct / 2)
        color = GREEN if i < 3 else (YELLOW if i < 5 else RED)
        print(f"  {labels[i]:>8s}: {color}{bar} {pct:5.1f}% ({count}){RESET}")

    print(f"\n  Parametric (rank ≤2):   {GREEN}{result['pct_parametric']:5.1f}%{RESET}")
    print(f"  Semi-param (rank 3-50): {YELLOW}{result['pct_semi']:5.1f}%{RESET}")
    print(f"  Novel (rank >50):       {RED}{result['pct_novel']:5.1f}%{RESET}")


def print_novel_tokens(result: dict, n: int = 15) -> None:
    """Print the most surprising tokens."""
    top_idx = np.argsort(result["ranks"])[::-1][:n]
    print(f"\n  {BOLD}Top {n} most surprising tokens:{RESET}")
    for idx in top_idx:
        tok = result["decoded_tokens"][idx]
        rank = result["ranks"][idx]
        prob = result["probs"][idx]
        color = RED if rank > 200 else (YELLOW if rank > 50 else GREEN)
        print(f"    {color}rank {rank:>6,d}  p={prob:.4f}  → {repr(tok)}{RESET}")


def main():
    print(f"\n{BOLD}{'═' * 60}")
    print("  EXPERIMENT B — Novelty Density Across Document Types")
    print(f"{'═' * 60}{RESET}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n{DIM}Loading model...{RESET}")
    from chuk_lazarus.inference.unified import UnifiedPipeline
    from chuk_lazarus.inference.context.kv_generator import make_kv_generator

    pipeline = UnifiedPipeline.from_pretrained(
        "google/gemma-3-4b-it", verbose=False
    )
    kv_gen = make_kv_generator(pipeline.model)
    tokenizer = pipeline.tokenizer
    print(f"  Model loaded: {pipeline.family_type.value}")

    # ------------------------------------------------------------------
    # Phase B1: Surprise measurement per document
    # ------------------------------------------------------------------
    all_results = {}

    for doc_key, doc in DOCUMENTS.items():
        t0 = time.time()
        result = compute_per_token_surprise(kv_gen, tokenizer, doc["text"])
        elapsed = time.time() - t0

        all_results[doc_key] = result
        print_surprise_histogram(result, f"{doc['label']} (expected novel: {doc['expected_novel_pct']})")
        print_novel_tokens(result)
        print(f"  {DIM}Computed in {elapsed:.1f}s{RESET}")

    # ------------------------------------------------------------------
    # Phase B2: Summary table
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{'═' * 60}")
    print("  NOVELTY DENSITY SUMMARY")
    print(f"{'═' * 60}{RESET}")
    print()
    print(f"  {'Document':<25s} {'Tokens':>6s} {'Parametric':>11s} {'Semi':>8s} {'Novel':>8s} {'Expected':>10s}")
    print(f"  {'─' * 70}")
    for doc_key, doc in DOCUMENTS.items():
        r = all_results[doc_key]
        print(
            f"  {doc['label'][:25]:<25s} "
            f"{r['n_scored']:>6d} "
            f"{GREEN}{r['pct_parametric']:>9.1f}%{RESET}  "
            f"{YELLOW}{r['pct_semi']:>6.1f}%{RESET}  "
            f"{RED}{r['pct_novel']:>6.1f}%{RESET}  "
            f"{DIM}{doc['expected_novel_pct']:>10s}{RESET}"
        )

    # ------------------------------------------------------------------
    # Phase B4: Projected index sizes
    # ------------------------------------------------------------------
    typical_sizes = {
        "wikipedia": 5_000,
        "fiction": 100_000,
        "ml_paper": 10_000,
        "transcript": 370_000,
        "proprietary": 20_000,
    }

    print(f"\n{BOLD}  Projected Index Sizes (full documents){RESET}")
    print(f"  {'─' * 70}")
    print(f"  {'Document':<25s} {'Full size':>10s} {'Novel %':>8s} {'Index':>10s} {'Compression':>12s}")
    print(f"  {'─' * 70}")

    for doc_key, doc in DOCUMENTS.items():
        r = all_results[doc_key]
        full_tokens = typical_sizes[doc_key]
        novel_pct = r["pct_novel"] / 100
        # Estimate: ~3 index tokens per novel token cluster, ~4 bytes per token
        novel_tokens = int(full_tokens * novel_pct)
        index_tokens = novel_tokens // 3  # rough clustering
        index_bytes = index_tokens * 4
        kv_bytes = full_tokens * 2 * 34 * 4 * 256 * 2  # per-token KV for gemma-3-4b
        compression = kv_bytes / max(index_bytes, 1)

        def fmt_bytes(n):
            if n < 1024: return f"{n} B"
            if n < 1024**2: return f"{n/1024:.1f} KB"
            if n < 1024**3: return f"{n/1024**2:.1f} MB"
            return f"{n/1024**3:.1f} GB"

        print(
            f"  {doc['label'][:25]:<25s} "
            f"{full_tokens:>8,d}t  "
            f"{RED}{r['pct_novel']:>6.1f}%{RESET}  "
            f"{fmt_bytes(index_bytes):>10s}  "
            f"{GREEN}{compression:>10,.0f}x{RESET}"
        )

    # ------------------------------------------------------------------
    # Phase B5: Apollo 11 transcript full-scale surprise
    # ------------------------------------------------------------------
    lean_path = Path("apollo11_lean")
    if lean_path.exists():
        surprise_path = lean_path / "surprise.npz"
        if surprise_path.exists():
            print(f"\n{BOLD}  Apollo 11 Full Transcript Surprise (from prefill){RESET}")
            d = np.load(str(surprise_path))
            max_ranks = d["max_rank"]
            print(f"  Windows: {len(max_ranks)}")
            print(f"  Median max-rank: {np.median(max_ranks):,.0f}")
            print(f"  Mean max-rank: {np.mean(max_ranks):,.0f}")
            print(f"  Max max-rank: {np.max(max_ranks):,.0f}")

            # Distribution of max-surprise per window
            low = int(np.sum(max_ranks < 1000))
            mid = int(np.sum((max_ranks >= 1000) & (max_ranks < 50000)))
            high = int(np.sum(max_ranks >= 50000))
            total = len(max_ranks)
            print(f"  Low surprise windows (<1K):  {low}/{total} ({100*low/total:.0f}%)")
            print(f"  Mid surprise windows:        {mid}/{total} ({100*mid/total:.0f}%)")
            print(f"  High surprise windows (>50K): {high}/{total} ({100*high/total:.0f}%)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_data = {
        "experiment": "B_novelty_density",
        "model": "google/gemma-3-4b-it",
    }
    for doc_key in DOCUMENTS:
        r = all_results[doc_key]
        save_data[doc_key] = {
            "n_scored": r["n_scored"],
            "pct_parametric": round(r["pct_parametric"], 1),
            "pct_semi": round(r["pct_semi"], 1),
            "pct_novel": round(r["pct_novel"], 1),
            "median_rank": r["median_rank"],
            "mean_rank": round(r["mean_rank"], 1),
        }

    out_path = "results_experiment_b.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  {DIM}Results saved to {out_path}{RESET}\n")


if __name__ == "__main__":
    main()
