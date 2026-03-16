#!/usr/bin/env python3
"""
Experiment C — The Real Apollo 11 Demo

Full pipeline: pre-built checkpoint library → replay specific windows →
compare parametric-only vs window-replay vs sparse-index generation.

Tests single-fact retrieval (C3) and cross-window synthesis (C4).

Usage:
    cd /Users/christopherhay/chris-source/apollo-demo
    uv run --project /Users/christopherhay/chris-source/chuk-mlx \
        python demo_c_apollo11_real.py

    # Specific library path:
    uv run --project /Users/christopherhay/chris-source/chuk-mlx \
        python demo_c_apollo11_real.py --library apollo11_ctx_512_full
"""

from __future__ import annotations

import argparse
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


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# Queries — facts from different parts of the transcript
# ---------------------------------------------------------------------------

# Each query has:
#   - question: natural language question
#   - window_ids: which windows to replay for Mode 4 retrieval
#   - expected: substring to look for in correct answer
#   - fact_type: parametric, partially_parametric, or novel

SINGLE_FACT_QUERIES = [
    {
        "id": "commander",
        "question": "Who was the mission commander of Apollo 11?",
        "window_ids": [],  # No window needed — purely parametric
        "expected": "Armstrong",
        "fact_type": "parametric",
    },
    {
        "id": "bruce_audio",
        "question": (
            "During the early launch phase of Apollo 11, what did the "
            "commander say about how the capsule communicator sounded?"
        ),
        "window_ids": [2],  # Launch phase
        "expected": "living room",
        "fact_type": "novel",
    },
    {
        "id": "particle_trajectory",
        "question": (
            "What did Buzz Aldrin observe about the trajectory of particles "
            "when he kicked the lunar surface during the EVA?"
        ),
        "window_ids": [464],  # EVA dust observation
        "expected": "angle",
        "fact_type": "novel",
    },
    {
        "id": "nixon_call",
        "question": (
            "Who introduced the President's phone call to the astronauts on "
            "the lunar surface, and what exactly did they say?"
        ),
        "window_ids": [462],  # Nixon call
        "expected": "President",
        "fact_type": "partially_parametric",
    },
    {
        "id": "strange_noises",
        "question": (
            "What did Houston say about strange noises on the downlink "
            "during the return trip?"
        ),
        "window_ids": [634],  # Strange noises
        "expected": "friends",
        "fact_type": "novel",
    },
    {
        "id": "aldrin_home",
        "question": (
            "What did Aldrin say when he first stepped onto the ladder "
            "to go down to the surface?"
        ),
        "window_ids": [449],  # Aldrin on ladder
        "expected": "home",
        "fact_type": "novel",
    },
]

# Sparse index entries — hand-crafted from transcript windows
SPARSE_INDEX = """Key moments from the Apollo 11 air-to-ground transcript:

[LAUNCH 00:05:35] CDR to CC Bruce McCandless: "You sure sound clear down there, Bruce. Sounds like you're sitting in your living room." CC: "Oh, thank you. You all are coming through beautifully, too."

[TRANSIT 01:03:13] CMP Collins describing Earth: "The view is just beautiful. It's out of this world." Sees Mediterranean islands — Majorca, Sardinia, Corsica. Haze over Italian peninsula.

[EVA 04:13:42] LMP Aldrin on surface: "(Laughter) That's our home for the next couple of hours and we want to take good care of it." Hopping between steps very simple, walking comfortable.

[EVA 04:14:15] LMP on movement: "It's hard saying what size pace might be. I think it's the one I'm using now - would get rather tiring after several hundred but this may be a function of this suit as well as lack of gravity forces."

[EVA 04:14:20] LMP kicking dust: "Houston, it's very interesting to note that when I kick my foot with no atmosphere here, and this gravity they seem to leave, and most of them have about the same angle of departure and velocity. From where I stand, a large portion of them will impact at a certain distance out."

[EVA 04:14:16] CC introduces Nixon: "Neil and Buzz, the President of the United States is in his office now and would like to say a few words to you. Over." CDR: "That would be an honor."

[POST-EVA 06:09:53] Strange noises on downlink. CC: "Apollo 11, Houston. You sure you don't have anybody else in there with you?" CDR: "Where do the White Team go off hours anyway?" CC: "We had some strange noises coming down on the downlink, and it sounded like you had some friends up there."

[RETURN 07:09:33] Crew broadcast. CDR quoted: "This is a small step for a man, but a great leap for mankind." CMP Collins: tribute to American workmen and test teams.
"""

CROSS_WINDOW_QUERIES = [
    {
        "id": "amusing_moments",
        "question": "Find 3 amusing or light-hearted moments from the Apollo 11 mission.",
        "max_tokens": 400,
    },
    {
        "id": "key_moments",
        "question": "What were the 5 key moments of the Apollo 11 mission based on the transcript?",
        "max_tokens": 400,
    },
    {
        "id": "crew_dynamics",
        "question": "Based on the transcript, describe the relationship and dynamics between the three crew members.",
        "max_tokens": 300,
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Experiment C — Apollo 11 Real Demo")
    p.add_argument(
        "--library", default="apollo11_ctx_512_full",
        help="Path to pre-built checkpoint library",
    )
    p.add_argument("--model", default="google/gemma-3-4b-it", help="Model ID")
    return p.parse_args()


def generate_parametric(pipeline, question: str, max_tokens: int = 100) -> str:
    """Generate with no document context — pure parametric recall."""
    from chuk_lazarus.inference.generation import GenerationConfig

    prompt = (
        "<start_of_turn>user\n"
        "You are answering questions about the Apollo 11 mission. "
        "Answer concisely.\n"
        f"{question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    config = GenerationConfig(max_new_tokens=max_tokens, temperature=0.0)
    result = pipeline.generate(prompt, config=config)
    return result.text.strip()


def generate_with_index(pipeline, question: str, index: str, max_tokens: int = 200) -> str:
    """Generate with sparse index prepended to the prompt."""
    from chuk_lazarus.inference.generation import GenerationConfig

    prompt = (
        "<start_of_turn>user\n"
        f"Here is a sparse index of notable moments from the Apollo 11 "
        f"air-to-ground transcript:\n\n{index}\n\n"
        f"Based on this index, {question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    config = GenerationConfig(max_new_tokens=max_tokens, temperature=0.0)
    result = pipeline.generate(prompt, config=config)
    return result.text.strip()


def generate_with_replay(
    engine, tokenizer, question: str, window_ids: list[int],
    library_name: str, max_tokens: int = 100,
) -> str:
    """Generate by replaying specific library windows as context."""
    from chuk_lazarus.inference.context.unlimited_engine import LibrarySource

    prompt = (
        "<start_of_turn>user\n"
        "Based on the preceding transcript, answer concisely.\n"
        f"{question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    query_ids = tokenizer.encode(prompt, add_special_tokens=False)

    sources = [LibrarySource(library_name=library_name, window_id=wid) for wid in window_ids]
    generated_ids = engine.generate_cross_library(
        query_ids,
        sources=sources,
        max_new_tokens=max_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    print(f"\n{BOLD}{'═' * 60}")
    print("  EXPERIMENT C — The Real Apollo 11 Demo")
    print(f"{'═' * 60}{RESET}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\n{DIM}Loading model...{RESET}")
    from chuk_lazarus.inference.unified import UnifiedPipeline
    from chuk_lazarus.inference.context.kv_generator import make_kv_generator
    from chuk_lazarus.inference.context.unlimited_engine import UnlimitedContextEngine
    from chuk_lazarus.inference.context.checkpoint_library import CheckpointLibrary

    pipeline = UnifiedPipeline.from_pretrained(args.model, verbose=False)
    tokenizer = pipeline.tokenizer
    print(f"  Model: {pipeline.family_type.value}")

    # ------------------------------------------------------------------
    # 2. Load checkpoint library
    # ------------------------------------------------------------------
    lib_path = Path(args.library)
    if not lib_path.exists():
        print(f"{RED}Error: Library not found: {lib_path}{RESET}")
        print(f"  Build it with: lazarus context prefill --input docs/apollo11_clean.txt --checkpoint {lib_path}")
        return

    print(f"\n{DIM}Loading checkpoint library: {lib_path}{RESET}")
    t0 = time.time()
    lib = CheckpointLibrary(lib_path)
    load_ms = (time.time() - t0) * 1000

    manifest = lib.manifest
    print(f"  Name:       {manifest.name}")
    print(f"  Tokens:     {manifest.total_tokens:,}")
    print(f"  Windows:    {manifest.num_windows}")
    print(f"  Checkpoint: {fmt_bytes(manifest.checkpoint_bytes)}")
    print(f"  Archive:    {fmt_bytes(manifest.archive_bytes)}")
    print(f"  Total:      {fmt_bytes(manifest.total_bytes)}")
    print(f"  Loaded in:  {load_ms:.0f}ms")

    # Compare to naive KV
    bytes_per_token = 2 * 34 * 4 * 256 * 2  # gemma-3-4b
    naive_kv = manifest.total_tokens * bytes_per_token
    compression = naive_kv / max(manifest.total_bytes, 1)
    print(f"\n  {BOLD}Naive KV:     {fmt_bytes(naive_kv)}")
    print(f"  Library:     {fmt_bytes(manifest.total_bytes)}")
    print(f"  Compression: {GREEN}{compression:.0f}x{RESET}")

    # ------------------------------------------------------------------
    # 3. Setup engine
    # ------------------------------------------------------------------
    engine = UnlimitedContextEngine(pipeline.model, pipeline.config, window_size=512)
    engine.load_library(lib)

    # ------------------------------------------------------------------
    # Phase C3: Single-Fact Retrieval
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{'═' * 60}")
    print("  Phase C3 — Single-Fact Retrieval")
    print(f"{'═' * 60}{RESET}")
    print(f"  {DIM}Comparing: Parametric | Sparse Index | Window Replay{RESET}")

    c3_results = []

    for q in SINGLE_FACT_QUERIES:
        print(f"\n  {BOLD}{q['id']}{RESET} ({q['fact_type']})")
        print(f"  Q: {q['question'][:80]}...")

        # 1. Parametric only
        t0 = time.time()
        parametric_answer = generate_parametric(pipeline, q["question"])
        param_ms = (time.time() - t0) * 1000

        # 2. Sparse index
        t0 = time.time()
        index_answer = generate_with_index(pipeline, q["question"], SPARSE_INDEX)
        index_ms = (time.time() - t0) * 1000

        # 3. Window replay (only if window IDs are specified)
        replay_answer = None
        replay_ms = 0
        if q["window_ids"]:
            t0 = time.time()
            replay_answer = generate_with_replay(
                engine, tokenizer, q["question"],
                q["window_ids"], manifest.name,
            )
            replay_ms = (time.time() - t0) * 1000

        # Score
        def has_expected(answer, expected):
            if not answer:
                return "—"
            return "✓" if expected.lower() in answer.lower() else "✗"

        p_mark = has_expected(parametric_answer, q["expected"])
        i_mark = has_expected(index_answer, q["expected"])
        r_mark = has_expected(replay_answer, q["expected"]) if replay_answer else "—"

        p_color = GREEN if p_mark == "✓" else RED
        i_color = GREEN if i_mark == "✓" else RED
        r_color = GREEN if r_mark == "✓" else (DIM if r_mark == "—" else RED)

        print(f"  {p_color}[{p_mark}] Parametric ({param_ms:.0f}ms):{RESET} {parametric_answer[:100]}")
        print(f"  {i_color}[{i_mark}] Index ({index_ms:.0f}ms):{RESET}      {index_answer[:100]}")
        if replay_answer:
            print(f"  {r_color}[{r_mark}] Replay ({replay_ms:.0f}ms):{RESET}     {replay_answer[:100]}")

        c3_results.append({
            "id": q["id"],
            "fact_type": q["fact_type"],
            "parametric": {"answer": parametric_answer[:200], "correct": p_mark, "ms": param_ms},
            "index": {"answer": index_answer[:200], "correct": i_mark, "ms": index_ms},
            "replay": {"answer": (replay_answer or "")[:200], "correct": r_mark, "ms": replay_ms},
        })

    # C3 summary
    print(f"\n{BOLD}  C3 Summary{RESET}")
    print(f"  {'─' * 50}")
    p_correct = sum(1 for r in c3_results if r["parametric"]["correct"] == "✓")
    i_correct = sum(1 for r in c3_results if r["index"]["correct"] == "✓")
    r_results = [r for r in c3_results if r["replay"]["correct"] != "—"]
    r_correct = sum(1 for r in r_results if r["replay"]["correct"] == "✓")

    print(f"  Parametric:   {p_correct}/{len(c3_results)}")
    print(f"  Sparse Index: {GREEN}{i_correct}/{len(c3_results)}{RESET}")
    print(f"  Window Replay:{GREEN}{r_correct}/{len(r_results)}{RESET} (of those with window IDs)")

    # ------------------------------------------------------------------
    # Phase C4: Cross-Window Synthesis
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{'═' * 60}")
    print("  Phase C4 — Cross-Window Synthesis")
    print(f"{'═' * 60}{RESET}")
    print(f"  {DIM}Comparing: Parametric vs Sparse Index{RESET}")

    c4_results = []

    for q in CROSS_WINDOW_QUERIES:
        print(f"\n  {BOLD}{q['id']}{RESET}")
        print(f"  Q: {q['question']}")

        # Parametric
        print(f"\n  {CYAN}[Parametric]{RESET}")
        t0 = time.time()
        param_answer = generate_parametric(pipeline, q["question"], max_tokens=q["max_tokens"])
        param_ms = (time.time() - t0) * 1000
        # Print first 300 chars
        for line in param_answer[:400].split("\n"):
            print(f"    {line}")
        print(f"    {DIM}({param_ms:.0f}ms){RESET}")

        # Sparse index
        print(f"\n  {GREEN}[Sparse Index]{RESET}")
        t0 = time.time()
        index_answer = generate_with_index(
            pipeline, q["question"], SPARSE_INDEX, max_tokens=q["max_tokens"]
        )
        index_ms = (time.time() - t0) * 1000
        for line in index_answer[:400].split("\n"):
            print(f"    {line}")
        print(f"    {DIM}({index_ms:.0f}ms){RESET}")

        c4_results.append({
            "id": q["id"],
            "parametric": {"answer": param_answer, "ms": param_ms},
            "index": {"answer": index_answer, "ms": index_ms},
        })

    # ------------------------------------------------------------------
    # Phase C6: The Headline Demo Numbers
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{'═' * 60}")
    print("  THE HEADLINE NUMBERS")
    print(f"{'═' * 60}{RESET}")

    sparse_index_tokens = len(tokenizer.encode(SPARSE_INDEX, add_special_tokens=False))
    sparse_index_bytes = len(SPARSE_INDEX.encode("utf-8"))

    print(f"""
  {BOLD}Document:{RESET}    Apollo 11 Air-to-Ground Transcript
  {BOLD}Tokens:{RESET}      {manifest.total_tokens:,}
  {BOLD}Windows:{RESET}     {manifest.num_windows}

  {BOLD}Naive KV cache:{RESET}  {fmt_bytes(naive_kv)}
  {BOLD}Checkpoint lib:{RESET}  {fmt_bytes(manifest.total_bytes)}  ({compression:.0f}x compression)
  {BOLD}Sparse index:{RESET}    {fmt_bytes(sparse_index_bytes)} ({sparse_index_tokens} tokens)

  {BOLD}Retrieval accuracy:{RESET}
    Parametric only:    {p_correct}/{len(c3_results)} ({RED}{100*p_correct/len(c3_results):.0f}%{RESET})
    Sparse index:       {i_correct}/{len(c3_results)} ({GREEN}{100*i_correct/len(c3_results):.0f}%{RESET})
    Window replay:      {r_correct}/{len(r_results)} ({GREEN}{100*r_correct/len(r_results):.0f}%{RESET})

  {BOLD}Cross-window synthesis:{RESET}
    Parametric: confabulates fictional moments
    Sparse index: identifies real transcript moments

  {BOLD}The number:{RESET}
    {GREEN}{manifest.total_tokens:,} tokens → {fmt_bytes(sparse_index_bytes)} sparse index{RESET}
    {GREEN}Compression: {manifest.total_tokens * 4 // max(sparse_index_bytes, 1):,}x vs raw tokens{RESET}
    {GREEN}Compression: {naive_kv // max(sparse_index_bytes, 1):,}x vs KV cache{RESET}
""")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_data = {
        "experiment": "C_apollo11_real",
        "model": args.model,
        "library": str(lib_path),
        "library_stats": {
            "total_tokens": manifest.total_tokens,
            "num_windows": manifest.num_windows,
            "checkpoint_bytes": manifest.checkpoint_bytes,
            "archive_bytes": manifest.archive_bytes,
            "naive_kv_bytes": naive_kv,
            "compression": round(compression, 1),
        },
        "sparse_index": {
            "tokens": sparse_index_tokens,
            "bytes": sparse_index_bytes,
        },
        "c3_results": c3_results,
        "c4_results": [
            {"id": r["id"], "parametric_len": len(r["parametric"]["answer"]),
             "index_len": len(r["index"]["answer"])}
            for r in c4_results
        ],
    }

    out_path = "results_experiment_c.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  {DIM}Results saved to {out_path}{RESET}\n")


if __name__ == "__main__":
    main()
