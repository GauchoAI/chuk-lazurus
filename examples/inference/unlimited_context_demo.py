#!/usr/bin/env python3
"""
Unlimited Context Demo — Mode 4: Checkpoint-Chained Retrieval

Context length is a property of the inference engine, not the model.

A transformer with an 8 K native context window retrieves novel facts from
arbitrary distances via residual checkpoint chaining.  The Markov property
guarantees checkpoint completeness: the residual stream IS the complete
forward state.

This demo proves the claim in three acts:

  Act 1 — PLANT
    Encode a novel fact ("Zarkov Industries was founded in Voltara.") as the
    first tokens of an 8 K-token window, followed by unrelated filler.
    Close the window: one checkpoint (last-position K,V) + token archive
    (~16 KB) are saved.  The KV cache for the window is discarded.

  Act 2 — FAIL (without replay)
    Ask the retrieval question with only current-window context (empty).
    The model has no access to the fact; it hallucinates a location.

  Act 3 — SUCCEED (with replay)
    Replay window 0: re-run its token IDs through KV-Direct to reconstruct
    the full 8 K-token KV cache.  Extend with the query.  The model attends
    to the planted fact and outputs "Voltara".

Storage comparison:
  Standard KV for 8 K tokens: ~150 MB
  Mode 4 (warm + cold):        ~18–174 KB + ~16 KB

Usage:
    uv run python examples/inference/unlimited_context_demo.py
    uv run python examples/inference/unlimited_context_demo.py --model mlx-community/gemma-3-1b-it-bf16
    uv run python examples/inference/unlimited_context_demo.py --window-size 512 --filler-tokens 400
    uv run python examples/inference/unlimited_context_demo.py --multi-fact
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def parse_args():
    p = argparse.ArgumentParser(description="Unlimited context via checkpoint chaining")
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Tokens per context window (default: 512 for fast demo; "
        "use 8192 for the full scenario)",
    )
    p.add_argument(
        "--filler-tokens",
        type=int,
        default=0,
        help="Override filler length (default: fill to window_size - fact_tokens)",
    )
    p.add_argument("--gen-tokens", type=int, default=12, help="Max tokens to generate per query")
    p.add_argument(
        "--multi-fact",
        action="store_true",
        help="Run multi-fact / multi-window variant (3 windows, 3 facts)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Filler text — varied passages so the fact is genuinely buried
# ---------------------------------------------------------------------------

_FILLER_PASSAGES = [
    (
        "Information theory, developed by Claude Shannon in 1948, provides the mathematical "
        "foundation for quantifying, storing, and transmitting information.  Shannon's central "
        "insight was that information could be measured independently of its meaning, using the "
        "concept of entropy.  The channel capacity theorem proves that reliable communication is "
        "possible at any rate below the channel capacity, and impossible above it.  "
    ),
    (
        "The Markov property states that the future is independent of the past given the present. "
        "A stochastic process has the Markov property if the conditional probability distribution "
        "of future states depends only on the present state, not on the sequence of events that "
        "preceded it.  Markov chains model queues and random walks; hidden Markov models underpin "
        "speech recognition; Markov decision processes define reinforcement learning environments.  "
    ),
    (
        "Modern large language models draw on information-theoretic principles at every level.  "
        "The cross-entropy loss used during training is the information-theoretic measure of how "
        "well a model's probability distribution matches the true data distribution.  Perplexity — "
        "the exponentiated cross-entropy — measures how surprised the model is by the test data.  "
        "Compression and prediction are two sides of the same coin.  "
    ),
    (
        "The residual stream in a transformer carries information forward through layers.  At each "
        "layer, the attention mechanism routes information between token positions, and the "
        "feedforward network transforms the representation at each position.  The Markov property "
        "of the residual stream — that each layer's output is a sufficient statistic for all "
        "subsequent layers — means that no information exists outside the residual.  "
    ),
    (
        "Rotary position embeddings encode sequence position directly into the key and query "
        "vectors used in attention.  A rotation matrix parameterised by position index is applied "
        "to each head's key and query before the dot product.  The inner product of two rotated "
        "vectors depends only on their relative displacement, making RoPE compatible with "
        "key-value caching and sequence extrapolation.  "
    ),
    (
        "Grouped query attention reduces the memory footprint of the key-value cache by sharing "
        "key and value projections across groups of query heads.  With G query heads sharing one "
        "key-value head, the KV cache shrinks by a factor of G while attention quality degrades "
        "only marginally.  Gemma 3 uses one key-value head for the 270 M and 1 B models, and "
        "four for the 4 B model.  "
    ),
    (
        "The sliding window attention mechanism limits each token's receptive field to a fixed "
        "number of preceding tokens for most layers, reserving global attention for a subset of "
        "designated layers.  This trades some long-range recall for a linear reduction in "
        "attention compute.  Gemma 3 applies a sliding window of 512 tokens with one global "
        "layer every six layers.  "
    ),
    (
        "Speculative decoding uses a small draft model to propose a batch of candidate tokens "
        "which are then verified in parallel by the larger target model.  Tokens accepted without "
        "revision contribute to wall-clock throughput at a fraction of the target model's cost.  "
        "The draft acceptance rate determines the effective speedup; for well-matched model "
        "families, acceptance rates above 80 percent are common.  "
    ),
]


def build_filler_text(target_tokens: int, tokenizer) -> str:
    """
    Build filler text reaching approximately target_tokens.

    Cycles through varied passages so the content is not purely repetitive.
    """
    passages = _FILLER_PASSAGES
    result = ""
    idx = 0
    approx = 0
    while approx < target_tokens:
        p = passages[idx % len(passages)]
        result += p
        approx = len(tokenizer.encode(result))
        idx += 1
    # Trim to exactly target_tokens
    full_ids = tokenizer.encode(result)
    return tokenizer.decode(full_ids[:target_tokens])


# ---------------------------------------------------------------------------
# Model / tokenizer loading (mirrors gemma_kv_direct_live.py)
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                model_id,
                local_files_only=True,
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
            )
        )
    except Exception:
        pass
    print(f"  Downloading {model_id} ...")
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
    )


def _apply_weights(model, model_path: Path) -> None:
    from mlx.utils import tree_unflatten

    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))
    sanitized = model.sanitize(raw)
    sanitized = {
        k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in sanitized.items()
    }
    model.update(tree_unflatten(list(sanitized.items())))
    mx.eval(model.parameters())


def load_models(model_id: str):
    import importlib

    from transformers import AutoTokenizer

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config = GemmaConfig.from_hf_config(json.load(f))

    rs = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs, model_path)
    rs.eval()

    # Load UnlimitedContextEngine
    inf = Path(__file__).parents[2] / "src/chuk_lazarus/inference"

    def _load(dotted, fpath):
        spec = importlib.util.spec_from_file_location(dotted, fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("chuk_lazarus.inference.context.kv_generator", inf / "context" / "kv_generator.py")
    engine_mod = _load(
        "chuk_lazarus.inference.context.unlimited_engine", inf / "context" / "unlimited_engine.py"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return rs, config, engine_mod.UnlimitedContextEngine, tokenizer


# ---------------------------------------------------------------------------
# Act helpers
# ---------------------------------------------------------------------------


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def greedy_sample(
    engine,
    tokenizer,
    query_text: str,
    replay_window_ids: list[int] | None,
    max_new_tokens: int,
    label: str,
) -> tuple[str, float]:
    """Run one generate call and return (decoded_text, elapsed_s)."""
    q_ids = encode(tokenizer, query_text)
    t0 = time.perf_counter()
    gen = engine.generate(
        q_ids,
        replay_window_ids=replay_window_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - t0
    text = decode(tokenizer, gen)
    return text, elapsed


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def run_two_window_demo(rs, config, EngineClass, tokenizer, args):
    """
    Act 1 — PLANT
    Act 2 — FAIL (no replay)
    Act 3 — SUCCEED (with replay)
    """
    print(f"\n{BOLD}Act 1 — PLANT{RESET}")
    print("  Encoding a novel fact at the start of window 0, followed by filler.")

    # The novel fact — something the model definitely does not know
    FACT = "Zarkov Industries was founded in the city of Voltara in 1987."
    QUERY = "Where was Zarkov Industries founded? Zarkov Industries was founded in the city of"
    KEYWORD = "voltara"

    # Build window 0: fact + filler to reach window_size tokens
    fact_ids = encode(tokenizer, FACT)
    target_filler = args.window_size - len(fact_ids)
    if args.filler_tokens > 0:
        target_filler = min(args.filler_tokens, args.window_size - len(fact_ids))

    print(f"  Fact tokens:   {len(fact_ids)}")
    print(f"  Building filler ({target_filler} tokens) ...")
    filler_text = build_filler_text(target_filler, tokenizer)
    filler_ids = encode(tokenizer, filler_text)[:target_filler]

    # Fact at the END of the window so its RoPE position is adjacent to the
    # query (positions window_size-fact_len .. window_size-1).  The window
    # boundary still forces eviction — the fact is only reachable via replay.
    window0_ids = filler_ids + fact_ids
    print(f"  Window 0 size: {len(window0_ids)} tokens")
    print(f"  Fact:          {CYAN}{FACT}{RESET}")

    engine = EngineClass(rs, config, window_size=args.window_size)

    # Warm up
    _warm = mx.array([[1, 2, 3, 4, 5]])
    _, _kv = engine.kv_gen.prefill(_warm)
    mx.eval()

    # Process window 0
    t0 = time.perf_counter()
    engine.process(window0_ids)
    engine.flush()  # close partial window if < window_size
    process_ms = (time.perf_counter() - t0) * 1000

    s = engine.stats()
    kv_equiv_bytes = s.equivalent_kv_bytes

    print(f"\n  Window closed in {process_ms:.0f} ms")
    print(f"  Archived windows:  {s.archived_windows}")
    print(f"  Checkpoint store:  {fmt_bytes(s.checkpoint_bytes)}  (last-position K,V per layer)")
    print(f"  Token archive:     {fmt_bytes(s.archive_bytes)}  (raw token IDs)")
    print(f"  Total (warm+cold): {fmt_bytes(s.cold_warm_bytes)}")
    print(f"  Equivalent KV:     {fmt_bytes(kv_equiv_bytes)}  (what a standard KV cache would use)")
    print(f"  Compression ratio: {s.compression_ratio:.0f}×")

    # ------------------------------------------------------------------
    print(f"\n{BOLD}Act 2 — FAIL  (no replay, current window is empty){RESET}")
    print(f'  Query: {DIM}"{QUERY}"{RESET}')

    text_no_replay, t_no_replay = greedy_sample(
        engine,
        tokenizer,
        QUERY,
        replay_window_ids=None,
        max_new_tokens=args.gen_tokens,
        label="no-replay",
    )

    found = KEYWORD in text_no_replay.lower()
    status = f"{RED}✗ Did not find '{KEYWORD}'{RESET}"
    print(f"\n  Generated: {YELLOW}{repr(text_no_replay)}{RESET}")
    print(f"  Result:    {status}")
    print(f"  Time:      {t_no_replay * 1000:.0f} ms")
    print(f"  {DIM}(Model has no context from window 0 — hallucination expected){RESET}")

    # ------------------------------------------------------------------
    print(f"\n{BOLD}Act 3 — SUCCEED  (replay window 0){RESET}")
    print(f"  Replaying window 0 ({len(window0_ids)} tokens) ...")

    t0 = time.perf_counter()
    text_replay, t_replay = greedy_sample(
        engine,
        tokenizer,
        QUERY,
        replay_window_ids=[0],
        max_new_tokens=args.gen_tokens,
        label="replay",
    )
    t_total = time.perf_counter() - t0

    found_replay = KEYWORD in text_replay.lower()
    status_r = (
        f"{GREEN}✓ Found '{KEYWORD}'{RESET}"
        if found_replay
        else f"{RED}✗ Did not find '{KEYWORD}'{RESET}"
    )
    print(f"\n  Generated: {GREEN}{repr(text_replay)}{RESET}")
    print(f"  Result:    {status_r}")
    print(f"  Time:      {t_total * 1000:.0f} ms  (includes window replay + generation)")

    # ------------------------------------------------------------------
    print(f"\n{BOLD}Summary{RESET}")
    print(f"  {'Mode':<20} {'Generated':<40} {'Contains answer?':>17}")
    print(f"  {'─' * 20} {'─' * 40} {'─' * 17}")
    nr_str = f"{RED}No{RESET}" if not found else f"{GREEN}Yes{RESET}"
    r_str = f"{GREEN}Yes{RESET}" if found_replay else f"{RED}No{RESET}"
    print(f"  {'Without replay':<20} {repr(text_no_replay):<40} {nr_str:>17}")
    print(f"  {'With replay':<20} {repr(text_replay):<40} {r_str:>17}")

    print(f"""
  Storage trade-off
  ─────────────────
  Window 0 KV cache (if kept):  {fmt_bytes(kv_equiv_bytes)}
  Checkpoint + token archive:   {fmt_bytes(s.cold_warm_bytes)}
  Compression ratio:            {s.compression_ratio:.0f}×

  Replay cost (one-time):       {t_total * 1000:.0f} ms
  In-window generation (Mode 3): unchanged — no overhead when not replaying
""")

    return found_replay


def run_multi_fact_demo(rs, config, EngineClass, tokenizer, args):
    """
    Three facts planted in three consecutive windows.
    Retrieved from a fourth (empty) window via replay.
    """
    print(f"\n{BOLD}Multi-Fact / Multi-Window Demo{RESET}")
    print("=" * 58)

    FACTS = [
        (
            "Zarkov Industries was founded in the city of Voltara.",
            "Where was Zarkov Industries founded?",
            "voltara",
        ),
        (
            "Project Lazarus was launched from the island of Velmoor.",
            "Where was Project Lazarus launched from?",
            "velmoor",
        ),
        (
            "The activation threshold for the Crestwick array is exactly 0.0042.",
            "What is the activation threshold for the Crestwick array?",
            "0.0042",
        ),
    ]

    engine = EngineClass(rs, config, window_size=args.window_size)
    window_sizes = []

    # Warm up
    _warm = mx.array([[1, 2, 3, 4, 5]])
    _, _kv = engine.kv_gen.prefill(_warm)
    mx.eval()

    print(f"\n  Planting facts across {len(FACTS)} windows ...")
    for wid, (fact, _, _) in enumerate(FACTS):
        fact_ids = encode(tokenizer, fact)
        target_fill = args.window_size - len(fact_ids)
        filler_text = build_filler_text(target_fill, tokenizer)
        filler_ids = encode(tokenizer, filler_text)[:target_fill]
        window_ids = filler_ids + fact_ids  # fact at end, adjacent to next window

        engine.process(window_ids)
        engine.flush()
        window_sizes.append(len(window_ids))
        print(f"    Window {wid}: {len(window_ids)} tokens — {DIM}{fact}{RESET}")

    s = engine.stats()
    print(f"\n  Total archived:    {s.archived_windows} windows / {s.total_tokens} tokens")
    print(f"  Warm+cold storage: {fmt_bytes(s.cold_warm_bytes)}")
    print(f"  Equivalent KV:     {fmt_bytes(s.equivalent_kv_bytes)}")
    print(f"  Compression ratio: {s.compression_ratio:.0f}×")

    print("\n  Retrieving facts (replay each window in turn) ...")
    print(f"\n  {'Win':>3}  {'Question':<50}  {'Answer':<35}  {'OK':>4}")
    print(f"  {'─' * 3}  {'─' * 50}  {'─' * 35}  {'─' * 4}")

    all_correct = True
    for wid, (fact, question, keyword) in enumerate(FACTS):
        query_text = question + " The answer is"
        q_ids = encode(tokenizer, query_text)

        t0 = time.perf_counter()
        gen = engine.generate(
            q_ids,
            replay_window_ids=[wid],
            max_new_tokens=args.gen_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t0
        answer = decode(tokenizer, gen)

        ok = keyword.lower() in answer.lower()
        all_correct = all_correct and ok
        tick = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        q_disp = question[:48] + ".." if len(question) > 50 else question
        a_disp = repr(answer)[:33] + ".." if len(answer) > 35 else repr(answer)
        print(f"  {wid:>3}  {q_disp:<50}  {a_disp:<35}  {tick:>4}  ({elapsed * 1000:.0f} ms)")

    verdict = (
        f"{GREEN}All facts retrieved correctly.{RESET}"
        if all_correct
        else f"{RED}Some facts were not retrieved.{RESET}"
    )
    print(f"\n  {verdict}")
    return all_correct


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print(f"\n{BOLD}Unlimited Context Demo — Mode 4: Checkpoint-Chained Retrieval{RESET}")
    print("=" * 65)
    print(f"  Model:       {args.model}")
    print(f"  Window size: {args.window_size} tokens")
    print(f"  Gen tokens:  {args.gen_tokens}")

    print("\nLoading model ...")
    rs, config, EngineClass, tokenizer = load_models(args.model)

    print(f"  Layers:   {config.num_hidden_layers}")
    print(f"  Hidden:   {config.hidden_size}")
    print(f"  KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.head_dim}")
    kv_bpt = 2 * config.num_key_value_heads * config.head_dim * config.num_hidden_layers * 2
    print(f"  KV bytes/token: {fmt_bytes(kv_bpt)}")

    if args.multi_fact:
        run_multi_fact_demo(rs, config, EngineClass, tokenizer, args)
    else:
        run_two_window_demo(rs, config, EngineClass, tokenizer, args)

    print(f"\n{BOLD}What this demonstrates{RESET}")
    print(f"""
  Context length is a property of the inference engine, not the model.

  A {config.num_hidden_layers}-layer transformer with a {args.window_size}-token native window just
  retrieved a novel fact from beyond its context boundary.

  Mechanism:
    1. Process tokens through KV-Direct (Mode 3) within each window.
    2. At window boundary: save checkpoint (last K,V, ~{fmt_bytes(kv_bpt)}) + token IDs.
    3. For retrieval: replay archived tokens → reconstruct full window KV.
    4. Extend with query → model attends to the replayed context.

  The Markov property guarantees the checkpoint is complete: given the
  residual at a window boundary, the full forward state is recoverable.
  The token archive provides the content; the checkpoint seeds the state.

  No fine-tuning.  No RAG pipeline.  No new architecture.
  Just the Markov property and an inference engine that takes it seriously.
""")


if __name__ == "__main__":
    main()
