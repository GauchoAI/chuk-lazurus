"""Generate command — load a checkpoint library and generate with window replay.

Supports automatic compass routing via multiple strategies:
  - BM25: token-level keyword matching (fast, content-aware)
  - Deflection: residual shift from checkpoint context (geometric)
  - Hybrid: BM25 pre-filter → deflection re-rank (default)
  - Residual: legacy mean-centered cosine similarity
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace

from ._types import GenerateConfig, GenerateResult
from .compass_routing import RoutingStrategy, compass_route, two_pass_generate


async def context_generate_cmd(args: Namespace) -> None:
    """CLI entry point: load a checkpoint library, replay windows, generate."""
    import mlx.core as mx

    from ....inference import UnifiedPipeline
    from ....inference.context import CheckpointLibrary
    from ....inference.context.unlimited_engine import UnlimitedContextEngine

    config = GenerateConfig.from_args(args)

    # ------------------------------------------------------------------
    # 1. Load library
    # ------------------------------------------------------------------
    if not config.checkpoint.exists():
        print(f"Error: library not found: {config.checkpoint}", file=sys.stderr)
        return

    print(f"Loading library: {config.checkpoint}", file=sys.stderr)
    lib = CheckpointLibrary(config.checkpoint)
    print(
        f"  {lib.manifest.name}  |  {lib.total_tokens:,} tokens  |  "
        f"{lib.num_windows} windows  |  model: {lib.manifest.model_id}",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {config.model}", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(config.model, verbose=False)

    if lib.manifest.model_id != config.model:
        print(
            f"Warning: library model_id={lib.manifest.model_id!r} but loading model={config.model!r}",
            file=sys.stderr,
        )

    tokenizer = pipeline.tokenizer

    # ------------------------------------------------------------------
    # 3. Build engine
    # ------------------------------------------------------------------
    engine = UnlimitedContextEngine(
        pipeline.model, pipeline.config, window_size=lib.window_size
    )
    engine.load_library(lib)
    kv_gen = engine.kv_gen

    # ------------------------------------------------------------------
    # 4. Encode prompt (with chat template by default)
    # ------------------------------------------------------------------
    prompt_text = config.prompt_text
    if not prompt_text:
        print("Error: no prompt specified. Use --prompt or --prompt-file.", file=sys.stderr)
        return

    no_chat = getattr(args, "no_chat_template", False)
    system_prompt = getattr(args, "system_prompt", None)
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # 5. Resolve which windows to replay
    # ------------------------------------------------------------------
    replay_arg = getattr(args, "replay", None)
    find_term = getattr(args, "find", None)
    top_k_override = getattr(args, "top_k", None)
    strategy_arg = getattr(args, "strategy", None)

    replay_ids = _resolve_replay(lib, tokenizer, replay_arg, find_term)

    if replay_ids is None:
        # Auto mode: use compass routing
        strategy = RoutingStrategy(strategy_arg) if strategy_arg else RoutingStrategy.BM25
        top_k = top_k_override if top_k_override is not None else 3

        # Two-pass strategy handles its own replay and generation
        if strategy == RoutingStrategy.TWOPASS:
            speculative_tokens = getattr(args, "speculative_tokens", 50) or 50
            result = two_pass_generate(
                lib, kv_gen, prompt_ids, prompt_text, tokenizer, engine,
                max_new_tokens=config.max_tokens,
                speculative_tokens=speculative_tokens,
                top_k=top_k,
                temperature=config.temperature,
            )
            gen_result = GenerateResult(
                response=tokenizer.decode(result["tokens"], skip_special_tokens=True),
                tokens_generated=len(result["tokens"]),
                context_tokens=result["context_tokens"],
            )
            print(gen_result.to_display())
            return

        replay_ids = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=pipeline.config,
            strategy=strategy,
            top_k=top_k,
        )

    print(f"  Replaying windows: {replay_ids}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 6. Replay windows contiguously (repack positions from 0)
    #
    # Original absolute positions are too far apart for the model's
    # attention to reach.  We re-encode each window's tokens at fresh
    # contiguous positions so all content is within effective range.
    # ------------------------------------------------------------------
    context_kv = engine._make_empty_kv()
    seq_len = 0

    # Replay in relevance order (lowest-scoring first, best last).
    # Sliding-window attention means non-global layers only attend to
    # nearby positions — the best-matching window goes last, adjacent
    # to the prompt, for maximum attention coverage.
    for wid in replay_ids:
        w_tokens = lib.get_window_tokens(wid)
        w_ids = mx.array(w_tokens)[None]
        t0 = time.time()
        if seq_len == 0:
            _logits, context_kv = kv_gen.prefill(w_ids)
        else:
            _logits, context_kv = kv_gen.extend(w_ids, context_kv, abs_start=seq_len)
        mx.eval(*[t for pair in context_kv for t in pair])
        elapsed_ms = (time.time() - t0) * 1000
        print(
            f"  Replayed window {wid} @ pos {seq_len}–{seq_len + len(w_tokens) - 1} ({elapsed_ms:.0f}ms)",
            file=sys.stderr,
        )
        seq_len += len(w_tokens)

    # ------------------------------------------------------------------
    # 7. Extend context with prompt
    # ------------------------------------------------------------------
    q_ids = mx.array(prompt_ids)[None]
    print(f"Extending with {len(prompt_ids)} prompt tokens @ pos {seq_len}...", file=sys.stderr)
    logits, gen_kv = kv_gen.extend(q_ids, context_kv, abs_start=seq_len)
    seq_len += len(prompt_ids)

    context_tokens = seq_len

    # ------------------------------------------------------------------
    # 8. Autoregressive decode
    # ------------------------------------------------------------------
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    generated_tokens: list[int] = []

    for _ in range(config.max_tokens):
        last_logits = logits[0, -1]

        if config.temperature == 0.0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / config.temperature
            next_token = int(mx.random.categorical(scaled[None]).item())

        if next_token in stop_ids:
            break

        generated_tokens.append(next_token)

        # Stream token to stdout
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        sys.stdout.write(token_text)
        sys.stdout.flush()

        logits, gen_kv = kv_gen.step_uncompiled(mx.array([[next_token]]), gen_kv, seq_len=seq_len)
        seq_len += 1

    print()  # newline after streamed output

    result = GenerateResult(
        response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        tokens_generated=len(generated_tokens),
        context_tokens=context_tokens,
    )
    print(result.to_display())


def _resolve_replay(
    lib: object,
    tokenizer: object,
    replay_arg: list[str] | None,
    find_term: str | None,
) -> list[int] | None:
    """Resolve --replay and --find flags into a list of window IDs.

    Returns None to signal "use auto compass routing".
    """
    num_windows = lib.num_windows

    # --find takes priority: locate the window containing the term
    if find_term:
        wid = lib.find_window_for_term(find_term, tokenizer)
        if wid is not None:
            print(f"  Found '{find_term}' in window {wid}", file=sys.stderr)
            return [wid]
        else:
            print(f"  Warning: '{find_term}' not found in any window, using auto", file=sys.stderr)
            return None

    # --replay: parse the argument
    if replay_arg is not None:
        if len(replay_arg) == 1:
            val = replay_arg[0]
            if val == "auto":
                return None
            elif val == "all":
                return list(range(num_windows))
            elif val == "last":
                return [num_windows - 1] if num_windows > 0 else []
            else:
                try:
                    return [int(val)]
                except ValueError:
                    print(f"  Warning: invalid replay value '{val}', using auto", file=sys.stderr)
                    return None
        else:
            # Multiple window IDs: --replay 0 1 45
            ids = []
            for v in replay_arg:
                try:
                    ids.append(int(v))
                except ValueError:
                    pass
            return ids if ids else None

    # Default: auto compass routing
    return None


__all__ = ["context_generate_cmd"]
