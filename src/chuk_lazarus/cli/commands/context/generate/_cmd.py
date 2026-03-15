"""Generate command — load a checkpoint library and generate with window replay.

Supports automatic compass routing via multiple strategies:
  - BM25: token-level keyword matching (fast, content-aware)
  - Deflection: residual shift from checkpoint context (geometric)
  - Hybrid: BM25 pre-filter → deflection re-rank (default)
  - Residual: legacy mean-centered cosine similarity
"""

from __future__ import annotations

import sys
from argparse import Namespace

from .._types import GenerateConfig, GenerateResult
from ..compass_routing import RoutingStrategy, compass_route, two_pass_generate
from ._iterative import _iterative_generate
from ._resolve import _resolve_replay


def _decode_loop(logits, gen_kv, kv_gen, tokenizer, config, seq_len, mx):
    """Common autoregressive decode loop.

    Returns a GenerateResult.
    """
    context_tokens = seq_len

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

    return GenerateResult(
        response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        tokens_generated=len(generated_tokens),
        context_tokens=context_tokens,
    )


async def context_generate_cmd(args: Namespace) -> None:
    """CLI entry point: load a checkpoint library, replay windows, generate."""
    import mlx.core as mx

    from .....inference import UnifiedPipeline
    from .....inference.context import CheckpointLibrary
    from .....inference.context.unlimited_engine import UnlimitedContextEngine

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
    # 4. Encode prompt for routing (chat-wrapped for consistent geometry)
    # ------------------------------------------------------------------
    prompt_text = config.prompt_text
    if not prompt_text:
        print("Error: no prompt specified. Use --prompt or --prompt-file.", file=sys.stderr)
        return

    no_chat = getattr(args, "no_chat_template", False)
    system_prompt = getattr(args, "system_prompt", None)

    # Chat-wrapped prompt for compass routing — matches calibration geometry
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        routing_messages = []
        if system_prompt:
            routing_messages.append({"role": "system", "content": system_prompt})
        routing_messages.append({"role": "user", "content": prompt_text})
        routing_prompt = tokenizer.apply_chat_template(
            routing_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(routing_prompt, add_special_tokens=False)
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

        # Iterative strategy handles its own multi-round navigation
        if strategy == RoutingStrategy.ITERATIVE:
            max_rounds = getattr(args, "max_rounds", 3) or 3
            result = _iterative_generate(
                lib, kv_gen, engine, tokenizer, pipeline.config,
                prompt_ids, prompt_text, config,
                top_k=top_k,
                max_rounds=max_rounds,
                no_chat=no_chat,
                system_prompt=system_prompt,
            )
            print(result.to_display())
            return

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
    # 6. Build framed context: preamble + transcript windows + prompt
    #
    # The model needs to understand the raw transcript is context to
    # answer from. We wrap it in a chat-formatted structure:
    #   [preamble tokens] [window tokens...] [postamble + question tokens]
    #
    # Original absolute positions are too far apart for the model's
    # attention to reach.  We re-encode each window's tokens at fresh
    # contiguous positions so all content is within effective range.
    # ------------------------------------------------------------------
    no_chat = getattr(args, "no_chat_template", False)
    system_prompt = getattr(args, "system_prompt", None)

    # Build preamble + postamble for a single continuous user turn.
    # The transcript tokens go between them, inside the same turn.
    #
    # Target structure (Gemma 3):
    #   <bos><start_of_turn>user
    #   [system instruction]
    #   Here is the relevant transcript:
    #   [TRANSCRIPT TOKENS — raw, inside the user turn]
    #   ---
    #   Based on the transcript above, [question]<end_of_turn>
    #   <start_of_turn>model
    #
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        sys_content = system_prompt or (
            "You are answering questions based on the document transcript provided below. "
            "Answer using only information from the transcript. Quote exact text when possible."
        )
        # Preamble: BOS + start of user turn + system instruction + context header
        # We build this manually to keep the turn open for transcript tokens.
        preamble_text = (
            "<start_of_turn>user\n"
            f"{sys_content}\n\n"
            "Here is the relevant transcript:\n\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)

        # Postamble: close transcript, ask question, end turn, start model turn
        postamble_text = (
            f"\n\n---\nBased on the transcript above, {prompt_text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)
    else:
        preamble_ids = tokenizer.encode(
            "Transcript:\n\n", add_special_tokens=False,
        )
        postamble_ids = tokenizer.encode(
            f"\n\n---\nQuestion: {prompt_text}\nAnswer:", add_special_tokens=False,
        )

    # Prefill: preamble → transcript windows → postamble
    seq_len = 0

    # Check for special replay modes
    use_accumulated = (
        isinstance(replay_ids, list)
        and len(replay_ids) == 1
        and replay_ids[0] == "accumulated"
    )
    use_compressed = (
        isinstance(replay_ids, list)
        and len(replay_ids) == 1
        and replay_ids[0] == "compressed"
    )
    use_explore = (
        isinstance(replay_ids, list)
        and len(replay_ids) == 1
        and replay_ids[0] == "explore"
    )
    use_inject = (
        isinstance(replay_ids, list)
        and len(replay_ids) == 1
        and replay_ids[0] == "inject"
    )

    # Dispatch to mode handlers
    if use_accumulated:
        from ._modes._accumulated import run_accumulated
        context_kv, seq_len = run_accumulated(lib, kv_gen, mx)

    elif use_inject:
        from ._modes._inject import run_inject
        run_inject(lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx)
        return

    elif use_explore:
        from ._modes._explore import run_explore
        run_explore(lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx)
        return

    elif use_compressed:
        from ._modes._compressed import run_compressed
        context_kv, seq_len = run_compressed(
            lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, args, mx,
        )

    else:
        from ._modes._standard import run_standard
        context_kv, seq_len = run_standard(lib, kv_gen, replay_ids, preamble_ids, mx)

    # ------------------------------------------------------------------
    # 7. Extend context with postamble (question + generation prompt)
    # ------------------------------------------------------------------
    q_ids = mx.array(postamble_ids)[None]
    print(f"Extending with {len(postamble_ids)} prompt tokens @ pos {seq_len}...", file=sys.stderr)
    logits, gen_kv = kv_gen.extend(q_ids, context_kv, abs_start=seq_len)
    seq_len += len(postamble_ids)

    # ------------------------------------------------------------------
    # 8. Autoregressive decode
    # ------------------------------------------------------------------
    result = _decode_loop(logits, gen_kv, kv_gen, tokenizer, config, seq_len, mx)
    print(result.to_display())
