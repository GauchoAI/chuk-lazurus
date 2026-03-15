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

    if use_accumulated:
        # Inject accumulated checkpoint KVs from evenly-spaced windows.
        # Each checkpoint is the Markov state at that window's boundary —
        # the accumulated understanding after all tokens up to that point.
        # Concatenating N checkpoints gives N positions spanning the full
        # document for the model to attend to.
        n_positions = min(72, lib.num_windows)
        step = max(1, lib.num_windows // n_positions)
        sample_wids = list(range(0, lib.num_windows, step))
        # Always include the last window
        if sample_wids[-1] != lib.num_windows - 1:
            sample_wids.append(lib.num_windows - 1)

        # Concatenate checkpoint KVs: per-layer K,V along seq dimension
        first_kv = lib.get_checkpoint(sample_wids[0])
        num_layers = len(first_kv)
        concat_kv = [
            (
                mx.concatenate([lib.get_checkpoint(wid)[li][0] for wid in sample_wids], axis=2),
                mx.concatenate([lib.get_checkpoint(wid)[li][1] for wid in sample_wids], axis=2),
            )
            for li in range(num_layers)
        ]
        context_kv = concat_kv
        seq_len = len(sample_wids)
        mx.eval(*[t for pair in context_kv for t in pair])
        print(
            f"  Injected {len(sample_wids)} accumulated states "
            f"(every {step} windows, {lib.total_tokens} tokens compressed to "
            f"{seq_len} positions)",
            file=sys.stderr,
        )
    elif use_inject:
        # Dark space injection — the model reads its own L26 residuals.
        #
        # 1. Load ALL stored L26 residuals (5,800 positions from prefill)
        # 2. Process query through L0-L26 → query residual at L26
        # 3. Concatenate: [stored residuals] + [query residual] at L26
        # 4. Run layers 27-34 on the combined sequence
        # 5. The model's own attention reads ALL dark space states
        #    in full 2560D — content, tone, register, entity
        #
        # No tokens. No projection. No compass selection.
        # The model reads its own computation through its own layers.

        compass_layer = lib.compass_layer

        # Select windows: compass-routed if strategy given, otherwise all
        strategy_arg = getattr(args, "strategy", None)
        top_k_override = getattr(args, "top_k", None)

        if strategy_arg:
            strategy = RoutingStrategy(strategy_arg)
            top_k = top_k_override if top_k_override is not None else 3
            inject_wids = compass_route(
                lib, kv_gen, prompt_ids, prompt_text, tokenizer,
                model_config=pipeline.config,
                strategy=strategy,
                top_k=top_k,
            )
        else:
            inject_wids = list(range(lib.num_windows))

        # Load L26 residuals for selected windows
        inject_vecs = []
        for wid in inject_wids:
            for res in lib.get_compass_residuals(wid):
                inject_vecs.append(res.reshape(-1))

        n_stored = len(inject_vecs)
        print(
            f"  Loading {n_stored} L26 residuals from {len(inject_wids)} windows",
            file=sys.stderr,
        )

        # Process query through L0-L26 to get its residual
        q_ids = mx.array(prompt_ids)[None]
        q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
        # q_h: (1, q_len, 2560) — all query positions at L26
        q_len = q_h.shape[1]

        # Concatenate: [stored dark space] + [query] at L26
        stored_h = mx.stack(inject_vecs, axis=0)[None, :, :]  # (1, N, 2560)
        combined_h = mx.concatenate([stored_h, q_h], axis=1)  # (1, N+q_len, 2560)
        total_seq = n_stored + q_len

        print(
            f"  Combined sequence: {n_stored} stored + {q_len} query = "
            f"{total_seq} positions at L{compass_layer}",
            file=sys.stderr,
        )

        # Run layers 27-34 on the combined sequence
        t0 = time.time()
        logits, context_kv = kv_gen.prefill_from_layer(
            combined_h, start_layer=compass_layer,
        )
        elapsed_ms = (time.time() - t0) * 1000
        seq_len = total_seq
        print(
            f"  Dark space processed: {total_seq} positions through "
            f"L{compass_layer}-L33 ({elapsed_ms:.0f}ms)",
            file=sys.stderr,
        )

        # Skip the normal postamble — logits already include query positions.
        # Generate directly from the last position's logits.
        context_tokens = seq_len
        gen_kv = context_kv

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
            token_text = tokenizer.decode([next_token], skip_special_tokens=True)
            sys.stdout.write(token_text)
            sys.stdout.flush()

            logits, gen_kv = kv_gen.step_uncompiled(
                mx.array([[next_token]]), gen_kv, seq_len=seq_len,
            )
            seq_len += 1

        print()
        result = GenerateResult(
            response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
            tokens_generated=len(generated_tokens),
            context_tokens=context_tokens,
        )
        print(result.to_display())
        return

    elif use_explore:
        # Agentic navigation: the model explores the document using the compass.
        # The compass map is shown to the model. The model generates text and
        # can request windows to read by emitting <READ:NNN>. The engine
        # replays the requested window into the KV cache and the model continues.

        # Get compass scores for the map
        strategy_arg = getattr(args, "strategy", None)
        strategy = RoutingStrategy(strategy_arg) if strategy_arg else RoutingStrategy.GEOMETRIC
        compass_scores = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=pipeline.config,
            strategy=strategy,
            top_k=20,  # show top 20 on the map
        )

        # Build the compass map as text
        map_lines = ["Document map (725 windows, 370K tokens of Apollo 11 transcript):"]
        map_lines.append("Top windows by geometric relevance to your query:")
        for i, wid in enumerate(compass_scores[:20]):
            w = lib.windows[wid]
            preview = w.preview[:60]
            map_lines.append(f"  Window {wid}: {preview}")
        map_lines.append("")
        map_lines.append("To read a window's full content, output: <READ:NNN>")
        map_lines.append("Read windows that seem relevant, then answer the question.")
        compass_map = "\n".join(map_lines)

        # Build prompt with compass map
        no_chat = getattr(args, "no_chat_template", False)
        system_prompt = getattr(args, "system_prompt", None)
        sys_content = system_prompt or (
            "You are exploring a document transcript. You have a compass map showing the "
            "most relevant windows. You can read any window by outputting <READ:NNN> where "
            "NNN is the window number. Read 2-3 windows that seem relevant, then answer "
            "the question using exact quotes from what you read."
        )

        explore_prompt = (
            f"<start_of_turn>user\n{sys_content}\n\n"
            f"{compass_map}\n\n"
            f"Question: {prompt_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        explore_ids = tokenizer.encode(explore_prompt, add_special_tokens=True)

        # Prefill the explore prompt
        e_ids = mx.array(explore_ids)[None]
        logits, gen_kv = kv_gen.prefill(e_ids)
        mx.eval(logits, *[t for pair in gen_kv for t in pair])
        seq_len = len(explore_ids)
        print(f"  Explore prompt: {seq_len} tokens", file=sys.stderr)

        # Agentic generation loop
        stop_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(tokenizer.eos_token_id)

        generated_tokens: list[int] = []
        generated_text = ""
        windows_read: set[int] = set()
        max_reads = 5  # safety limit

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
            token_text = tokenizer.decode([next_token], skip_special_tokens=True)
            sys.stdout.write(token_text)
            sys.stdout.flush()
            generated_text += token_text

            # Check for <READ:NNN> pattern (flexible — 4B models may omit closing >)
            import re
            read_match = re.search(r'<READ:(\d+)(?:>|\s|\n)', generated_text)
            if read_match and len(windows_read) < max_reads:
                read_wid = int(read_match.group(1))
                if 0 <= read_wid < lib.num_windows and read_wid not in windows_read:
                    windows_read.add(read_wid)
                    # Strip the <READ:NNN> from tracked text
                    generated_text = generated_text[:read_match.start()]

                    # Replay the requested window
                    w_tokens = lib.get_window_tokens(read_wid)
                    w_ids = mx.array(w_tokens)[None]
                    t0 = time.time()
                    _logits, gen_kv = kv_gen.extend(w_ids, gen_kv, abs_start=seq_len)
                    mx.eval(*[t for pair in gen_kv for t in pair])
                    seq_len += len(w_tokens)
                    elapsed_ms = (time.time() - t0) * 1000
                    print(
                        f"\n  [Read window {read_wid}: {len(w_tokens)} tokens injected "
                        f"@ pos {seq_len - len(w_tokens)}–{seq_len - 1} ({elapsed_ms:.0f}ms)]",
                        file=sys.stderr, flush=True,
                    )

                    # Inject continuation prompt so model resumes analysis
                    cont_text = (
                        f"\n[Window {read_wid} content loaded. "
                        f"Windows read so far: {sorted(windows_read)}. "
                        f"Continue your analysis.]\n"
                    )
                    cont_ids = tokenizer.encode(cont_text, add_special_tokens=False)
                    cont_mx = mx.array(cont_ids)[None]
                    logits, gen_kv = kv_gen.extend(cont_mx, gen_kv, abs_start=seq_len)
                    mx.eval(logits, *[t for pair in gen_kv for t in pair])
                    seq_len += len(cont_ids)
                    continue

            logits, gen_kv = kv_gen.step_uncompiled(
                mx.array([[next_token]]), gen_kv, seq_len=seq_len,
            )
            seq_len += 1

        print()

        context_tokens = seq_len
        result = GenerateResult(
            response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
            tokens_generated=len(generated_tokens),
            context_tokens=context_tokens,
        )
        print(result.to_display())
        return

    elif use_compressed:
        # Hybrid: compass-routed windows at full 512 tokens +
        # everything else as compressed pre-RoPE pages.
        #
        # Top-k windows: full replay → 512 KV entries each (precise attention)
        # Remaining windows: 8 pre-RoPE pages each (document awareness)
        #
        # The model sees the whole document. It focuses on what matters.
        n_pages = 8
        strategy_arg = getattr(args, "strategy", None)
        top_k_override = getattr(args, "top_k", None)
        strategy = RoutingStrategy(strategy_arg) if strategy_arg else RoutingStrategy.GEOMETRIC
        top_k = top_k_override if top_k_override is not None else 3

        # Route: find the windows that matter
        routed_ids = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=pipeline.config,
            strategy=strategy,
            top_k=top_k,
        )
        routed_set = set(routed_ids)

        # Load pre-stored pages or compute on-the-fly
        pages_path = lib.path / "pages.npz"
        has_stored_pages = pages_path.exists()

        all_pages = []
        t_comp = time.time()
        compressed_count = 0

        if has_stored_pages:
            # Load pre-computed pre-RoPE pages from library (instant)
            raw_pages = dict(mx.load(str(pages_path)))
            num_layers = len(kv_gen.backbone.adapted_layers)

            for wid in range(lib.num_windows):
                if wid in routed_set:
                    continue
                for pi in range(n_pages):
                    page_kv = []
                    for li in range(num_layers):
                        k_key = f"w{wid}_p{pi}_l{li}_k"
                        v_key = f"w{wid}_p{pi}_l{li}_v"
                        if k_key in raw_pages:
                            page_kv.append((raw_pages[k_key], raw_pages[v_key]))
                    if page_kv:
                        all_pages.append(page_kv)
                        compressed_count += 1
            load_ms = (time.time() - t_comp) * 1000
            print(
                f"  Loaded {compressed_count} pre-stored pages ({load_ms:.0f}ms)",
                file=sys.stderr,
            )
        else:
            # Compute on-the-fly (slow — ~8 minutes for 725 windows)
            print("  No pre-stored pages — computing on-the-fly (use --store-pages during prefill)", file=sys.stderr)
            for wid in range(lib.num_windows):
                if wid in routed_set:
                    continue
                w_tokens = lib.get_window_tokens(wid)
                w_ids = mx.array(w_tokens)[None]
                _logits, _kv, pages = kv_gen.prefill_pages(w_ids, n_pages=n_pages)
                all_pages.extend(pages)
                compressed_count += 1

                if compressed_count % 50 == 0 or wid == lib.num_windows - 1:
                    elapsed = time.time() - t_comp
                    rate = compressed_count / elapsed if elapsed > 0 else 0
                    remaining = lib.num_windows - len(routed_set) - compressed_count
                    eta = remaining / rate if rate > 0 else 0
                    print(
                        f"\r  Compressing: {compressed_count}/{lib.num_windows - len(routed_set)} windows  "
                        f"{rate:.1f} w/s  ETA {eta:.0f}s\033[K",
                        end="", file=sys.stderr, flush=True,
                    )
            print(file=sys.stderr)

        # Build KV cache: compressed pages first (background), then full windows (foreground)
        # Compressed pages at positions 0..N_compressed-1
        total_compressed = len(all_pages)
        target_offsets = list(range(total_compressed))

        print(
            f"  Injecting {total_compressed} compressed pages "
            f"({compressed_count} windows × {n_pages} pages)",
            file=sys.stderr, flush=True,
        )
        context_kv = kv_gen.inject_pages(all_pages, target_offsets)
        seq_len = total_compressed

        # Full replay of routed windows at positions after compressed pages
        # Best-match window goes last (closest to prompt for sliding window attention)
        print(f"  Full replay of {len(routed_ids)} routed windows:", file=sys.stderr)
        for wid in routed_ids:
            w_tokens = lib.get_window_tokens(wid)
            w_ids = mx.array(w_tokens)[None]
            t0 = time.time()
            _logits, context_kv = kv_gen.extend(w_ids, context_kv, abs_start=seq_len)
            mx.eval(*[t for pair in context_kv for t in pair])
            elapsed_ms = (time.time() - t0) * 1000
            print(
                f"    window {wid} @ pos {seq_len}–{seq_len + len(w_tokens) - 1} "
                f"({elapsed_ms:.0f}ms, full 512 tokens)",
                file=sys.stderr,
            )
            seq_len += len(w_tokens)

        comp_elapsed = time.time() - t_comp
        print(
            f"  Hybrid loaded: {total_compressed} compressed + "
            f"{sum(len(lib.get_window_tokens(wid)) for wid in routed_ids)} full = "
            f"{seq_len} positions ({comp_elapsed:.1f}s)",
            file=sys.stderr,
        )

    else:
        # Preamble
        if preamble_ids:
            p_ids = mx.array(preamble_ids)[None]
            _logits, context_kv = kv_gen.prefill(p_ids)
            mx.eval(*[t for pair in context_kv for t in pair])
            seq_len += len(preamble_ids)
            print(f"  Preamble: {len(preamble_ids)} tokens", file=sys.stderr)

        # Replay transcript windows
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
    # 7. Extend context with postamble (question + generation prompt)
    # ------------------------------------------------------------------
    q_ids = mx.array(postamble_ids)[None]
    print(f"Extending with {len(postamble_ids)} prompt tokens @ pos {seq_len}...", file=sys.stderr)
    logits, gen_kv = kv_gen.extend(q_ids, context_kv, abs_start=seq_len)
    seq_len += len(postamble_ids)

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


def _iterative_generate(
    lib,
    kv_gen,
    engine,
    tokenizer,
    model_config,
    prompt_ids: list[int],
    prompt_text: str,
    config,
    top_k: int = 1,
    max_rounds: int = 3,
    no_chat: bool = False,
    system_prompt: str | None = None,
) -> "GenerateResult":
    """Iterative compass navigation — the model's judgment drives routing.

    Each round:
      1. Compass routes to best unvisited window (using generation residual
         from prior round, or bare query for round 1)
      2. Replay window at full resolution
      3. Model generates short notes (reading + assessment)
      4. Extract L26 residual at the LAST GENERATED TOKEN
      5. That residual carries the model's judgment — not just "I read something"
         but "I read THIS and here's what I think"
      6. Next round routes with the generation residual

    Key insight: reading content shifts the compass by 7° through content-type
    space (PC2), but the tonal channel is invariant (0.25°). The model's
    GENERATION residual discriminates 33× better than the post-read residual
    because judgment lives in the output state, not the reading state.

    Final round: replay ALL discovered windows, generate full answer.
    """
    import mlx.core as mx

    visited: set[int] = set()
    notes: list[dict] = []  # {window, content_preview, response}
    gen_residual = None  # L26 at last generated token — the model's judgment

    compass_layer = lib.compass_layer

    sys_content = system_prompt or (
        "You are answering questions based on a document transcript. "
        "Answer using only information from the transcript. Quote exact text when possible."
    )

    print(f"  Iterative navigation: up to {max_rounds} rounds (generation-guided)", file=sys.stderr)

    for round_idx in range(max_rounds):
        # ── Compass route: use generation residual from prior round ──
        t0 = time.time()
        routed = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=model_config,
            strategy=RoutingStrategy.GEOMETRIC,
            top_k=top_k,
            exclude=visited,
            query_residual=gen_residual,  # None for round 1 → bare query
        )
        route_ms = (time.time() - t0) * 1000

        if not routed:
            print(f"  Round {round_idx + 1}: no unvisited windows left", file=sys.stderr)
            break

        # Take the best window (last in routed list = highest score)
        best_wid = routed[-1]
        visited.add(best_wid)

        # ── Replay best window + generate short response ──
        w_tokens = lib.get_window_tokens(best_wid)
        window_text = tokenizer.decode(w_tokens, skip_special_tokens=True)

        # Build framed context for this round's generation.
        # The prompt forces NOTE-TAKING mode, not answer mode.
        # This matters: the generation residual must reflect content
        # judgment ("is this relevant/amusing/interesting?"), not
        # answer formatting ("Here are three points...").
        if not no_chat and hasattr(tokenizer, "apply_chat_template"):
            preamble_text = (
                f"<start_of_turn>user\n{sys_content}\n\n"
                f"Here is the relevant transcript:\n\n"
            )
            preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)

            postamble_text = (
                f"\n\n---\nTask: {prompt_text}\n"
                f"DO NOT answer yet. Write 1-2 sentences of reading notes: "
                f"what specific content is in this excerpt and how relevant "
                f"is it to the task? What else would you need?<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)
        else:
            preamble_ids = tokenizer.encode("Transcript:\n\n", add_special_tokens=False)
            postamble_text = (
                f"\n\n---\nTask: {prompt_text}\n"
                f"DO NOT answer yet. Write 1-2 sentences of reading notes: "
                f"what specific content is in this excerpt and how relevant "
                f"is it to the task?\nNotes:"
            )
            postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)

        # Prefill: preamble
        seq_len = 0
        p_ids = mx.array(preamble_ids)[None]
        _logits, round_kv = kv_gen.prefill(p_ids)
        mx.eval(*[t for pair in round_kv for t in pair])
        seq_len += len(preamble_ids)

        # Replay window
        t0 = time.time()
        w_ids = mx.array(w_tokens)[None]
        _logits, round_kv = kv_gen.extend(w_ids, round_kv, abs_start=seq_len)
        mx.eval(*[t for pair in round_kv for t in pair])
        replay_ms = (time.time() - t0) * 1000
        seq_len += len(w_tokens)

        # Extend with postamble
        q_ids = mx.array(postamble_ids)[None]
        logits, round_kv = kv_gen.extend(q_ids, round_kv, abs_start=seq_len)
        seq_len += len(postamble_ids)

        # Generate short notes (NOT a full answer).
        # Shorter generation = residual reflects judgment, not elaboration.
        round_max_tokens = 50
        stop_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(tokenizer.eos_token_id)

        round_tokens: list[int] = []
        for _ in range(round_max_tokens):
            last_logits = logits[0, -1]
            if config.temperature == 0.0:
                next_token = int(mx.argmax(last_logits).item())
            else:
                scaled = last_logits / config.temperature
                next_token = int(mx.random.categorical(scaled[None]).item())
            if next_token in stop_ids:
                break
            round_tokens.append(next_token)
            logits, round_kv = kv_gen.step_uncompiled(
                mx.array([[next_token]]), round_kv, seq_len=seq_len,
            )
            seq_len += 1

        round_response = tokenizer.decode(round_tokens, skip_special_tokens=True)

        # ── Extract L26 at the LAST GENERATED TOKEN ──
        # The model's judgment lives here — not at the post-read position.
        # Re-run the full sequence through to L26 to capture the generation
        # residual. This is a partial forward (layers 0-26 only), so fast.
        full_seq = list(preamble_ids) + list(w_tokens) + list(postamble_ids) + round_tokens
        t0_l26 = time.time()
        full_ids = mx.array(full_seq)[None]
        gen_h = kv_gen.prefill_to_layer(full_ids, target_layer=compass_layer)
        gen_residual = gen_h[0, -1:, :]  # (1, hidden_size) — last generated token
        mx.eval(gen_residual)
        l26_ms = (time.time() - t0_l26) * 1000

        notes.append({
            "window": best_wid,
            "content_preview": window_text[:500],
            "response": round_response,
        })

        print(
            f"  Round {round_idx + 1}: window {best_wid} "
            f"(route {route_ms:.0f}ms, replay {replay_ms:.0f}ms, "
            f"{len(round_tokens)} tok, L{compass_layer} extract {l26_ms:.0f}ms)",
            file=sys.stderr,
        )
        print(f"    Notes: {round_response[:120]}...", file=sys.stderr)

    # ── Final generation: replay last 2 windows + notes summary ──
    # The model already read every window and wrote notes. Use them.
    # Replay the LAST 2 discovered windows at full resolution — later
    # rounds find better content because the generation-guided compass
    # refines toward it through iterative judgment.
    # All rounds contribute notes; the best discoveries get full replay.
    all_windows = [n["window"] for n in notes]

    # Last-discovered windows are the compass's best finds.
    # Replay up to 3 (the later half of navigation).
    replay_limit = min(3, len(all_windows))
    replay_windows = all_windows[-replay_limit:]

    # Build notes summary text
    notes_summary_parts = []
    for n in notes:
        notes_summary_parts.append(
            f"Window {n['window']}: {n['response'].strip()}"
        )
    notes_summary = "\n".join(notes_summary_parts)

    print(
        f"  Final: replaying {len(replay_windows)} windows {replay_windows}, "
        f"+ notes from {len(all_windows)} rounds",
        file=sys.stderr,
    )

    # Build final framed context with notes + replay
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        preamble_text = (
            f"<start_of_turn>user\n{sys_content}\n\n"
            f"Reading notes from document exploration:\n{notes_summary}\n\n"
            f"Here is the relevant transcript:\n\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)
        postamble_text = (
            f"\n\n---\nUsing both the reading notes above and the transcript, "
            f"{prompt_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)
    else:
        preamble_text = (
            f"Reading notes:\n{notes_summary}\n\n"
            f"Transcript:\n\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=False)
        postamble_ids = tokenizer.encode(
            f"\n\n---\nUsing both the reading notes and transcript, "
            f"{prompt_text}\nAnswer:", add_special_tokens=False,
        )

    # Prefill preamble (includes notes summary)
    seq_len = 0
    p_ids = mx.array(preamble_ids)[None]
    _logits, final_kv = kv_gen.prefill(p_ids)
    mx.eval(*[t for pair in final_kv for t in pair])
    seq_len += len(preamble_ids)
    print(f"    preamble + notes: {len(preamble_ids)} tokens", file=sys.stderr)

    # Replay selected windows at full resolution
    for wid in replay_windows:
        w_tokens = lib.get_window_tokens(wid)
        w_ids = mx.array(w_tokens)[None]
        t0 = time.time()
        _logits, final_kv = kv_gen.extend(w_ids, final_kv, abs_start=seq_len)
        mx.eval(*[t for pair in final_kv for t in pair])
        elapsed_ms = (time.time() - t0) * 1000
        print(
            f"    window {wid} @ pos {seq_len}–{seq_len + len(w_tokens) - 1} "
            f"({elapsed_ms:.0f}ms)",
            file=sys.stderr,
        )
        seq_len += len(w_tokens)

    # Extend with postamble
    q_ids = mx.array(postamble_ids)[None]
    print(f"  Extending with {len(postamble_ids)} prompt tokens @ pos {seq_len}...", file=sys.stderr)
    logits, final_kv = kv_gen.extend(q_ids, final_kv, abs_start=seq_len)
    seq_len += len(postamble_ids)

    context_tokens = seq_len

    # Generate final answer
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
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        sys.stdout.write(token_text)
        sys.stdout.flush()
        logits, final_kv = kv_gen.step_uncompiled(
            mx.array([[next_token]]), final_kv, seq_len=seq_len,
        )
        seq_len += 1

    print()  # newline after streamed output

    return GenerateResult(
        response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        tokens_generated=len(generated_tokens),
        context_tokens=context_tokens,
    )


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
            elif val == "accumulated":
                return ["accumulated"]  # special marker
            elif val == "compressed":
                return ["compressed"]  # full document via page injection
            elif val == "explore":
                return ["explore"]  # agentic navigation
            elif val == "inject":
                return ["inject"]  # L26 residual injection
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
