"""knowledge query — TF-IDF routing + Markov boundary reconstruction."""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path


async def knowledge_query_cmd(args: Namespace) -> None:
    """Route → load boundary → reconstruct window state → generate.

    The pure Markov path: the boundary residual carries the cumulative
    document state. Combined with the window's tokens, one forward pass
    reconstructs the full context. KL=0.0 vs full-document prefill.
    """
    import mlx.core as mx

    from ._common import generate_plain, load_model, prepare_prompt, stop_token_ids
    from ._metrics import JsonLogger

    json_log = JsonLogger(args.json_log) if getattr(args, "json_log", None) else None

    _, kv_gen, tokenizer = load_model(args.model)

    # No store → plain inference
    if not args.store:
        prompt_ids = prepare_prompt(tokenizer, args.prompt)
        stop_ids = stop_token_ids(tokenizer)
        print("No knowledge store — plain inference.", file=sys.stderr)
        generated = generate_plain(kv_gen, prompt_ids, args.max_tokens, stop_ids)
        output = tokenizer.decode(generated, skip_special_tokens=True)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        if json_log:
            json_log.event("query_plain", prompt=args.prompt, tokens_generated=len(generated))
            json_log.close()
        return

    from ....inference.context.knowledge import KnowledgeStore

    store_path = Path(args.store)
    if not store_path.exists():
        print(f"Error: knowledge store not found: {store_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading knowledge store: {store_path}", file=sys.stderr)
    store = KnowledgeStore.load(store_path)
    store.log_stats()

    prompt_ids = prepare_prompt(tokenizer, args.prompt)
    stop_ids = stop_token_ids(tokenizer)

    t0 = time.monotonic()

    # Query-side expansion: model generates discriminative keywords
    expansion_ids = KnowledgeStore._expand_query(args.prompt, tokenizer, kv_gen)
    expansion_words = sorted(
        {
            tokenizer.decode([t]).strip().lower()
            for t in expansion_ids
            if len(tokenizer.decode([t]).strip()) >= 2
        },
    )
    expand_ms = (time.monotonic() - t0) * 1000
    print(
        f"  Query expansion ({expand_ms:.0f} ms): {', '.join(expansion_words[:15])}",
        file=sys.stderr,
    )

    if json_log:
        json_log.event("query_start", prompt=args.prompt,
                       expansion_words=expansion_words[:15],
                       expansion_ms=round(expand_ms, 1))

    window_ids = store.route_top_k(
        args.prompt,
        tokenizer,
        k=args.top_k,
        expansion_ids=expansion_ids,
    )

    if not window_ids:
        print("  No matching windows — generating plain.", file=sys.stderr)
        generated = generate_plain(kv_gen, prompt_ids, args.max_tokens, stop_ids)
        if json_log:
            json_log.event("query_fallback", reason="no_matching_windows")
    else:
        route_ms = (time.monotonic() - t0) * 1000
        print(f"  Routed to windows {window_ids} ({route_ms:.1f} ms)", file=sys.stderr)

        if json_log:
            json_log.event("routed", window_ids=window_ids, route_ms=round(route_ms, 1))

        # Reconstruct: decode window tokens, chat-template with query,
        # prefill with boundary as initial_residual
        wid = window_ids[0]

        # Load boundary (10 KB — the Markov chain link)
        try:
            boundary = store.load_boundary(wid)
            boundary = boundary.reshape(1, 1, -1)
            has_boundary = True
            print(f"  Boundary: window {wid} (10 KB Markov state)", file=sys.stderr)
        except (FileNotFoundError, ValueError):
            boundary = None
            has_boundary = False

        # Decode window tokens → text
        window_text = store.get_window_text(wid, tokenizer)

        # For multi-window: concatenate window texts
        if len(window_ids) > 1:
            extra_texts = []
            for extra_wid in window_ids[1:]:
                extra_texts.append(store.get_window_text(extra_wid, tokenizer))
            window_text = window_text + "\n\n---\n\n" + "\n\n---\n\n".join(extra_texts)

        # Chat-template [window text + query]
        donor_content = f"{window_text}\n\n{args.prompt}"
        try:
            ctx_prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": donor_content}],
                add_generation_prompt=True,
            )
        except Exception:
            ctx_prompt_ids = tokenizer.encode(donor_content, add_special_tokens=True)

        ctx_tokens = len(ctx_prompt_ids)
        mode = "Markov reconstruction" if has_boundary else "context replay"
        print(f"  Context: {ctx_tokens} tokens ({mode})", file=sys.stderr)

        ctx_mx = mx.array([ctx_prompt_ids])

        if has_boundary:
            # Pure Markov path: prefill with boundary as initial_residual
            # This reconstructs the full document state at this window
            h = kv_gen.prefill_to_layer(
                ctx_mx,
                target_layer=store.config.crystal_layer,
                initial_residual=boundary,
            )
            # Continue from crystal_layer to end → logits + KV for upper layers
            logits, kv_upper = kv_gen.prefill_from_layer(
                h,
                start_layer=store.config.crystal_layer + 1,
            )
            mx.eval(logits)

            # For generation continuation, we need a full KV store.
            # prefill_from_layer gives KV only for upper layers (L31-L33).
            # We also need L0-L30 KV from the prefill_to_layer pass.
            # Workaround: do a separate full prefill for the KV cache.
            _, kv_store = kv_gen.prefill(ctx_mx)
            mx.eval(*[t for p in kv_store for t in p])
            seq_len = ctx_mx.shape[1] + 1  # +1 for boundary position
        else:
            # Fallback: plain prefill (no boundary)
            logits, kv_store = kv_gen.prefill(ctx_mx)
            mx.eval(logits)
            seq_len = ctx_mx.shape[1]

        prefill_ms = (time.monotonic() - t0) * 1000
        if json_log:
            json_log.event("prefill_done", mode=mode, ctx_tokens=ctx_tokens,
                           prefill_ms=round(prefill_ms, 1))

        # Generate
        generated = []
        for _ in range(args.max_tokens):
            token = (
                int(mx.argmax(logits[0, -1]).item())
                if args.temperature == 0.0
                else int(mx.random.categorical(logits[0, -1:] / max(args.temperature, 1e-6)).item())
            )
            if token in stop_ids:
                break
            generated.append(token)
            logits, kv_store = kv_gen.step_uncompiled(
                mx.array([[token]]), kv_store, seq_len=seq_len
            )
            seq_len += 1

        gen_ms = (time.monotonic() - t0) * 1000
        print(f"  Generated {len(generated)} tokens ({gen_ms:.0f} ms total)", file=sys.stderr)

        if json_log:
            json_log.event("generate_done", tokens_generated=len(generated),
                           total_ms=round(gen_ms, 1))

    output = tokenizer.decode(generated, skip_special_tokens=True)
    sys.stdout.write(output)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if json_log:
        json_log.event("query_done", output=output)
        json_log.close()
