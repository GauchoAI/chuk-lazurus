"""knowledge query — Query a knowledge store with persistent 1D injection."""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path


async def knowledge_query_cmd(args: Namespace) -> None:
    """Query a knowledge store and generate with persistent injection."""
    import mlx.core as mx

    from ....inference.context.knowledge import (
        KnowledgeStore,
        generate_with_injection,
    )
    from ._common import generate_plain, load_model, prepare_prompt, stop_token_ids

    store_path = Path(args.store)
    if not store_path.exists():
        print(f"Error: knowledge store not found: {store_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load model + store ────────────────────────────────────────────
    _, kv_gen, tokenizer = load_model(args.model)
    print(f"Loading knowledge store: {store_path}", file=sys.stderr)
    store = KnowledgeStore.load(store_path)
    store.log_stats()

    # ── Route + generate ──────────────────────────────────────────────
    prompt_ids = prepare_prompt(tokenizer, args.prompt)
    stop_ids = stop_token_ids(tokenizer)

    t0 = time.monotonic()
    window_id = store.route(args.prompt, tokenizer=tokenizer)

    if window_id is None:
        print("  No matching window found — generating plain.", file=sys.stderr)
        generated = generate_plain(kv_gen, prompt_ids, args.max_tokens, stop_ids)
    else:
        entries = store.get_entries_for_query(window_id, args.prompt, tokenizer)
        route_ms = (time.monotonic() - t0) * 1000
        print(
            f"  Routed to window {window_id} ({len(entries)} entries, {route_ms:.1f} ms)",
            file=sys.stderr,
        )
        inject_text = "  Inject: " + " → ".join(
            f"token={tokenizer.decode([e.token_id])} coeff={e.coefficient:.0f}"
            for e in entries[:5]
        )
        print(inject_text, file=sys.stderr)

        generated = generate_with_injection(
            kv_gen=kv_gen,
            prompt_ids=mx.array(prompt_ids)[None],
            entries=entries,
            config=store.config,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop_ids=stop_ids,
        )
        gen_ms = (time.monotonic() - t0) * 1000
        print(f"  Generated {len(generated)} tokens ({gen_ms:.0f} ms)", file=sys.stderr)

    output = tokenizer.decode(generated, skip_special_tokens=True)
    sys.stdout.write(output)
    sys.stdout.write("\n")
    sys.stdout.flush()
