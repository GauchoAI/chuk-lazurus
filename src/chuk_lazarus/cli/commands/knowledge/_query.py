"""knowledge query — TF-IDF routing + context replay (fact sentence or focused passage)."""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path


async def knowledge_query_cmd(args: Namespace) -> None:
    """Route (top-k adaptive) → context replay → generate.

    Uses fact sentences (~50 tokens) when available (v12 stores).
    Falls back to focused passages (~200 tokens) from window token lists.
    """
    import mlx.core as mx

    from ....inference.context.knowledge import KnowledgeStore
    from ....inference.context.knowledge.inject import _extract_focused_passage
    from ._common import generate_plain, load_model, prepare_prompt, stop_token_ids

    store_path = Path(args.store)
    if not store_path.exists():
        print(f"Error: knowledge store not found: {store_path}", file=sys.stderr)
        sys.exit(1)

    _, kv_gen, tokenizer = load_model(args.model)
    print(f"Loading knowledge store: {store_path}", file=sys.stderr)
    store = KnowledgeStore.load(store_path)
    store.log_stats()

    prompt_ids = prepare_prompt(tokenizer, args.prompt)
    stop_ids = stop_token_ids(tokenizer)

    t0 = time.monotonic()
    window_ids = store.route_top_k(args.prompt, tokenizer, k=args.top_k)

    if not window_ids:
        print("  No matching windows — generating plain.", file=sys.stderr)
        generated = generate_plain(kv_gen, prompt_ids, args.max_tokens, stop_ids)
    else:
        route_ms = (time.monotonic() - t0) * 1000
        print(f"  Routed to windows {window_ids} ({route_ms:.1f} ms)", file=sys.stderr)

        # Build context from top-k windows using focused passages
        qtids = set(tokenizer.encode(args.prompt, add_special_tokens=False))
        passages = []
        for wid in window_ids:
            wt_list = store.window_token_lists.get(wid)
            if wt_list and store.idf:
                text = _extract_focused_passage(wt_list, qtids, store.idf, tokenizer, radius=100)
                passages.append(text)

        combined_context = "\n\n---\n\n".join(passages)
        ctx_tokens = tokenizer.encode(combined_context, add_special_tokens=False)
        source = "focused passages"
        print(f"  Context: {len(ctx_tokens)} tokens from {len(passages)} windows ({source})",
              file=sys.stderr)

        # Chat-template [context + query] → prefill → generate
        donor_content = f"{combined_context}\n\n{args.prompt}"
        try:
            ctx_prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": donor_content}],
                add_generation_prompt=True,
            )
        except Exception:
            ctx_prompt_ids = tokenizer.encode(donor_content, add_special_tokens=True)

        ctx_mx = mx.array([ctx_prompt_ids])
        logits, kv_store = kv_gen.prefill(ctx_mx)
        mx.eval(logits)
        seq_len = ctx_mx.shape[1]

        generated = []
        for _ in range(args.max_tokens):
            token = int(mx.argmax(logits[0, -1]).item()) if args.temperature == 0.0 else \
                    int(mx.random.categorical(logits[0, -1:] / max(args.temperature, 1e-6)).item())
            if token in stop_ids:
                break
            generated.append(token)
            logits, kv_store = kv_gen.step_uncompiled(
                mx.array([[token]]), kv_store, seq_len=seq_len)
            seq_len += 1

        gen_ms = (time.monotonic() - t0) * 1000
        print(f"  Generated {len(generated)} tokens ({gen_ms:.0f} ms total)", file=sys.stderr)

    output = tokenizer.decode(generated, skip_special_tokens=True)
    sys.stdout.write(output)
    sys.stdout.write("\n")
    sys.stdout.flush()
