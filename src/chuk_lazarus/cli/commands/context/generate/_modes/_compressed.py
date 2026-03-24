"""Compressed (page injection) mode — hybrid full + compressed windows."""

from __future__ import annotations

import sys
import time

from ...compass_routing import RoutingStrategy, compass_route


def run_compressed(lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, args, mx):
    """Hybrid: compass-routed windows at full 512 tokens +
    everything else as compressed pre-RoPE pages.

    Top-k windows: full replay -> 512 KV entries each (precise attention)
    Remaining windows: 8 pre-RoPE pages each (document awareness)

    The model sees the whole document. It focuses on what matters.

    Returns (context_kv, seq_len).
    """
    n_pages = 8
    strategy_arg = getattr(args, "strategy", None)
    top_k_override = getattr(args, "top_k", None)
    strategy = RoutingStrategy(strategy_arg) if strategy_arg else RoutingStrategy.GEOMETRIC
    top_k = top_k_override if top_k_override is not None else 3

    # Route: find the windows that matter
    routed_ids = compass_route(
        lib,
        kv_gen,
        prompt_ids,
        prompt_text,
        tokenizer,
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
        print(
            "  No pre-stored pages — computing on-the-fly (use --store-pages during prefill)",
            file=sys.stderr,
        )
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
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        print(file=sys.stderr)

    # Build KV cache: compressed pages first (background), then full windows (foreground)
    # Compressed pages at positions 0..N_compressed-1
    total_compressed = len(all_pages)
    target_offsets = list(range(total_compressed))

    print(
        f"  Injecting {total_compressed} compressed pages "
        f"({compressed_count} windows × {n_pages} pages)",
        file=sys.stderr,
        flush=True,
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

    return context_kv, seq_len
