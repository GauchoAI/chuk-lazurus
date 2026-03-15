"""Standard mode — normal window replay with preamble framing."""

from __future__ import annotations

import sys
import time


def run_standard(lib, kv_gen, replay_ids, preamble_ids, mx):
    """Normal window replay: preamble + transcript windows.

    Returns (context_kv, seq_len).
    """
    seq_len = 0
    context_kv = None

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

    return context_kv, seq_len
