"""Vector injection primitives — Experiment 2bd41b18.

The model's L29 H4 head copies a scalar projection of the answer token's
embedding direction into the residual stream.  The full information content
of one retrieved fact is 12 bytes:

    token_id   : int32   — the answer token
    coefficient: float32 — c = dot(R_L30, embed(token_id))

vec_inject() adds c × (e / ‖e‖²) to the residual at layer 30, where
e = embed(token_id).  This achieves KL = 0.000031 vs. full KV replay.
"""

from __future__ import annotations

import mlx.core as mx

from ._types import VecInjectMatch


def vec_inject(
    h: mx.array,
    token_id: int,
    coefficient: float,
    embed_matrix: mx.array,
) -> mx.array:
    """Add one fact's contribution to residual h at a single token position.

    Parameters
    ----------
    h            : (1, 1, hidden_size) — residual at the injection position.
    token_id     : Vocabulary index of the answer token.
    coefficient  : c from vec_inject.npz (dot product at L30 during prefill).
    embed_matrix : (vocab_size, hidden_size) — model embedding table (scaled).

    Returns
    -------
    (1, 1, hidden_size) — modified residual.

    Notes
    -----
    direction = e / ‖e‖²  (not e / ‖e‖) because c = dot(R, e), not
    dot(R, e / ‖e‖).  The denominator ‖e‖² cancels the scale so we
    reproduce R's component along e exactly.
    """
    e = embed_matrix[token_id]       # (hidden_size,)
    direction = e / mx.sum(e * e)    # (hidden_size,)
    return h + (coefficient * direction)[None, None, :]


def vec_inject_all(
    h: mx.array,
    matches: list[VecInjectMatch],
    embed_matrix: mx.array,
) -> mx.array:
    """Apply all fact injections to residual h at one token position.

    Parameters
    ----------
    h            : (1, 1, hidden_size)
    matches      : Retrieved facts (from VecInjectResult.matches).
    embed_matrix : (vocab_size, hidden_size)

    Returns
    -------
    (1, 1, hidden_size) with every fact's contribution added.
    Additions are independent — linear superposition holds.
    """
    for m in matches:
        h = vec_inject(h, m.token_id, m.coefficient, embed_matrix)
    return h
