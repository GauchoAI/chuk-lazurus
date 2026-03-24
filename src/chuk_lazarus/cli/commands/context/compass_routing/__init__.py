"""
Compass routing strategies for automatic window selection.

Four strategies, from cheapest to most expensive:

  1. BM25 (token-level)
     Decode each window's tokens, score against the query text using
     BM25.  Fast, content-aware, works on any query shape.  No model
     inference required.

  2. Residual deflection (geometric)
     Extend the query against each window's 1-token checkpoint.
     Measure how much the query residual shifts (L2 from bare query
     residual).  One cheap extend per window.

  3. Preview (model-routed)
     For each window, prefill a compressed preview (first + last N
     tokens), extend the query against it, and measure the model's
     logit entropy on the first predicted token.  Low entropy = the
     model is confident = it found relevant content.  The retrieval
     circuit fires on the preview content and routes itself.

  4. Hybrid (BM25 pre-filter → preview re-rank)
     BM25 narrows to ~10 candidates.  Preview scoring re-ranks.
     Best of both: fast keyword pre-filter + model-level routing.

Usage in context_generate_cmd:
    from .compass_routing import compass_route, RoutingStrategy

    replay_ids = compass_route(
        lib, kv_gen, prompt_ids, prompt_text, tokenizer,
        strategy=RoutingStrategy.HYBRID,
        top_k=3,
    )
"""

from ._orchestrator import compass_route
from ._strategy import RoutingStrategy
from ._twopass import two_pass_generate

__all__ = ["compass_route", "two_pass_generate", "RoutingStrategy"]
