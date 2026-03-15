"""
Experiment: Does the accumulated residual improve compass routing?

The checkpoint chain builds an accumulated L26 residual across all 725 windows.
After processing 370K tokens, that residual carries the geometric traces of
the entire document — the path coordinates of everything the model walked through.

Test: combine the accumulated residual with the query residual, then route.
Compare rankings to bare-query routing. If the accumulated state shifts
rankings toward actually relevant windows — it's working.

Three routing modes:
  1. Bare query (current default)
  2. Accumulated residual only (document state, no query)
  3. Combined: accumulated + query (document-aware query)
"""

import sys
import time
import math
import numpy as np
import mlx.core as mx


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default="./apollo11_ctx_512")
    parser.add_argument("--model", "-m", default="google/gemma-3-4b-it")
    cli_args = parser.parse_args()

    from chuk_lazarus.inference import UnifiedPipeline
    from chuk_lazarus.inference.context import CheckpointLibrary
    from chuk_lazarus.inference.context.unlimited_engine import UnlimitedContextEngine

    # Load compass_routing.py directly to avoid CLI __init__ chain
    # (which pulls in chuk_virtual_expert, not installed in dev env).
    import importlib.util
    _cr_path = str(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "src" / "chuk_lazarus" / "cli" / "commands" / "context" / "compass_routing.py"
    )
    _spec = importlib.util.spec_from_file_location("compass_routing", _cr_path)
    _cr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cr)
    compass_route = _cr.compass_route
    RoutingStrategy = _cr.RoutingStrategy

    print(f"Loading library: {cli_args.checkpoint}")
    lib = CheckpointLibrary(cli_args.checkpoint)
    print(f"  {lib.num_windows} windows, {lib.total_tokens:,} tokens")
    compass_layer = lib.compass_layer
    print(f"  Compass layer: L{compass_layer}")

    print(f"Loading model: {cli_args.model}")
    pipeline = UnifiedPipeline.from_pretrained(cli_args.model, verbose=False)
    tokenizer = pipeline.tokenizer
    engine = UnlimitedContextEngine(pipeline.model, pipeline.config, window_size=lib.window_size)
    engine.load_library(lib)
    kv_gen = engine.kv_gen

    # ── Load accumulated residual ──
    # Last sample of last window = final position after 370K tokens
    last_wid = lib.num_windows - 1
    compass_samples = lib.get_compass_residuals(last_wid)
    accumulated_vec = compass_samples[-1]  # last sample of last window
    acc_np = np.array(accumulated_vec.reshape(-1).tolist(), dtype=np.float32)
    print(f"  Accumulated residual: W{last_wid} sample {len(compass_samples)-1}, shape {accumulated_vec.shape}")

    # ── Test queries ──
    queries = [
        "Find 3 amusing moments from the transcript",
        "What sport and teams were discussed during the mission?",
        "What did the astronauts see when they looked at Earth?",
    ]

    # Known relevant windows for each query
    relevant = {
        "Find 3 amusing moments from the transcript": {170, 64, 313, 76},
        "What sport and teams were discussed during the mission?": {170, 64, 76},
        "What did the astronauts see when they looked at Earth?": {118, 129, 184},
    }

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print(f"{'=' * 70}")

        # Encode query
        messages = [{"role": "user", "content": query}]
        query_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(query_text, add_special_tokens=False)

        # Extract bare query L26
        q_ids = mx.array(prompt_ids)[None]
        q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
        mx.eval(q_h)
        bare_np = np.array(q_h[0, -1, :].tolist(), dtype=np.float32)

        # Combined: mean of accumulated + query (simple fusion)
        combined_np = (acc_np + bare_np) / 2.0

        # Also try: accumulated weighted more heavily (document knows more)
        weighted_np = 0.3 * bare_np + 0.7 * acc_np

        print(f"\n  Residual angles:")
        print(f"    Bare query ↔ Accumulated:      {angle_degrees(bare_np, acc_np):.2f}°")
        print(f"    Bare query ↔ Combined (50/50):  {angle_degrees(bare_np, combined_np):.2f}°")
        print(f"    Bare query ↔ Weighted (30/70):  {angle_degrees(bare_np, weighted_np):.2f}°")

        # Route with each mode
        modes = [
            ("Bare query", None),
            ("Accumulated only", mx.array(acc_np.reshape(1, -1))),
            ("Combined 50/50", mx.array(combined_np.reshape(1, -1))),
            ("Weighted 30/70", mx.array(weighted_np.reshape(1, -1))),
        ]

        target_wids = relevant.get(query, set())

        for mode_name, residual in modes:
            t0 = time.time()
            routed = compass_route(
                lib, kv_gen, prompt_ids, query, tokenizer,
                model_config=pipeline.config,
                strategy=RoutingStrategy.GEOMETRIC,
                top_k=5,
                query_residual=residual,
            )
            elapsed_ms = (time.time() - t0) * 1000

            # Check how many target windows are in top 5
            hits = [wid for wid in routed if wid in target_wids]
            print(f"\n  {mode_name} ({elapsed_ms:.0f}ms):")
            print(f"    Top 5: {routed}")
            if target_wids:
                print(f"    Target hits in top 5: {hits} ({len(hits)}/{len(target_wids)})")


if __name__ == "__main__":
    main()
