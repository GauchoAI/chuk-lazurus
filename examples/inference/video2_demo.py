"""Video 2 demo — Mode 7 auto-selects the fast path.

Runs four representative queries against the Apollo 11 checkpoint.
For FACTUAL queries the engine tries vec_inject first (200-330ms).
For non-factual queries it routes through the full Mode 7 pipeline.

Routing architecture (three-tier):
  Stage 1: Adaptive Q·K threshold — structurally distinctive queries (~40%)
  Stage 2: H4 output cosine — same-template entity discrimination (~45%)
           The same head that retrieves the fact also addresses it.
           Margins 2.4×–8.4× at N=12 same-template cluster.
  Stage 3: Replay fallback — entity-implicit queries (~15%)

Combined: ~85% injection rate at any N. No string matching. Pure model geometry.

Usage
-----
    uv run python examples/inference/video2_demo.py \\
        --checkpoint /path/to/apollo11_full/ \\
        --model google/gemma-3-4b-it

No --replay flag. Mode 7 decides. The dark space routes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


QUERIES = [
    # (label, prompt, expected_path)
    # S1:K-space — fast injection path
    ("FACTUAL — launch date",
     "When did the Apollo 11 mission launch?",
     "vec_inject"),
    # S1:K-space — crew question
    ("FACTUAL — command module pilot",
     "Who was the Command Module Pilot on Apollo 11?",
     "vec_inject"),
    # S1:K-space — landing site
    ("FACTUAL — landing site",
     "Where did the Apollo 11 lunar module land on the Moon?",
     "vec_inject"),
]


def _run_query(checkpoint: str, model: str, prompt: str) -> tuple[str, float]:
    """Run a single query and return (stderr_output, elapsed_s)."""
    # Find lazarus in the same venv as the current interpreter
    lazarus = Path(sys.executable).parent / "lazarus"
    cmd = [
        str(lazarus),
        "context", "generate",
        "--checkpoint", checkpoint,
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", "80",
        "--temperature", "0",
    ]
    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0
    return result.stderr, result.stdout, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Video 2 demo — Mode 7 + vec_inject")
    parser.add_argument("--checkpoint", "-c", required=True, help="Apollo 11 checkpoint dir")
    parser.add_argument("--model", "-m", default="google/gemma-3-4b-it", help="Model ID")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  VIDEO 2: Mode 7 + Vec Inject — The Filing Cabinet Is Empty")
    print("  One command. No flags. The model routes.")
    print("  Stage 1: Adaptive Q·K (K-space)  →  Stage 3: Replay fallback")
    print("  Stage 2: H4 copy head fires at scale (N≥500+ facts, same-template)")
    print("=" * 70)
    print()

    for label, prompt, expected in QUERIES:
        print(f"{'─' * 70}")
        print(f"  {label}")
        print(f"  Q: {prompt!r}")
        print()

        stderr, stdout, elapsed = _run_query(args.checkpoint, args.model, prompt)

        # Extract key routing lines from stderr
        for line in stderr.splitlines():
            if any(kw in line for kw in [
                "Query classification",
                "Vec inject",
                "INJECT:",
                "Routing:",
                "Inject 2-pass",
                "First token (injected)",
                "falling back",
                "Replaying",
            ]):
                print(f"  {line.strip()}")

        # stdout contains only the streamed response (stats printed separately by lazarus)
        response = stdout.strip()
        # Drop the stats line if it leaked into stdout capture
        response_lines = [l for l in response.splitlines() if not l.startswith("  Stats:")]
        print()
        print(f"  A: {chr(10).join(response_lines).strip()}")
        print(f"  [{elapsed:.1f}s total]")
        print()

    print("=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
