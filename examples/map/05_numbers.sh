#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Section 5 — The Numbers (live timing comparison)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHUK_MLX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT="${APOLLO_CHECKPOINT:-/Users/christopherhay/chris-source/apollo-demo/apollo11_lean}"
MODEL="google/gemma-3-4b-it"

echo ""
echo "  Same question. Same answer. How fast?"
echo ""
echo "  Replaying 5 windows through the model:"
echo ""

time uv run --directory "$CHUK_MLX_ROOT" python -c "
from chuk_lazarus.cli.main import main
import sys
sys.argv = [
    'lazarus', 'context', 'generate',
    '--model', '$MODEL',
    '--checkpoint', '$CHECKPOINT',
    '--prompt', 'What ship recovered the Apollo 11 crew?',
    '--strategy', 'sparse',
    '--top-k', '5',
    '--max-tokens', '100',
    '--temperature', '0',
]
main()
"

echo ""
echo "  Library on disk:"
du -sh "$CHECKPOINT/"
ls -lh "$CHECKPOINT/sparse_index.json"
echo ""
