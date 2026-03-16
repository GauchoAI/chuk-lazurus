#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# COLD OPEN — Two queries against 370,000 tokens of Apollo 11 transcript
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHUK_MLX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT="${APOLLO_CHECKPOINT:-/Users/christopherhay/chris-source/apollo-demo/apollo11_lean}"
MODEL="google/gemma-3-4b-it"

echo ""
echo "  Query 1: What sport and teams were discussed?"
echo ""

uv run --directory "$CHUK_MLX_ROOT" python -c "
from chuk_lazarus.cli.main import main
import sys
sys.argv = [
    'lazarus', 'context', 'generate',
    '--model', '$MODEL',
    '--checkpoint', '$CHECKPOINT',
    '--prompt', 'What sport and teams were discussed during the mission?',
    '--strategy', 'sparse',
    '--top-k', '5',
    '--max-tokens', '300',
    '--temperature', '0',
]
main()
"

echo ""
echo "  Query 2: What did the crew say about audio quality?"
echo ""

uv run --directory "$CHUK_MLX_ROOT" python -c "
from chuk_lazarus.cli.main import main
import sys
sys.argv = [
    'lazarus', 'context', 'generate',
    '--model', '$MODEL',
    '--checkpoint', '$CHECKPOINT',
    '--prompt', 'What did the crew say about the audio quality during the early part of the mission?',
    '--strategy', 'sparse',
    '--top-k', '5',
    '--max-tokens', '300',
    '--temperature', '0',
]
main()
"

echo ""
echo "  370,000 tokens. On a MacBook. 4-billion parameter model."
echo ""
