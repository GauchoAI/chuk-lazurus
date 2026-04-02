"""knowledge init — Create a base state from a system prompt."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


async def knowledge_init_cmd(args: Namespace) -> None:
    """Create a base state anchor from a system prompt file."""
    from ....inference.context.knowledge import ArchitectureConfig
    from ....inference.context.knowledge.append import build_base_state
    from ._common import load_model

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: system prompt file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    pipeline, kv_gen, tokenizer = load_model(args.model)

    ac = ArchitectureConfig.from_model_config(pipeline.config)
    print(
        f"  Architecture: crystal_layer=L{ac.crystal_layer}, window={ac.window_size}",
        file=sys.stderr,
    )

    system_prompt = input_path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(system_prompt, add_special_tokens=True)
    print(f"  System prompt: {len(tokens)} tokens ({len(system_prompt)} chars)", file=sys.stderr)

    build_base_state(kv_gen, tokenizer, system_prompt, ac, output_path)
    print(f"  Store initialized at {output_path}", file=sys.stderr)
