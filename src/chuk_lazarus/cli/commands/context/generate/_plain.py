"""Plain generation — no checkpoint library, just model + prompt."""

from __future__ import annotations

import sys
from argparse import Namespace

from .._types import GenerateConfig, GenerateResult


def _plain_generate(config: GenerateConfig, args: Namespace) -> GenerateResult:
    """Generate text from a model without any checkpoint context."""
    from .....inference import UnifiedPipeline

    prompt_text = config.prompt_text
    if not prompt_text:
        print("Error: no prompt specified. Use --prompt or --prompt-file.", file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"Loading model: {config.model}", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(config.model, verbose=False)

    no_chat = getattr(args, "no_chat_template", False)
    system_prompt = getattr(args, "system_prompt", None)

    # Generate via chat or raw depending on flags
    if no_chat:
        result = pipeline.generate(
            prompt_text,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    else:
        result = pipeline.chat(
            prompt_text,
            system_message=system_prompt,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    return GenerateResult(
        response=result.text,
        tokens_generated=result.stats.output_tokens,
        context_tokens=result.stats.input_tokens,
    )
