"""Agentic navigation mode — the model explores the document using the compass."""

from __future__ import annotations

import re
import sys
import time

from ..._types import GenerateResult
from ...compass_routing import RoutingStrategy, compass_route


def run_explore(
    lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx,
):
    """Agentic navigation: the model explores the document using the compass.

    The compass map is shown to the model. The model generates text and
    can request windows to read by emitting <READ:NNN>. The engine
    replays the requested window into the KV cache and the model continues.

    Handles its own generation and returns directly.
    """
    # Get compass scores for the map
    strategy_arg = getattr(args, "strategy", None)
    strategy = RoutingStrategy(strategy_arg) if strategy_arg else RoutingStrategy.GEOMETRIC
    compass_scores = compass_route(
        lib, kv_gen, prompt_ids, prompt_text, tokenizer,
        model_config=pipeline.config,
        strategy=strategy,
        top_k=20,  # show top 20 on the map
    )

    # Build the compass map as text
    map_lines = ["Document map (725 windows, 370K tokens of Apollo 11 transcript):"]
    map_lines.append("Top windows by geometric relevance to your query:")
    for i, wid in enumerate(compass_scores[:20]):
        w = lib.windows[wid]
        preview = w.preview[:60]
        map_lines.append(f"  Window {wid}: {preview}")
    map_lines.append("")
    map_lines.append("To read a window's full content, output: <READ:NNN>")
    map_lines.append("Read windows that seem relevant, then answer the question.")
    compass_map = "\n".join(map_lines)

    # Build prompt with compass map
    no_chat = getattr(args, "no_chat_template", False)
    system_prompt = getattr(args, "system_prompt", None)
    sys_content = system_prompt or (
        "You are exploring a document transcript. You have a compass map showing the "
        "most relevant windows. You can read any window by outputting <READ:NNN> where "
        "NNN is the window number. Read 2-3 windows that seem relevant, then answer "
        "the question using exact quotes from what you read."
    )

    explore_prompt = (
        f"<start_of_turn>user\n{sys_content}\n\n"
        f"{compass_map}\n\n"
        f"Question: {prompt_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    explore_ids = tokenizer.encode(explore_prompt, add_special_tokens=True)

    # Prefill the explore prompt
    e_ids = mx.array(explore_ids)[None]
    logits, gen_kv = kv_gen.prefill(e_ids)
    mx.eval(logits, *[t for pair in gen_kv for t in pair])
    seq_len = len(explore_ids)
    print(f"  Explore prompt: {seq_len} tokens", file=sys.stderr)

    # Agentic generation loop
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    generated_tokens: list[int] = []
    generated_text = ""
    windows_read: set[int] = set()
    max_reads = 5  # safety limit

    for _ in range(config.max_tokens):
        last_logits = logits[0, -1]
        if config.temperature == 0.0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / config.temperature
            next_token = int(mx.random.categorical(scaled[None]).item())

        if next_token in stop_ids:
            break

        generated_tokens.append(next_token)
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        sys.stdout.write(token_text)
        sys.stdout.flush()
        generated_text += token_text

        # Check for <READ:NNN> pattern (flexible — 4B models may omit closing >)
        read_match = re.search(r'<READ:(\d+)(?:>|\s|\n)', generated_text)
        if read_match and len(windows_read) < max_reads:
            read_wid = int(read_match.group(1))
            if 0 <= read_wid < lib.num_windows and read_wid not in windows_read:
                windows_read.add(read_wid)
                # Strip the <READ:NNN> from tracked text
                generated_text = generated_text[:read_match.start()]

                # Replay the requested window
                w_tokens = lib.get_window_tokens(read_wid)
                w_ids = mx.array(w_tokens)[None]
                t0 = time.time()
                _logits, gen_kv = kv_gen.extend(w_ids, gen_kv, abs_start=seq_len)
                mx.eval(*[t for pair in gen_kv for t in pair])
                seq_len += len(w_tokens)
                elapsed_ms = (time.time() - t0) * 1000
                print(
                    f"\n  [Read window {read_wid}: {len(w_tokens)} tokens injected "
                    f"@ pos {seq_len - len(w_tokens)}–{seq_len - 1} ({elapsed_ms:.0f}ms)]",
                    file=sys.stderr, flush=True,
                )

                # Inject continuation prompt so model resumes analysis
                cont_text = (
                    f"\n[Window {read_wid} content loaded. "
                    f"Windows read so far: {sorted(windows_read)}. "
                    f"Continue your analysis.]\n"
                )
                cont_ids = tokenizer.encode(cont_text, add_special_tokens=False)
                cont_mx = mx.array(cont_ids)[None]
                logits, gen_kv = kv_gen.extend(cont_mx, gen_kv, abs_start=seq_len)
                mx.eval(logits, *[t for pair in gen_kv for t in pair])
                seq_len += len(cont_ids)
                continue

        logits, gen_kv = kv_gen.step_uncompiled(
            mx.array([[next_token]]), gen_kv, seq_len=seq_len,
        )
        seq_len += 1

    print()

    context_tokens = seq_len
    result = GenerateResult(
        response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        tokens_generated=len(generated_tokens),
        context_tokens=context_tokens,
    )
    print(result.to_display())
