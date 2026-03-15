"""Resolve --replay and --find flags into window IDs."""

from __future__ import annotations

import sys


def _resolve_replay(
    lib: object,
    tokenizer: object,
    replay_arg: list[str] | None,
    find_term: str | None,
) -> list[int] | None:
    """Resolve --replay and --find flags into a list of window IDs.

    Returns None to signal "use auto compass routing".
    """
    num_windows = lib.num_windows

    # --find takes priority: locate the window containing the term
    if find_term:
        wid = lib.find_window_for_term(find_term, tokenizer)
        if wid is not None:
            print(f"  Found '{find_term}' in window {wid}", file=sys.stderr)
            return [wid]
        else:
            print(f"  Warning: '{find_term}' not found in any window, using auto", file=sys.stderr)
            return None

    # --replay: parse the argument
    if replay_arg is not None:
        if len(replay_arg) == 1:
            val = replay_arg[0]
            if val == "auto":
                return None
            elif val == "all":
                return list(range(num_windows))
            elif val == "last":
                return [num_windows - 1] if num_windows > 0 else []
            elif val == "accumulated":
                return ["accumulated"]  # special marker
            elif val == "compressed":
                return ["compressed"]  # full document via page injection
            elif val == "explore":
                return ["explore"]  # agentic navigation
            elif val == "inject":
                return ["inject"]  # L26 residual injection
            else:
                try:
                    return [int(val)]
                except ValueError:
                    print(f"  Warning: invalid replay value '{val}', using auto", file=sys.stderr)
                    return None
        else:
            # Multiple window IDs: --replay 0 1 45
            ids = []
            for v in replay_arg:
                try:
                    ids.append(int(v))
                except ValueError:
                    pass
            return ids if ids else None

    # Default: auto compass routing
    return None
