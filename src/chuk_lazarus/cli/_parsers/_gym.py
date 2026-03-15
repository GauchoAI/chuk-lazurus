"""Gym command parsers."""

from ...commands.gym import gym_info, gym_run


def register_gym_parsers(subparsers):
    """Register the gym subcommand and its sub-subcommands."""
    gym_parser = subparsers.add_parser("gym", help="Gym streaming utilities")
    gym_subparsers = gym_parser.add_subparsers(dest="gym_command", help="Gym commands")

    # Gym run command
    gym_run_parser = gym_subparsers.add_parser(
        "run", help="Run gym episode streaming and collect samples"
    )
    gym_run_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    gym_run_parser.add_argument("--host", default="localhost", help="Gym server host")
    gym_run_parser.add_argument("--port", type=int, default=8023, help="Gym server port")
    gym_run_parser.add_argument(
        "--transport",
        choices=["telnet", "websocket", "http"],
        default="telnet",
        help="Transport protocol",
    )
    gym_run_parser.add_argument(
        "--output-mode",
        choices=["json", "text", "binary"],
        default="json",
        help="Output format from gym",
    )
    gym_run_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size",
    )
    gym_run_parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout")
    gym_run_parser.add_argument("--retries", type=int, default=3, help="Max connection retries")
    gym_run_parser.add_argument(
        "--difficulty-min",
        type=float,
        default=0.0,
        help="Minimum puzzle difficulty",
    )
    gym_run_parser.add_argument(
        "--difficulty-max",
        type=float,
        default=1.0,
        help="Maximum puzzle difficulty",
    )
    gym_run_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to collect (infinite if not set)",
    )
    gym_run_parser.add_argument("--output", "-o", help="Output file for buffer (JSON)")
    gym_run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Mock mode for testing
    gym_run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock gym stream for testing",
    )
    gym_run_parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes (mock mode)",
    )
    gym_run_parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=5,
        help="Steps per episode (mock mode)",
    )
    gym_run_parser.add_argument(
        "--success-rate",
        type=float,
        default=0.7,
        help="Success rate (mock mode)",
    )
    gym_run_parser.set_defaults(func=gym_run)

    # Gym info command
    gym_info_parser = gym_subparsers.add_parser(
        "info", help="Display gym stream configuration info"
    )
    gym_info_parser.set_defaults(func=gym_info)
