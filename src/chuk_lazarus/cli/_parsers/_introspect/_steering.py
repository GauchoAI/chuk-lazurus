"""Steer introspect parser."""

from ....commands.introspect import introspect_steer


def register_steering_parsers(introspect_subparsers):
    """Register steer subcommand."""
    steer_parser = introspect_subparsers.add_parser(
        "steer",
        help="Apply activation steering to manipulate model behavior",
        description="""Activation steering: modify model behavior by adding learned directions.

Three modes of operation:
1. Extract direction: --extract --positive "good" --negative "bad" -o direction.npz
2. Apply direction: --direction direction.npz -p "prompt"
3. Compare coefficients: --direction direction.npz -p "prompt" --compare "-1,0,1"
        """,
    )
    steer_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    steer_parser.add_argument(
        "--prompts",
        "-p",
        help="Prompts to steer (pipe-separated or @file.txt)",
    )
    steer_parser.add_argument(
        "--direction",
        "-d",
        help="Path to direction file (.npz or .json)",
    )
    steer_parser.add_argument(
        "--neuron",
        type=int,
        help="Single neuron index to steer (creates a one-hot direction)",
    )
    steer_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to apply steering (default: auto from direction or middle)",
    )
    steer_parser.add_argument(
        "--coefficient",
        "-c",
        "--strength",
        type=float,
        default=1.0,
        help="Steering coefficient/strength (default: 1.0, negative = toward negative class)",
    )
    steer_parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract direction from contrastive prompts (requires --positive and --negative)",
    )
    steer_parser.add_argument(
        "--positive",
        help="Positive class prompt (for direction extraction or on-the-fly steering)",
    )
    steer_parser.add_argument(
        "--negative",
        help="Negative class prompt (for direction extraction or on-the-fly steering)",
    )
    steer_parser.add_argument(
        "--compare",
        help="Compare outputs at multiple coefficients (comma-separated, e.g., '-2,-1,0,1,2')",
    )
    steer_parser.add_argument(
        "--name",
        help="Name for the direction (for logging)",
    )
    steer_parser.add_argument(
        "--positive-label",
        help="Label for positive class (default: 'positive')",
    )
    steer_parser.add_argument(
        "--negative-label",
        help="Label for negative class (default: 'negative')",
    )
    steer_parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )
    steer_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0 = greedy)",
    )
    steer_parser.add_argument(
        "--output",
        "-o",
        help="Save results/direction to file",
    )
    steer_parser.set_defaults(func=introspect_steer)
