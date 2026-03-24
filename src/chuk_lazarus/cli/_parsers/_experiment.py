"""Experiment command parsers."""

from ..commands.experiment import (
    experiment_info,
    experiment_list,
    experiment_run,
    experiment_status,
)


def register_experiment_parsers(subparsers):
    """Register the experiment subcommand and its sub-subcommands."""
    exp_parser = subparsers.add_parser(
        "experiment",
        help="Discover and run experiments",
        description="Run experiments from the experiments/ directory using the experiments framework.",
    )
    exp_subparsers = exp_parser.add_subparsers(dest="exp_command", help="Experiment commands")

    # Experiment list command
    exp_list_parser = exp_subparsers.add_parser("list", help="List all discovered experiments")
    exp_list_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory (default: auto-detect)",
    )
    exp_list_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output as JSON",
    )
    exp_list_parser.set_defaults(
        func=lambda args: experiment_list(
            experiments_dir=args.dir,
            json_output=args.json,
        )
    )

    # Experiment info command
    exp_info_parser = exp_subparsers.add_parser("info", help="Show detailed experiment information")
    exp_info_parser.add_argument("name", help="Experiment name")
    exp_info_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory",
    )
    exp_info_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output as JSON",
    )
    exp_info_parser.set_defaults(
        func=lambda args: experiment_info(
            name=args.name,
            experiments_dir=args.dir,
            json_output=args.json,
        )
    )

    # Experiment run command
    exp_run_parser = exp_subparsers.add_parser("run", help="Run an experiment")
    exp_run_parser.add_argument("name", help="Experiment name")
    exp_run_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory",
    )
    exp_run_parser.add_argument(
        "--config",
        "-c",
        help="Path to custom config YAML file",
    )
    exp_run_parser.add_argument(
        "--param",
        "-p",
        action="append",
        dest="params",
        help="Parameter override (key=value), can specify multiple",
    )
    exp_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without running",
    )
    exp_run_parser.set_defaults(
        func=lambda args: experiment_run(
            name=args.name,
            experiments_dir=args.dir,
            config_file=args.config,
            params=args.params,
            dry_run=args.dry_run,
        )
    )

    # Experiment status command
    exp_status_parser = exp_subparsers.add_parser(
        "status", help="Show experiment status and results"
    )
    exp_status_parser.add_argument("name", help="Experiment name")
    exp_status_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory",
    )
    exp_status_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        dest="show_all",
        help="Show all runs, not just latest",
    )
    exp_status_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output as JSON",
    )
    exp_status_parser.set_defaults(
        func=lambda args: experiment_status(
            name=args.name,
            experiments_dir=args.dir,
            show_all=args.show_all,
            json_output=args.json,
        )
    )
