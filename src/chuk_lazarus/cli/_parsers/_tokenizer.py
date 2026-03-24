"""Tokenizer command parsers."""

from ..commands.tokenizer import (
    analyze_coverage,
    analyze_diff,
    analyze_efficiency,
    analyze_entropy,
    analyze_fit_score,
    analyze_vocab_suggest,
    curriculum_length_buckets,
    curriculum_reasoning_density,
    instrument_histogram,
    instrument_oov,
    instrument_vocab_diff,
    instrument_waste,
    regression_run,
    research_analyze_embeddings,
    research_morph,
    research_soft_tokens,
    runtime_registry,
    tokenizer_benchmark,
    tokenizer_compare,
    tokenizer_decode,
    tokenizer_doctor,
    tokenizer_encode,
    tokenizer_fingerprint,
    tokenizer_vocab,
    training_pack,
    training_throughput,
)


def register_tokenizer_parsers(subparsers):
    """Register the tokenizer subcommand and all sub-subcommands."""
    tok_parser = subparsers.add_parser("tokenizer", help="Tokenizer utilities")
    tok_subparsers = tok_parser.add_subparsers(dest="tok_command", help="Tokenizer commands")

    # Encode command
    encode_parser = tok_subparsers.add_parser("encode", help="Encode text to tokens")
    encode_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    encode_parser.add_argument("--text", help="Text to encode")
    encode_parser.add_argument("--file", "-f", help="File to encode")
    encode_parser.add_argument("--special-tokens", action="store_true", help="Add special tokens")
    encode_parser.set_defaults(func=tokenizer_encode)

    # Decode command
    decode_parser = tok_subparsers.add_parser("decode", help="Decode token IDs to text")
    decode_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    decode_parser.add_argument("--ids", required=True, help="Token IDs (comma or space separated)")
    decode_parser.set_defaults(func=tokenizer_decode)

    # Vocab command
    vocab_parser = tok_subparsers.add_parser("vocab", help="Display vocabulary info")
    vocab_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    vocab_parser.add_argument("--show-all", action="store_true", help="Show full vocabulary")
    vocab_parser.add_argument("--search", "-s", help="Search for tokens containing string")
    vocab_parser.add_argument("--limit", type=int, default=50, help="Max results for search")
    vocab_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for display")
    vocab_parser.add_argument("--pause", action="store_true", help="Pause between chunks")
    vocab_parser.set_defaults(func=tokenizer_vocab)

    # Compare command
    compare_parser = tok_subparsers.add_parser("compare", help="Compare two tokenizers")
    compare_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    compare_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    compare_parser.add_argument("--text", required=True, help="Text to compare")
    compare_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full tokenization"
    )
    compare_parser.set_defaults(func=tokenizer_compare)

    # Doctor command
    doctor_parser = tok_subparsers.add_parser("doctor", help="Run tokenizer health check")
    doctor_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    doctor_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix detected issues (patches chat template if missing)",
    )
    doctor_parser.add_argument(
        "--format",
        choices=["chatml", "llama", "phi", "gemma", "zephyr", "vicuna", "alpaca"],
        help="Specify chat template format when using --fix (auto-detects if not set)",
    )
    doctor_parser.add_argument(
        "--output",
        "-o",
        help="Save patched tokenizer to directory (requires --fix)",
    )
    doctor_parser.set_defaults(func=tokenizer_doctor)

    # Fingerprint command
    fingerprint_parser = tok_subparsers.add_parser(
        "fingerprint", help="Generate or verify tokenizer fingerprint"
    )
    fingerprint_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    fingerprint_parser.add_argument("--save", "-s", help="Save fingerprint to JSON file")
    fingerprint_parser.add_argument("--verify", help="Verify against fingerprint (file or string)")
    fingerprint_parser.add_argument(
        "--strict", action="store_true", help="Strict verification (merges must match)"
    )
    fingerprint_parser.set_defaults(func=tokenizer_fingerprint)

    # Benchmark command
    bench_parser = tok_subparsers.add_parser("benchmark", help="Benchmark tokenizer throughput")
    bench_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    bench_parser.add_argument("--file", "-f", help="Corpus file (one text per line)")
    bench_parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)",
    )
    bench_parser.add_argument(
        "--avg-length",
        type=int,
        default=100,
        help="Avg words per sample for synthetic corpus",
    )
    bench_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for synthetic corpus"
    )
    bench_parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of parallel workers"
    )
    bench_parser.add_argument("--warmup", type=int, default=10, help="Warmup samples before timing")
    bench_parser.add_argument(
        "--special-tokens",
        action="store_true",
        help="Add special tokens during encoding",
    )
    bench_parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare HuggingFace vs Fast (MLX) backend",
    )
    bench_parser.set_defaults(func=tokenizer_benchmark)

    # === Analyze subcommands ===
    analyze_parser = tok_subparsers.add_parser("analyze", help="Token analysis tools")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command", help="Analysis type")

    # Coverage analysis
    cov_parser = analyze_subparsers.add_parser("coverage", help="Analyze token coverage")
    cov_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    cov_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    cov_parser.add_argument("--fragments", action="store_true", help="Include fragment analysis")
    cov_parser.set_defaults(func=analyze_coverage)

    # Entropy analysis
    ent_parser = analyze_subparsers.add_parser("entropy", help="Analyze token entropy")
    ent_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    ent_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    ent_parser.add_argument("--top-n", type=int, default=100, help="Top N tokens to show")
    ent_parser.set_defaults(func=analyze_entropy)

    # Fit score
    fit_parser = analyze_subparsers.add_parser("fit-score", help="Calculate tokenizer-dataset fit")
    fit_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    fit_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    fit_parser.set_defaults(func=analyze_fit_score)

    # Diff analysis
    diff_parser = analyze_subparsers.add_parser("diff", help="Compare tokenizers on corpus")
    diff_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    diff_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    diff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    diff_parser.set_defaults(func=analyze_diff)

    # Efficiency analysis
    eff_parser = analyze_subparsers.add_parser(
        "efficiency", help="Analyze token efficiency metrics"
    )
    eff_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    eff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    eff_parser.set_defaults(func=analyze_efficiency)

    # Vocab suggestion
    vocab_parser = analyze_subparsers.add_parser(
        "vocab-suggest", help="Suggest vocabulary additions"
    )
    vocab_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    vocab_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    vocab_parser.add_argument("--min-freq", type=int, default=5, help="Minimum frequency")
    vocab_parser.add_argument("--min-frag", type=int, default=3, help="Minimum fragmentation")
    vocab_parser.add_argument("--limit", type=int, default=50, help="Maximum candidates")
    vocab_parser.add_argument("--show", type=int, default=20, help="Number to display")
    vocab_parser.set_defaults(func=analyze_vocab_suggest)

    # === Curriculum subcommands ===
    curr_parser = tok_subparsers.add_parser("curriculum", help="Curriculum learning tools")
    curr_subparsers = curr_parser.add_subparsers(dest="curriculum_command", help="Curriculum type")

    # Length buckets
    len_parser = curr_subparsers.add_parser("length-buckets", help="Create length-based curriculum")
    len_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    len_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    len_parser.add_argument("--num-buckets", type=int, default=5, help="Number of buckets")
    len_parser.add_argument("--schedule", action="store_true", help="Show curriculum schedule")
    len_parser.set_defaults(func=curriculum_length_buckets)

    # Reasoning density
    reason_parser = curr_subparsers.add_parser(
        "reasoning-density", help="Score by reasoning density"
    )
    reason_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    reason_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    reason_parser.add_argument(
        "--descending", action="store_true", help="Sort descending (hardest first)"
    )
    reason_parser.set_defaults(func=curriculum_reasoning_density)

    # === Training subcommands ===
    train_tok_parser = tok_subparsers.add_parser("training", help="Training utilities")
    train_tok_subparsers = train_tok_parser.add_subparsers(
        dest="training_command", help="Training tool"
    )

    # Throughput profiling
    thru_parser = train_tok_subparsers.add_parser(
        "throughput", help="Profile tokenization throughput"
    )
    thru_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    thru_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    thru_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    thru_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    thru_parser.set_defaults(func=training_throughput)

    # Sequence packing
    pack_parser = train_tok_subparsers.add_parser("pack", help="Pack sequences for training")
    pack_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    pack_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    pack_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    pack_parser.add_argument("--output", "-o", help="Output file (JSONL)")
    pack_parser.set_defaults(func=training_pack)

    # === Regression subcommands ===
    reg_parser = tok_subparsers.add_parser("regression", help="Token regression testing")
    reg_subparsers = reg_parser.add_subparsers(dest="regression_command", help="Regression tool")

    # Run tests
    run_parser = reg_subparsers.add_parser("run", help="Run regression tests")
    run_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    run_parser.add_argument("--tests", required=True, help="Test suite YAML file")
    run_parser.set_defaults(func=regression_run)

    # === Runtime subcommands ===
    runtime_parser = tok_subparsers.add_parser("runtime", help="Runtime token utilities")
    runtime_subparsers = runtime_parser.add_subparsers(dest="runtime_command", help="Runtime tool")

    # Registry
    registry_parser = runtime_subparsers.add_parser(
        "registry", help="Display special token registry"
    )
    registry_parser.add_argument("--tokenizer", "-t", help="Tokenizer name or path")
    registry_parser.add_argument("--standard", action="store_true", help="Show standard registry")
    registry_parser.set_defaults(func=runtime_registry)

    # === Research subcommands ===
    research_parser = tok_subparsers.add_parser("research", help="Research playground tools")
    research_subparsers = research_parser.add_subparsers(
        dest="research_command", help="Research tool"
    )

    # Soft tokens
    soft_parser = research_subparsers.add_parser(
        "soft-tokens", help="Create soft token bank for prompt tuning"
    )
    soft_parser.add_argument(
        "--num-tokens", "-n", type=int, default=10, help="Number of soft tokens"
    )
    soft_parser.add_argument(
        "--embedding-dim", "-d", type=int, default=768, help="Embedding dimension"
    )
    soft_parser.add_argument("--prefix", "-p", default="prompt", help="Token name prefix")
    soft_parser.add_argument(
        "--init-method",
        choices=["random_normal", "random_uniform", "zeros"],
        default="random_normal",
        help="Initialization method",
    )
    soft_parser.add_argument("--init-std", type=float, default=0.02, help="Std dev for random init")
    soft_parser.add_argument("--output", "-o", help="Save bank to JSON file")
    soft_parser.set_defaults(func=research_soft_tokens)

    # Analyze embeddings
    emb_parser = research_subparsers.add_parser(
        "analyze-embeddings", help="Analyze embedding space"
    )
    emb_parser.add_argument("--file", "-f", required=True, help="JSON file with embeddings")
    emb_parser.add_argument("--num-clusters", "-k", type=int, default=10, help="Number of clusters")
    emb_parser.add_argument("--cluster", action="store_true", help="Show cluster analysis")
    emb_parser.add_argument("--project", action="store_true", help="Show 2D projection stats")
    emb_parser.set_defaults(func=research_analyze_embeddings)

    # Morph
    morph_parser = research_subparsers.add_parser("morph", help="Morph between token embeddings")
    morph_parser.add_argument("--file", "-f", required=True, help="JSON file with embeddings")
    morph_parser.add_argument("--source", "-s", type=int, required=True, help="Source token index")
    morph_parser.add_argument("--target", "-t", type=int, required=True, help="Target token index")
    morph_parser.add_argument(
        "--method",
        "-m",
        choices=["linear", "spherical", "bezier", "cubic"],
        default="linear",
        help="Morph method",
    )
    morph_parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    morph_parser.add_argument("--normalize", action="store_true", help="Normalize output")
    morph_parser.add_argument("--output", "-o", help="Save trajectory to JSON")
    morph_parser.set_defaults(func=research_morph)

    # === Instrumentation subcommands ===
    instrument_parser = tok_subparsers.add_parser("instrument", help="Tokenizer instrumentation")
    instrument_subparsers = instrument_parser.add_subparsers(
        dest="instrument_command", help="Instrumentation tool"
    )

    # Histogram
    hist_parser = instrument_subparsers.add_parser(
        "histogram", help="Display token length histogram"
    )
    hist_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    hist_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    hist_parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    hist_parser.add_argument("--width", type=int, default=50, help="Chart width")
    hist_parser.add_argument("--quick", action="store_true", help="Quick stats only")
    hist_parser.set_defaults(func=instrument_histogram)

    # OOV analysis
    oov_parser = instrument_subparsers.add_parser("oov", help="Analyze OOV and rare tokens")
    oov_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    oov_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    oov_parser.add_argument("--vocab-size", type=int, default=50000, help="Expected vocab size")
    oov_parser.add_argument("--show-rare", action="store_true", help="Show rare tokens")
    oov_parser.add_argument("--max-freq", type=int, default=5, help="Max frequency for rare")
    oov_parser.add_argument("--top-k", type=int, default=20, help="Number of rare tokens to show")
    oov_parser.set_defaults(func=instrument_oov)

    # Waste analysis
    waste_parser = instrument_subparsers.add_parser(
        "waste", help="Analyze padding and truncation waste"
    )
    waste_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    waste_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    waste_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    waste_parser.set_defaults(func=instrument_waste)

    # Vocab diff
    vocab_diff_parser = instrument_subparsers.add_parser(
        "vocab-diff", help="Compare two tokenizers on a corpus"
    )
    vocab_diff_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    vocab_diff_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    vocab_diff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    vocab_diff_parser.add_argument("--examples", type=int, default=5, help="Max examples to show")
    vocab_diff_parser.add_argument(
        "--cost", action="store_true", help="Show retokenization cost estimate"
    )
    vocab_diff_parser.set_defaults(func=instrument_vocab_diff)
