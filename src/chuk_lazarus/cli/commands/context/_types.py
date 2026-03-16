"""Type definitions for context CLI commands."""

from __future__ import annotations

from argparse import Namespace
from enum import Enum
from pathlib import Path

from pydantic import Field

from .._base import CommandConfig, CommandResult, OutputMixin
from .._constants import ContextDefaults


class ResidualMode(str, Enum):
    """Residual extraction mode for prefill."""

    INTERVAL = "interval"  # 8 samples per window (~40 KB/window)
    FULL = "full"  # every position (~5 MB/window for 512-token windows)
    NONE = "none"  # skip residual extraction (fastest, no compass)
    DARKSPACE = "darkspace"  # frame bank projection per position (single-pass, ~90 KB/window)


class PrefillPhase(str, Enum):
    """Phases of the prefill pipeline. No magic strings."""

    ALL = "all"
    WINDOWS = "windows"
    INTERVAL = "interval"
    COMPASS = "compass"
    DARKSPACE = "darkspace"
    PAGES = "pages"
    SURPRISE = "surprise"
    SPARSE = "sparse"

    def __str__(self) -> str:
        return self.value


class PrefillConfig(CommandConfig):
    """Configuration for the context prefill command (windowed library format)."""

    model: str = Field(..., description="Model ID or local path")
    input_file: Path = Field(..., description="Text file to prefill")
    checkpoint: Path = Field(..., description="Output library directory")
    window_size: int = Field(
        default=ContextDefaults.WINDOW_SIZE,
        ge=1,
        description="Tokens per window",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Truncate input to at most N tokens",
    )
    resume: bool = Field(
        default=True,
        description="Resume from an existing partial library",
    )
    name: str | None = Field(
        default=None,
        description="Human-readable library name (defaults to input filename stem)",
    )
    residual_mode: ResidualMode = Field(
        default=ResidualMode.INTERVAL,
        description="Residual extraction mode: interval, full, darkspace, none",
    )
    frame_bank: Path | None = Field(
        default=None,
        description="Path to frame_bank.npz (required for darkspace mode)",
    )
    store_pages: bool = Field(
        default=False,
        description="Store pre-RoPE K,V pages for instant injection at generate time",
    )
    store_kv_full: bool = Field(
        default=False,
        description="Save full KV cache per window for Mode 6 KV injection",
    )
    phases: set[PrefillPhase] = Field(
        default_factory=lambda: {PrefillPhase.ALL},
        description="Phases to run",
    )

    def _should_run(self, phase: PrefillPhase) -> bool:
        """Check if a phase should run (enabled explicitly or via ALL)."""
        return PrefillPhase.ALL in self.phases or phase in self.phases

    @property
    def run_windows(self) -> bool:
        return self._should_run(PrefillPhase.WINDOWS)

    @property
    def run_interval(self) -> bool:
        return self._should_run(PrefillPhase.INTERVAL)

    @property
    def run_compass(self) -> bool:
        return self._should_run(PrefillPhase.COMPASS)

    @property
    def run_darkspace(self) -> bool:
        return self._should_run(PrefillPhase.DARKSPACE)

    @property
    def run_pages(self) -> bool:
        return self._should_run(PrefillPhase.PAGES)

    @property
    def run_surprise(self) -> bool:
        return self._should_run(PrefillPhase.SURPRISE)

    @property
    def run_sparse(self) -> bool:
        return self._should_run(PrefillPhase.SPARSE)

    @classmethod
    def from_args(cls, args: Namespace) -> PrefillConfig:
        fb = getattr(args, "frame_bank", None)
        raw_phases = getattr(args, "phases", "all")
        phases = {PrefillPhase(p.strip()) for p in raw_phases.split(",")}
        return cls(
            model=args.model,
            input_file=Path(args.input),
            checkpoint=Path(args.checkpoint),
            window_size=getattr(args, "window_size", None) or ContextDefaults.WINDOW_SIZE,
            max_tokens=getattr(args, "max_tokens", None),
            resume=not getattr(args, "no_resume", False),
            name=getattr(args, "name", None),
            residual_mode=ResidualMode(getattr(args, "residual_mode", "interval")),
            frame_bank=Path(fb) if fb else None,
            store_pages=getattr(args, "store_pages", False),
            store_kv_full=getattr(args, "store_kv_full", False),
            phases=phases,
        )


class GenerateConfig(CommandConfig):
    """Configuration for the context generate command (library format)."""

    model: str = Field(..., description="Model ID or local path")
    checkpoint: Path | None = Field(
        default=None, description="Library directory to load from (None for plain generation)"
    )
    prompt: str | None = Field(default=None, description="Prompt text")
    prompt_file: Path | None = Field(default=None, description="File containing the prompt")
    max_tokens: int = Field(
        default=ContextDefaults.GENERATE_MAX_TOKENS,
        ge=1,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=ContextDefaults.TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )

    @classmethod
    def from_args(cls, args: Namespace) -> GenerateConfig:
        cp = getattr(args, "checkpoint", None)
        return cls(
            model=args.model,
            checkpoint=Path(cp) if cp else None,
            prompt=getattr(args, "prompt", None),
            prompt_file=Path(args.prompt_file) if getattr(args, "prompt_file", None) else None,
            max_tokens=getattr(args, "max_tokens", ContextDefaults.GENERATE_MAX_TOKENS),
            temperature=getattr(args, "temperature", ContextDefaults.TEMPERATURE),
        )

    @property
    def prompt_text(self) -> str | None:
        """Resolve prompt from inline text or file."""
        if self.prompt:
            return self.prompt
        if self.prompt_file:
            return self.prompt_file.read_text()
        return None


class PrefillResult(CommandResult, OutputMixin):
    """Result of a prefill operation."""

    checkpoint: str = Field(..., description="Path to saved library")
    tokens_prefilled: int = Field(..., description="Number of tokens prefilled")
    num_windows: int = Field(0, description="Number of windows archived")
    status: str = Field(..., description="complete or partial")
    elapsed_seconds: float = Field(..., description="Wall-clock time")

    def to_display(self) -> str:
        lines = [self.format_header("Prefill Complete")]
        lines.append(self.format_field("Library", self.checkpoint))
        lines.append(self.format_field("Tokens prefilled", self.tokens_prefilled))
        lines.append(self.format_field("Windows", self.num_windows))
        lines.append(self.format_field("Status", self.status))
        lines.append(self.format_field("Time", f"{self.elapsed_seconds:.1f}s"))
        return "\n".join(lines)


class GenerateResult(CommandResult, OutputMixin):
    """Result of a context generate operation."""

    response: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    context_tokens: int = Field(..., description="Tokens in the KV context")

    def to_display(self) -> str:
        lines = [self.format_header("Generated Response")]
        lines.append(self.response)
        lines.append("")
        lines.append(
            self.format_field(
                "Stats",
                f"{self.tokens_generated} tokens generated, "
                f"{self.context_tokens} tokens in context",
            )
        )
        return "\n".join(lines)


__all__ = [
    "GenerateConfig",
    "GenerateResult",
    "PrefillConfig",
    "PrefillPhase",
    "PrefillResult",
    "ResidualMode",
]
