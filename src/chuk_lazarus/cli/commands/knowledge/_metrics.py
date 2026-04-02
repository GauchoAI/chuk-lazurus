"""Observability helpers for knowledge commands: JSONL logging + Prometheus metrics."""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


# ── JSONL Logger ─────────────────────────────────────────────────────

class JsonLogger:
    """Appends one JSON object per line to a file."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._f = open(self._path, "a", encoding="utf-8")
        self._t0 = time.monotonic()

    def event(self, kind: str, **data):
        record = {
            "ts": time.time(),
            "elapsed_s": round(time.monotonic() - self._t0, 3),
            "event": kind,
            **data,
        }
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ── Prometheus Metrics Server ────────────────────────────────────────

class MetricsState:
    """Thread-safe metrics state exposed via /metrics."""

    def __init__(self):
        self.windows_total = 0
        self.windows_processed = 0
        self.phase = "init"
        self.elapsed_s = 0.0
        self.eta_s = 0.0
        self.rate_windows_per_s = 0.0
        self.document_tokens = 0
        self.window_size = 512

    def render(self) -> str:
        lines = [
            "# HELP lazarus_build_windows_total Total number of windows to process.",
            "# TYPE lazarus_build_windows_total gauge",
            f"lazarus_build_windows_total {self.windows_total}",
            "# HELP lazarus_build_windows_processed Number of windows processed so far.",
            "# TYPE lazarus_build_windows_processed gauge",
            f"lazarus_build_windows_processed {self.windows_processed}",
            "# HELP lazarus_build_elapsed_seconds Wall-clock time since build started.",
            "# TYPE lazarus_build_elapsed_seconds gauge",
            f"lazarus_build_elapsed_seconds {self.elapsed_s:.3f}",
            "# HELP lazarus_build_eta_seconds Estimated seconds remaining.",
            "# TYPE lazarus_build_eta_seconds gauge",
            f"lazarus_build_eta_seconds {self.eta_s:.1f}",
            "# HELP lazarus_build_rate_windows_per_second Processing rate.",
            "# TYPE lazarus_build_rate_windows_per_second gauge",
            f"lazarus_build_rate_windows_per_second {self.rate_windows_per_s:.3f}",
            "# HELP lazarus_build_document_tokens Total tokens in document.",
            "# TYPE lazarus_build_document_tokens gauge",
            f"lazarus_build_document_tokens {self.document_tokens}",
            f'# HELP lazarus_build_phase Current phase (label).',
            f'# TYPE lazarus_build_phase gauge',
            f'lazarus_build_phase{{phase="{self.phase}"}} 1',
            "# HELP lazarus_build_progress_ratio Fraction of windows processed (0-1).",
            "# TYPE lazarus_build_progress_ratio gauge",
            f"lazarus_build_progress_ratio {self.windows_processed / max(self.windows_total, 1):.4f}",
        ]
        return "\n".join(lines) + "\n"


def _make_handler(state: MetricsState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                body = state.render().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok\n")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # suppress request logs

    return Handler


def start_metrics_server(state: MetricsState, port: int) -> HTTPServer:
    """Start a background HTTP server exposing Prometheus metrics on /metrics."""
    server = HTTPServer(("0.0.0.0", port), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"  Metrics server: http://localhost:{port}/metrics", file=sys.stderr)
    return server
