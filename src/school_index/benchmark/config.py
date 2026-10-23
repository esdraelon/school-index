"""Benchmark configuration from the environment.

Pattern: pure config builder — same shape as the other stages'
configs (``config_from_env`` over an explicit mapping; the CLI
shim owns ``os.environ``).

Knobs: ``CORPUS_ROOT`` (required — where the production corpus
lives; read-only for this package), ``BENCHMARK_ROOT`` (where
benchmark.db + throwaway arm indexes live; defaults to the corpus
root, which is the hal9k layout: ~/school-index-data holds both),
``MODELS_DIR`` (defaults to <corpus root>/models), and
``BENCHMARK_QUERIES`` (the labeled set; defaults to
benchmark/queries.json relative to the working directory — runs
happen from the repo root).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """The environment cannot produce a valid BenchmarkConfig."""


@dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
    """Everything the benchmark composition root needs."""

    corpus_root: Path
    benchmark_root: Path
    models_dir: Path
    queries_path: Path


def config_from_env(env: Mapping[str, str]) -> BenchmarkConfig:
    """Build a BenchmarkConfig from ``env``."""
    root = (env.get("CORPUS_ROOT") or "").strip()
    if not root:
        raise ConfigError("CORPUS_ROOT is required (where the corpus store lives)")
    corpus_root = Path(root).expanduser()
    bench = (env.get("BENCHMARK_ROOT") or "").strip()
    benchmark_root = Path(bench).expanduser() if bench else corpus_root
    models = (env.get("MODELS_DIR") or "").strip()
    models_dir = Path(models).expanduser() if models else corpus_root / "models"
    queries = (env.get("BENCHMARK_QUERIES") or "").strip()
    queries_path = (
        Path(queries).expanduser() if queries else Path("benchmark/queries.json")
    )
    return BenchmarkConfig(
        corpus_root=corpus_root,
        benchmark_root=benchmark_root,
        models_dir=models_dir,
        queries_path=queries_path,
    )
