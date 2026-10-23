"""Unit tests: benchmark config from an explicit environment mapping."""

from pathlib import Path

import pytest

from school_index.benchmark.config import ConfigError, config_from_env


def test_corpus_root_is_required() -> None:
    with pytest.raises(ConfigError, match="CORPUS_ROOT"):
        config_from_env({})


def test_defaults_follow_the_corpus_root() -> None:
    config = config_from_env({"CORPUS_ROOT": "/data/corpus"})
    assert config.corpus_root == Path("/data/corpus")
    assert config.benchmark_root == Path("/data/corpus")
    assert config.models_dir == Path("/data/corpus/models")
    assert config.queries_path == Path("benchmark/queries.json")


def test_explicit_overrides_win() -> None:
    config = config_from_env(
        {
            "CORPUS_ROOT": "/data/corpus",
            "BENCHMARK_ROOT": "/data/bench",
            "MODELS_DIR": "/data/models",
            "BENCHMARK_QUERIES": "/repo/benchmark/queries.json",
        }
    )
    assert config.benchmark_root == Path("/data/bench")
    assert config.models_dir == Path("/data/models")
    assert config.queries_path == Path("/repo/benchmark/queries.json")
