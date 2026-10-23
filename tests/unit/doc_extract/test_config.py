"""Unit tests: extraction config from the environment."""

from pathlib import Path

import pytest

from school_index.doc_extract.config import ConfigError, config_from_env


def test_missing_corpus_root_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="CORPUS_ROOT"):
        config_from_env({})


def test_blank_corpus_root_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="CORPUS_ROOT"):
        config_from_env({"CORPUS_ROOT": "   "})


def test_config_builds_from_env_and_strips_whitespace() -> None:
    config = config_from_env({"CORPUS_ROOT": " /tmp/corpus "})
    assert config.corpus_root == Path("/tmp/corpus")
