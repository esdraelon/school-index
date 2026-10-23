"""Unit tests: embed config from the environment."""

from pathlib import Path

import pytest

from school_index.embed_index.config import ConfigError, config_from_env


def test_defaults_derive_from_corpus_root() -> None:
    config = config_from_env({"CORPUS_ROOT": "/srv/corpus"})
    assert config.corpus_root == Path("/srv/corpus")
    assert config.models_dir == Path("/srv/corpus/models")
    assert config.model_key == "onnx-bge-small"


def test_explicit_models_dir_and_model() -> None:
    config = config_from_env(
        {
            "CORPUS_ROOT": "/srv/corpus",
            "MODELS_DIR": "/srv/models",
            "EMBED_MODEL": "onnx-bge-small",
        }
    )
    assert config.models_dir == Path("/srv/models")
    assert config.model_key == "onnx-bge-small"


def test_missing_corpus_root_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="CORPUS_ROOT"):
        config_from_env({})
    with pytest.raises(ConfigError):
        config_from_env({"CORPUS_ROOT": "   "})
