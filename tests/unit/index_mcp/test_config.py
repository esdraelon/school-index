"""Unit tests: the server config builder (pure, env-mapping in)."""

import pytest

from school_index.index_mcp.config import ConfigError, config_from_env


def test_minimal_env_uses_defaults() -> None:
    config = config_from_env({"INDEX_MCP_TOKEN": "t0ken", "CORPUS_ROOT": "/data/corpus"})
    assert config.host == "127.0.0.1"
    assert config.port == 8766
    assert str(config.corpus_root) == "/data/corpus"
    assert str(config.models_dir) == "/data/corpus/models"
    assert config.model_key == "onnx-bge-small"


def test_explicit_values_win() -> None:
    config = config_from_env(
        {
            "INDEX_MCP_TOKEN": "t0ken",
            "CORPUS_ROOT": "/data/corpus",
            "INDEX_MCP_HOST": "0.0.0.0",
            "INDEX_MCP_PORT": "9001",
            "MODELS_DIR": "/models",
            "EMBED_MODEL": "other-model",
        }
    )
    assert config.host == "0.0.0.0"
    assert config.port == 9001
    assert str(config.models_dir) == "/models"
    assert config.model_key == "other-model"


def test_token_is_required() -> None:
    with pytest.raises(ConfigError, match="INDEX_MCP_TOKEN"):
        config_from_env({"CORPUS_ROOT": "/data/corpus"})


def test_corpus_root_is_required() -> None:
    with pytest.raises(ConfigError, match="CORPUS_ROOT"):
        config_from_env({"INDEX_MCP_TOKEN": "t0ken"})


def test_port_must_be_an_integer() -> None:
    with pytest.raises(ConfigError, match="INDEX_MCP_PORT"):
        config_from_env(
            {"INDEX_MCP_TOKEN": "t0ken", "CORPUS_ROOT": "/x", "INDEX_MCP_PORT": "abc"}
        )


def test_token_never_appears_in_repr() -> None:
    config = config_from_env({"INDEX_MCP_TOKEN": "t0ken-secret", "CORPUS_ROOT": "/x"})
    assert "t0ken-secret" not in repr(config)
