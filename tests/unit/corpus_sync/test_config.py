"""Unit tests: config_from_env + the CLI's config-failure path."""

from pathlib import Path

import pytest

from school_index.corpus_sync.cli import main
from school_index.corpus_sync.config import (
    DEFAULT_MCP_URL,
    ConfigError,
    config_from_env,
)

FULL_ENV = {"SCHOOL_MCP_TOKEN": "tok", "CORPUS_ROOT": "/tmp/corpus"}


def test_config_defaults_the_gateway_url() -> None:
    config = config_from_env(FULL_ENV)
    assert config.mcp_url == DEFAULT_MCP_URL
    assert config.corpus_root == Path("/tmp/corpus")


def test_config_honors_explicit_url_and_expands_root() -> None:
    env = {**FULL_ENV, "SCHOOL_MCP_URL": "http://localhost:9999/", "CORPUS_ROOT": "~/c"}
    config = config_from_env(env)
    assert config.mcp_url == "http://localhost:9999/"
    assert config.corpus_root == Path.home() / "c"


def test_config_requires_token_and_root() -> None:
    with pytest.raises(ConfigError, match="SCHOOL_MCP_TOKEN"):
        config_from_env({"CORPUS_ROOT": "/tmp/corpus"})
    with pytest.raises(ConfigError, match="CORPUS_ROOT"):
        config_from_env({"SCHOOL_MCP_TOKEN": "tok"})
    with pytest.raises(ConfigError):
        config_from_env({"SCHOOL_MCP_TOKEN": "  ", "CORPUS_ROOT": "/tmp/corpus"})


def test_token_never_appears_in_repr() -> None:
    config = config_from_env(FULL_ENV)
    assert "tok" not in repr(config)


def test_cli_returns_2_on_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCHOOL_MCP_TOKEN", raising=False)
    monkeypatch.delenv("CORPUS_ROOT", raising=False)
    assert main(["sync", "--student", "sam"]) == 2
