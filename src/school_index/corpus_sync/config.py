"""Sync configuration from the environment.

Pattern: pure config builder — ``config_from_env`` maps an explicit
environment mapping to a frozen ``SyncConfig``; nothing here reads
``os.environ`` (the CLI shim owns that), so every branch is
unit-testable without monkeypatching the process environment.

A missing token or corpus root raises ``ConfigError`` at startup:
the harvester carries a bearer credential and writes a corpus of
record, so a misconfigured deployment must fail loudly, never fall
back to a plausible-looking default. The MCP URL has a safe default
(the production gateway route) because it is not a secret.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

#: The school adapter's production gateway route.
DEFAULT_MCP_URL = "https://haliffer.ululuth.com/mcp/school/"


class ConfigError(Exception):
    """The environment cannot produce a valid SyncConfig."""


@dataclass(frozen=True, kw_only=True)
class SyncConfig:
    """Everything the composition root needs, in one frozen value."""

    mcp_url: str
    # repr=False: the token must never leak into logs via a config dump.
    mcp_token: str = field(repr=False)
    corpus_root: Path


def config_from_env(env: Mapping[str, str]) -> SyncConfig:
    """Build a SyncConfig from ``env`` (SCHOOL_MCP_URL / _TOKEN, CORPUS_ROOT)."""
    token = (env.get("SCHOOL_MCP_TOKEN") or "").strip()
    if not token:
        raise ConfigError(
            "SCHOOL_MCP_TOKEN is required (the school MCP's bearer token)"
        )
    root = (env.get("CORPUS_ROOT") or "").strip()
    if not root:
        raise ConfigError("CORPUS_ROOT is required (where the corpus store lives)")
    url = (env.get("SCHOOL_MCP_URL") or "").strip() or DEFAULT_MCP_URL
    return SyncConfig(
        mcp_url=url, mcp_token=token, corpus_root=Path(root).expanduser()
    )
