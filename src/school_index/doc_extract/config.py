"""Extraction configuration from the environment.

Pattern: pure config builder — ``config_from_env`` maps an explicit
environment mapping to a frozen ``ExtractConfig``; nothing here
reads ``os.environ`` (the CLI shim owns that), so every branch is
unit-testable without monkeypatching the process environment.

Extraction needs exactly one thing: where the corpus lives. There
is no credential and no network here, but a missing ``CORPUS_ROOT``
must still fail loudly — extracting into the wrong directory would
build a second, plausible-looking store next to the corpus of
record.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """The environment cannot produce a valid ExtractConfig."""


@dataclass(frozen=True, kw_only=True)
class ExtractConfig:
    """Everything the composition root needs, in one frozen value."""

    corpus_root: Path


def config_from_env(env: Mapping[str, str]) -> ExtractConfig:
    """Build an ExtractConfig from ``env`` (CORPUS_ROOT)."""
    root = (env.get("CORPUS_ROOT") or "").strip()
    if not root:
        raise ConfigError("CORPUS_ROOT is required (where the corpus store lives)")
    return ExtractConfig(corpus_root=Path(root).expanduser())
