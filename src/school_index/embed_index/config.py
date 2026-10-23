"""Embed configuration from the environment.

Pattern: pure config builder — ``config_from_env`` maps an explicit
environment mapping to a frozen ``EmbedConfig``; nothing here reads
``os.environ`` (the CLI shim owns that), so every branch is
unit-testable without monkeypatching the process environment.

Three knobs: where the corpus lives (``CORPUS_ROOT``, required —
same rule as the other stages), where model files live
(``MODELS_DIR``, defaulting to ``<corpus root>/models``: the model
is part of the corpus root's data on hal9k), and which registered
provider to run (``EMBED_MODEL``, default ``onnx-bge-small``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "onnx-bge-small"


class ConfigError(Exception):
    """The environment cannot produce a valid EmbedConfig."""


@dataclass(frozen=True, kw_only=True)
class EmbedConfig:
    """Everything the composition root needs, in one frozen value."""

    corpus_root: Path
    models_dir: Path
    model_key: str


def config_from_env(env: Mapping[str, str]) -> EmbedConfig:
    """Build an EmbedConfig from ``env``."""
    root = (env.get("CORPUS_ROOT") or "").strip()
    if not root:
        raise ConfigError("CORPUS_ROOT is required (where the corpus store lives)")
    corpus_root = Path(root).expanduser()
    models = (env.get("MODELS_DIR") or "").strip()
    models_dir = Path(models).expanduser() if models else corpus_root / "models"
    model_key = (env.get("EMBED_MODEL") or "").strip() or DEFAULT_MODEL
    return EmbedConfig(
        corpus_root=corpus_root, models_dir=models_dir, model_key=model_key
    )
