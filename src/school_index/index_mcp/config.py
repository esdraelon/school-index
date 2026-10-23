"""Server configuration from the environment.

Pattern: pure config builder — ``config_from_env`` maps an explicit
environment mapping to a frozen ``ServerConfig``; nothing here
reads ``os.environ`` (the CLI shim owns that), mirroring the other
stages' config modules.

The token is required and fails loudly when absent (a bearer-gated
service must never fall back to a plausible-looking default); it is
``repr=False`` so it cannot leak into logs via a config dump. The
corpus knobs match embed_index's: ``CORPUS_ROOT`` required,
``MODELS_DIR`` defaulting to ``<corpus root>/models``, and
``EMBED_MODEL`` selecting the provider registry key (the query
embedder must be the model the index was built with — the CLI
checks it against the store's census before serving).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from school_index.embed_index.config import DEFAULT_MODEL

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766


class ConfigError(Exception):
    """The environment cannot produce a valid ServerConfig."""


@dataclass(frozen=True, kw_only=True)
class ServerConfig:
    """Everything the composition root needs, in one frozen value."""

    host: str
    port: int
    # repr=False: the token must never leak into logs via a config dump.
    token: str = field(repr=False)
    corpus_root: Path
    models_dir: Path
    model_key: str


def config_from_env(env: Mapping[str, str]) -> ServerConfig:
    """Build a ServerConfig from ``env``."""
    token = (env.get("INDEX_MCP_TOKEN") or "").strip()
    if not token:
        raise ConfigError("INDEX_MCP_TOKEN is required (the index MCP's bearer token)")
    root = (env.get("CORPUS_ROOT") or "").strip()
    if not root:
        raise ConfigError("CORPUS_ROOT is required (where the corpus store lives)")
    corpus_root = Path(root).expanduser()
    models = (env.get("MODELS_DIR") or "").strip()
    models_dir = Path(models).expanduser() if models else corpus_root / "models"
    port_text = (env.get("INDEX_MCP_PORT") or "").strip()
    try:
        port = int(port_text) if port_text else DEFAULT_PORT
    except ValueError:
        raise ConfigError(
            f"INDEX_MCP_PORT must be an integer, got {port_text!r}"
        ) from None
    return ServerConfig(
        host=(env.get("INDEX_MCP_HOST") or "").strip() or DEFAULT_HOST,
        port=port,
        token=token,
        corpus_root=corpus_root,
        models_dir=models_dir,
        model_key=(env.get("EMBED_MODEL") or "").strip() or DEFAULT_MODEL,
    )
