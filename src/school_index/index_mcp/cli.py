"""CLI composition root for index_mcp: serve the search MCP.

Pattern: Composition Root — the only place that reads ``os.environ``,
builds the real collaborators (provider registry, read store,
SearchService), checks them against each other, and hands the ASGI
app to uvicorn.

Startup checks (fail loudly, exit 2 — a misconfigured search server
must not come up plausible-looking):
- config must parse (token + corpus root present);
- the provider key must be registered;
- the corpus DB must exist and hold a servable index;
- the index's model census must name the configured provider's
  model — queries embedded with a different model than the stored
  vectors would return confident garbage. An empty index (nothing
  embedded yet) warns and serves empty results instead.

Usage: ``python -m school_index.index_mcp``

Proxy headers are trusted unconditionally: the only production
shape is behind the haliffer.ululuth.com gateway nginx (localhost
bind, token-gated), and without X-Forwarded-* trust the app builds
redirect targets from the internal scheme/host — the gateway
deploy's 307-to-http bug. Direct local requests carry no
X-Forwarded-* headers, so local serving is unaffected.
"""

import os
import sys

import uvicorn

from school_index.corpus_sync.lookup import Found
from school_index.embed_index.providers import build_providers
from school_index.index_mcp.config import ConfigError, config_from_env
from school_index.index_mcp.queries import SearchService
from school_index.index_mcp.server import build_server
from school_index.index_mcp.store import IndexNotReady, IndexReadStore


def main(argv: list[str] | None = None) -> int:
    """Wire the real collaborators and serve until interrupted."""
    del argv  # serving takes no arguments; config is the environment
    try:
        config = config_from_env(os.environ)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    providers = build_providers(models_dir=config.models_dir)
    provider = providers.get(config.model_key)
    if provider is None:
        print(
            f"config error: unknown EMBED_MODEL {config.model_key!r} "
            f"(registered: {sorted(providers)})",
            file=sys.stderr,
        )
        return 2

    try:
        store = IndexReadStore.open(config.corpus_root, dim=provider.dim)
        census = store.index_model_id()
    except IndexNotReady as exc:
        print(f"index error: {exc}", file=sys.stderr)
        return 2
    if isinstance(census, Found) and census.value != provider.model_id:
        print(
            f"index error: index holds model {census.value!r} but EMBED_MODEL "
            f"selects {provider.model_id!r} — refusing to serve",
            file=sys.stderr,
        )
        return 2
    if not isinstance(census, Found):
        print(
            "warning: index holds no embedded items; searches will be empty",
            file=sys.stderr,
        )

    print(
        f"index-mcp backend={store.backend_name} model={provider.model_id} "
        f"corpus={config.corpus_root} listen={config.host}:{config.port}"
    )
    service = SearchService(store=store, provider=provider)
    uvicorn.run(
        build_server(service, expected_token=config.token),
        host=config.host,
        port=config.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
