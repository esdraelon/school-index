"""CLI composition root for embed_index.

Pattern: Composition Root — the only place that reads ``os.environ``,
builds the real collaborators (the three stage repositories over the
one corpus root, the provider registry), and hands them to the
EmbedRunner. The provider is selected from the registry by config
key (``EMBED_MODEL``); an unknown key is a configuration error, not
a fallback.

Usage:
  ``python -m school_index.embed_index build [--rebuild]``
  ``python -m school_index.embed_index fetch-model``
  ``python -m school_index.embed_index search "query" [--limit N]``

Exit codes: 0 = the command completed (per-item embed failures are
data — recorded in the report, not a process failure); 2 =
configuration or index-state failure (bad env, unknown model,
model not downloaded, mixed-model index without --rebuild).
"""

import argparse
import os
import statistics
import sys
import time

from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.store import ExtractStore
from school_index.embed_index.config import ConfigError, config_from_env
from school_index.embed_index.providers import (
    ModelNotDownloaded,
    build_providers,
    download_model,
)
from school_index.embed_index.runner import SOURCE_NAME, EmbedRunner, ModelMismatchError
from school_index.embed_index.store import EmbedStore


def main(argv: list[str] | None = None) -> int:
    """Wire the real collaborators and run the requested command."""
    parser = argparse.ArgumentParser(
        prog="embed-index", description="Chunk, embed, and search the corpus."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="run one embed pass")
    build.add_argument(
        "--rebuild",
        action="store_true",
        help="clear the index first (required when changing models)",
    )
    subcommands.add_parser("fetch-model", help="download the embedding model")
    search = subcommands.add_parser("search", help="similarity-search the index")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        config = config_from_env(os.environ)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "fetch-model":
        target = download_model(config.models_dir)
        print(f"model ready at {target}")
        return 0

    providers = build_providers(models_dir=config.models_dir)
    provider = providers.get(config.model_key)
    if provider is None:
        print(
            f"config error: unknown EMBED_MODEL {config.model_key!r} "
            f"(registered: {sorted(providers)})",
            file=sys.stderr,
        )
        return 2

    embed_store = EmbedStore.open(config.corpus_root, dim=provider.dim)

    if args.command == "search":
        try:
            query_vector = provider.embed_query(args.query)
        except ModelNotDownloaded as exc:
            print(f"model error: {exc}", file=sys.stderr)
            return 2
        hits = embed_store.search(
            query_vector=query_vector, model_id=provider.model_id, limit=args.limit
        )
        titles = {
            item.id: item.title
            for item in CorpusStore.open(config.corpus_root).all_items()
        }
        for hit in hits:
            title = titles.get(hit.chunk.item_id, hit.chunk.item_id)
            snippet = " ".join(hit.chunk.text.split())[:140]
            print(
                f"{hit.score:.4f}  [{title}] ({hit.chunk.heading or 'no page'}) "
                f"{snippet}"
            )
        if not hits:
            print("no hits (index empty?)")
        return 0

    runner = EmbedRunner(
        corpus_store=CorpusStore.open(config.corpus_root),
        extract_store=ExtractStore.open(config.corpus_root),
        embed_store=embed_store,
        provider=provider,
    )
    started = time.perf_counter()
    try:
        report = runner.run(rebuild=args.rebuild)
    except (ModelMismatchError, ModelNotDownloaded) as exc:
        print(f"embed error: {exc}", file=sys.stderr)
        return 2
    wall = time.perf_counter() - started
    counts = embed_store.chunk_count_by_item()
    per_item = sorted(counts.values())
    census = (
        f"chunks_per_item min={per_item[0]} median={int(statistics.median(per_item))} "
        f"max={per_item[-1]}"
        if per_item
        else "chunks_per_item none"
    )
    print(
        f"embed source={SOURCE_NAME} backend={embed_store.backend_name} "
        f"model={provider.model_id} total={report.total} "
        f"embedded={report.embedded} skipped={report.skipped} "
        f"purged={report.purged} failed={report.failed} "
        f"not_embeddable={report.not_embeddable} "
        f"chunks_written={report.chunks_written} chunks_total={report.chunks_total} "
        f"wall={wall:.2f}s"
    )
    print(f"census: {census}")
    for error, count in sorted(report.failure_tally().items()):
        print(f"failure: {error} ×{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
