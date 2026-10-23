"""CLI composition root for doc_extract.

Pattern: Composition Root — the only place that reads ``os.environ``,
builds the real collaborators (both store repositories over the one
corpus root, the extractor registry), and hands them to the
ExtractionRunner.

Usage: ``python -m school_index.doc_extract extract``

Exit codes: 0 = the run completed (per-item failures are data —
recorded in the store and tallied in the output, not a process
failure); 2 = configuration failure (nothing extracted).
"""

import argparse
import os
import sys

from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.config import ConfigError, config_from_env
from school_index.doc_extract.extractors import build_extractors
from school_index.doc_extract.runner import SOURCE_NAME, ExtractionRunner
from school_index.doc_extract.store import ExtractStore


def main(argv: list[str] | None = None) -> int:
    """Wire the real collaborators and run the requested command."""
    parser = argparse.ArgumentParser(
        prog="doc-extract", description="Extract harvested files to text."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("extract", help="run one extraction pass")
    args = parser.parse_args(argv)

    try:
        config = config_from_env(os.environ)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    runner = ExtractionRunner(
        corpus_store=CorpusStore.open(config.corpus_root),
        extract_store=ExtractStore.open(config.corpus_root),
        extractors=build_extractors(),
    )
    report = runner.run()
    print(
        f"extract source={SOURCE_NAME} total={report.total} "
        f"extracted={report.extracted} empty={report.empty} "
        f"failed={report.failed} not_extractable={report.not_extractable} "
        f"skipped={report.skipped} not_harvested={report.not_harvested}"
    )
    for error, count in sorted(report.failure_tally().items()):
        print(f"failure: {error} ×{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
