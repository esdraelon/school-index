"""CLI composition root for corpus_sync.

Pattern: Composition Root — the only place that reads ``os.environ``,
builds the real adapters (MCP source, HTTP downloader, strategy
chain, file-backed store), and hands them to the Harvester.

Usage: ``python -m school_index.corpus_sync sync --student sam``

Exit codes: 0 = the run completed (per-item failures are data —
recorded in the store and tallied in the output, not a process
failure); 2 = configuration or contract failure (nothing synced).
"""

import argparse
import os
import sys

from school_index.corpus_sync.client import McpResourceSource, PayloadError
from school_index.corpus_sync.config import ConfigError, config_from_env
from school_index.corpus_sync.fetch import HttpDownloader
from school_index.corpus_sync.harvester import SOURCE_NAME, Harvester
from school_index.corpus_sync.store import CorpusStore
from school_index.corpus_sync.strategies import build_strategies


def main(argv: list[str] | None = None) -> int:
    """Wire the real collaborators and run the requested command."""
    parser = argparse.ArgumentParser(
        prog="corpus-sync", description="Harvest school resources into the corpus."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    sync_parser = subcommands.add_parser("sync", help="run one harvest")
    sync_parser.add_argument("--student", default="sam", help="student id (default: sam)")
    args = parser.parse_args(argv)

    try:
        config = config_from_env(os.environ)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    harvester = Harvester(
        source=McpResourceSource(url=config.mcp_url, token=config.mcp_token),
        downloader=HttpDownloader(),
        strategies=build_strategies(),
        store=CorpusStore.open(config.corpus_root),
    )
    try:
        report = harvester.run(student=args.student)
    except PayloadError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2

    print(
        f"sync student={report.student} source={SOURCE_NAME} total={report.total}"
    )
    print(
        "coverage exhaustive={} probe_distinct={} partition_calls={}".format(
            report.exhaustive,
            report.probe_distinct,
            report.partition_calls,
        )
    )
    print(
        "downloaded={} link_only={} unchanged={} failed={} tombstoned={}".format(
            report.downloaded,
            report.link_only,
            report.unchanged,
            report.failed,
            report.tombstoned,
        )
    )
    for error, count in sorted(report.failure_tally().items()):
        print(f"failure: {error} ×{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
