"""Self-maintaining pipeline chain: sync -> extract -> embed.

Pattern: deploy composition script (an artifact, like the systemd
units beside it — not package code). Runs the three stage CLIs in
sequence as subprocesses of the invoking interpreter (the systemd
unit invokes it with the repo venv's python), stops at the first
failed stage, and propagates its exit code, so the timer's unit
state is the run's honest verdict.

The sync stage needs the school MCP's bearer token. Following the
estate rule that tokens never travel through command lines, logs,
or files this script writes: the token is read HERE, in-process,
from the school adapter's deploy .env (the credential's home), put
into the sync child's environment only, and never printed. The
other stages take no credential.

Environment:
  CORPUS_ROOT       corpus root (default ~/school-index-data)
  SCHOOL_DEPLOY_ENV  path to the adapter's deploy .env
                    (default ~/school-info-adapter/deploy-run/.env)
"""

import os
import subprocess
import sys
from pathlib import Path


def _read_school_token(env_path: Path) -> str:
    """The SCHOOL_MCP_TOKEN value from a dotenv file (never printed)."""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "SCHOOL_MCP_TOKEN":
            token = value.strip().strip('"').strip("'")
            if token:
                return token
    raise RuntimeError(f"SCHOOL_MCP_TOKEN not found in {env_path}")


def main() -> int:
    """Run the three stages in order; return the first failure's code."""
    corpus_root = os.environ.get("CORPUS_ROOT") or str(
        Path.home() / "school-index-data"
    )
    env_path = Path(
        os.environ.get("SCHOOL_DEPLOY_ENV")
        or Path.home() / "school-info-adapter" / "deploy-run" / ".env"
    )
    try:
        school_token = _read_school_token(env_path)
    except (OSError, RuntimeError) as exc:
        print(f"pipeline config error: {exc}", file=sys.stderr)
        return 2

    base_env = dict(os.environ)
    base_env["CORPUS_ROOT"] = corpus_root
    base_env["PYTHONUNBUFFERED"] = "1"
    stages: list[tuple[str, list[str], dict[str, str]]] = [
        (
            "sync",
            [sys.executable, "-m", "school_index.corpus_sync", "sync", "--student", "sam"],
            {"SCHOOL_MCP_TOKEN": school_token},
        ),
        (
            "extract",
            [sys.executable, "-m", "school_index.doc_extract", "extract"],
            {},
        ),
        (
            "embed",
            [sys.executable, "-m", "school_index.embed_index", "build"],
            {},
        ),
    ]
    for name, command, extra_env in stages:
        print(f"== pipeline stage: {name} ==", flush=True)
        completed = subprocess.run(command, env={**base_env, **extra_env}, check=False)
        if completed.returncode != 0:
            print(
                f"pipeline failed at stage {name} (exit {completed.returncode})",
                file=sys.stderr,
            )
            return completed.returncode
    print("pipeline complete: sync -> extract -> embed all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
