"""LOC cap check: fail if any .py file exceeds 500 lines.

Usage: python scripts/check_loc.py [root ...]   (default root: src)
Tooling script — not part of the shipped package, not coverage-measured.
Mirrors school-info-adapter's checker (same convention, same cap).
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES = 500


def main(argv: list[str]) -> int:
    roots = [Path(arg) for arg in argv[1:]] or [Path("src")]
    offenders: list[tuple[Path, int]] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            line_count = sum(1 for _ in path.open(encoding="utf-8"))
            if line_count > MAX_LINES:
                offenders.append((path, line_count))
    if offenders:
        for path, line_count in offenders:
            print(f"LOC CAP EXCEEDED: {path} has {line_count} lines (max {MAX_LINES})")
        return 1
    print(f"LOC cap OK: no .py file under {', '.join(str(r) for r in roots)} exceeds {MAX_LINES} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
