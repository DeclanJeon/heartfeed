#!/usr/bin/env python3
"""Count OKF claims and list conflict pairs."""

from __future__ import annotations

import re
from pathlib import Path

CLAIMS = Path(__file__).resolve().parents[1] / "data" / "okf" / "claims"


def main() -> None:
    files = sorted(CLAIMS.glob("claim-*.md"))
    active = 0
    conflicts = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if re.search(r"status:\s*active", text) or "status:" not in text.split("---")[1] if "---" in text else True:
            active += 1
        m = re.search(r"conflicts_with:\s*\[([^\]]*)\]", text)
        if m and m.group(1).strip():
            conflicts.append((f.name, m.group(1).strip()))
    print(f"claims_total={len(files)} active_est={active} conflict_pairs={len(conflicts)}")
    for c in conflicts[:20]:
        print(" conflict", c)


if __name__ == "__main__":
    main()
