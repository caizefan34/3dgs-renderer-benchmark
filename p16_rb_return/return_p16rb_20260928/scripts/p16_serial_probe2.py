#!/usr/bin/env python3
"""P16-RB probe2: dump RAW tqdm lines around nerficg curve save points (read-only)."""
import re
import sys
from collections import Counter
from pathlib import Path

D = Path("/root/epic05_p16rb")
M = sys.argv[1] if len(sys.argv) > 1 else "FasterGS"
SC = sys.argv[2] if len(sys.argv) > 2 else "room"
log = (D / "curves" / M / (SC + "_materials") / "train.stdout.log").read_text(errors="replace")

RE = re.compile(r"(?:training|Training progress):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)\s*([a-z/]+)\]")
ms = list(RE.finditer(log))
print("total raw matches:", len(ms))
bars = [int(m.group(1)) for m in ms]
c = Counter(bars)
print("unique bars:", len(c), "bars_with_multiple_matches:", sum(1 for b, n in c.items() if n > 1))

for it in (5970, 11970):
    print("######## window around", it)
    for m in ms:
        b = int(m.group(1))
        if it - 130 <= b <= it + 90:
            pre = log[max(0, m.start() - 60):m.start()]
            print("bar", b, "el", m.group(3), "rem", m.group(4), "rate", m.group(5), m.group(6),
                  "| pre:", repr(pre[-60:]))

i = log.find("5970/30000")
print("######## raw slice around first 5970/30000 at", i)
print(repr(log[max(0, i - 400):i + 200]))
