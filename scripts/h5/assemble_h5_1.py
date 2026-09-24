#!/usr/bin/env python3
"""Assemble final H5-1 deliverables under artifacts/higs-h5-1/ and reports/.
Merges per-scene outputs from scripts/h5/tmp_out/{room,bicycle,garden}."""
import json, csv, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TMP = BASE / "tmp_out"
SCENES = ["room", "bicycle", "garden"]
REPO = BASE.parents[1] if BASE.parent.name == "scripts" else BASE.parent
ART = REPO / "artifacts" / "higs-h5-1"
REP = REPO / "reports" / "higs"
ART.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

def rd(scene, name):
    return json.loads((TMP / scene / name).read_text())

# ---- segment_distribution.csv : concatenate rows (per scene) ----
header = None; rows = []
for sc in SCENES:
    p = TMP / sc / "segment_distribution.csv"
    lines = p.read_text().strip().splitlines()
    if header is None:
        header = lines[0]
    rows.extend(lines[1:])
(ART / "segment_distribution.csv").write_text(header + "\n" + "\n".join(rows) + "\n")

# ---- aggregate json artifacts keyed by scene ----
def agg(name, wrap=True):
    out = {} if wrap else []
    for sc in SCENES:
        out[sc] = rd(sc, name)
    (ART / name).write_text(json.dumps(out, indent=2))

agg("critical_path.json")
agg("state_cost.json")
agg("fp_exactness.json")
agg("lastid_composition.json")

# mask_mapping, resource_usage, analysis: make a top-level per-scene map
def agg_flat(name):
    (ART / name).write_text(json.dumps({sc: rd(sc, name) for sc in SCENES}, indent=2))
agg_flat("mask_mapping.json")
agg_flat("resource_usage.json")
agg_flat("analysis.json")

# ---- microkernel_timing.csv : concatenate all rows, tag scene ----
mh = None; mrows = []
for sc in SCENES:
    p = TMP / sc / "microkernel_timing.csv"
    lines = p.read_text().strip().splitlines()
    if mh is None:
        mh = "scene," + lines[0]
    for ln in lines[1:]:
        mrows.append(sc + "," + ln)
(ART / "microkernel_timing.csv").write_text(mh + "\n" + "\n".join(mrows) + "\n")

# ---- provenance : aggregate ----
prov = {"scripts": ["h5_capture.py", "h5_run.py", "h5_micro.cu", "assemble_h5_1.py"],
        "device": "mx A100-PCIE-40GB", "env": "higs-13scene (torch 2.9.1+cu128)",
        "capture": "validated Python oracle == native forward order (depth-then-id)",
        "scenes": {sc: rd(sc, "provenance.json") for sc in SCENES},
        "gate_summary": "WEAK on all scenes (see report)"}
(ART / "provenance.json").write_text(json.dumps(prov, indent=2))

print("wrote", len(list(ART.iterdir())), "artifacts to", ART)