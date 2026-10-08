#!/usr/bin/env python3
import json, sys

p=sys.argv[1]
d=json.load(open(p))

if not d.get("source_identity_pass", False):
    verdict="LAVOS0_SOURCE_IDENTITY_FAIL"
elif not d.get("event_semantics_resolved", False):
    verdict="LAVOS0_EVENT_SEMANTICS_UNRESOLVED"
elif not d.get("structural_headroom_pass", False):
    verdict="LAVOS0_NO_STRUCTURAL_HEADROOM"
elif not d.get("topology_exactness_pass", False):
    verdict="LAVOS0_TOPOLOGY_EXACTNESS_FAIL"
elif not d.get("performance_headroom_pass", False):
    verdict="LAVOS0_PERFORMANCE_HEADROOM_FAIL"
else:
    strong = bool(d.get("strong_headroom_pass", False))
    verdict="LAVOS0_STRONG_HEADROOM_PROMOTE" if strong else "LAVOS0_HEADROOM_PROMOTE"

out={
    "verdict": verdict,
    "source_identity_pass": d.get("source_identity_pass"),
    "event_semantics_resolved": d.get("event_semantics_resolved"),
    "structural_headroom_pass": d.get("structural_headroom_pass"),
    "topology_exactness_pass": d.get("topology_exactness_pass"),
    "performance_headroom_pass": d.get("performance_headroom_pass"),
    "strong_headroom_pass": d.get("strong_headroom_pass", False),
    "per_scene": d.get("per_scene", {}),
    "global": d.get("global", {})
}
vp=p.replace("SUMMARY","VERDICT")
open(vp,"w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
