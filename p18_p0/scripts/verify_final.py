import json, pathlib
art = pathlib.Path(r"C:\Users\36570\3dgs-renderer-benchmark\p18_p0\artifacts")
for f in ["P18_FINAL_CLOSEOUT.md", "P18_EVIDENCE_CHAIN.json", "P18_VERDICT.json", "MANIFEST_P18_FINAL.json"]:
    p = art / f
    if f.endswith(".json"):
        d = json.load(open(p, encoding="utf-8"))
        print("OK  %-26s  artifact=%s  verdict=%s" % (f, d.get("artifact"), d.get("verdict", "n/a")))
    else:
        print("OK  %-26s  (%d bytes, markdown)" % (f, p.stat().st_size))

m = json.load(open(art / "MANIFEST_P18_FINAL.json", encoding="utf-8"))
r1 = {e["path"].split("/")[-1]: e["sha256"] for e in m["valid_r1_evidence"]}
claimed = m["r1_sass_audit_artifact_hashes"]
print("\nR1 hash reconciliation (manifest vs R1-verdict claims):")
for k, v in claimed.items():
    match = "MATCH" if r1.get(k) == v else "MISMATCH"
    print("  %-9s  %s  %s" % (match, k, v[:16]))
print("\nAll superseded marked INVALID_SUPERSEDED_BY_R1:",
      all(e.get("status") == "INVALID_SUPERSEDED_BY_R1" for e in m["superseded_invalid_pre_r1"]))
print("Final S_eval recorded:", json.load(open(art / "P18_VERDICT.json", encoding="utf-8"))["S_eval_final"])
