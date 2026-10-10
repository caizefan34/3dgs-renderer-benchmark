#!/usr/bin/env python3
"""Run allowed local checks and emit reproducible delivery hashes; no MX claims."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def main():
    checks={}
    files=list(HERE.glob("*.py"))
    for path in files:ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    checks["python_ast"]={"status":"PASS","files":len(files)}
    commands={
        "cpu_regressions":[sys.executable,str(HERE/"R21_REGRESSION_TESTS.py")],
        "capture_static":[sys.executable,str(HERE/"R21_CAPTURE_FIRST_DIVERGENCE.py"),
            "--project",str(REPO/"HIGS_SOURCE_HANDOFF_ACTUAL/source"),"--backend","std_ll",
            "--scene","bicycle","--out",str(HERE/"_unused"),"--static-check"],
    }
    for name in ("R21_CAPTURE_FIRST_DIVERGENCE.py","R21_COMPARE_GAUSSIAN_STATE.py",
                 "R21_CAUSAL_REPLAY.py","R21_ANALYZE_THRESHOLD_CROSSING.py"):
        commands[name+"_help"]=[sys.executable,str(HERE/name),"--help"]
    for name,command in commands.items():
        result=subprocess.run(command,capture_output=True,text=True)
        checks[name]={"status":"PASS" if result.returncode==0 else "FAIL","exit_code":result.returncode}
        if result.returncode:
            print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)
    from R21_STATE import sha256,write_json
    r20=json.loads((REPO/"experiments/r20_true_absgrad/R20_SOURCE_MANIFEST.json").read_text())
    for rel,entry in r20["files"].items():
        assert sha256(REPO/"experiments/r20_true_absgrad/source"/rel)==entry["patched_sha256"],rel
    checks["r20_source_unchanged"]={"status":"PASS","files":len(r20["files"])}
    required=("R21_CODEX_SOURCE_AUDIT.md","R21_CAUSAL_TEST_DESIGN.md","R21_CAPTURE_FIRST_DIVERGENCE.py",
        "R21_COMPARE_GAUSSIAN_STATE.py","R21_CAUSAL_REPLAY.py","R21_ANALYZE_THRESHOLD_CROSSING.py",
        "R21_ROOT_CAUSE_REPORT.md","R21_REGRESSION_TESTS.py","R21_DSH_INTEGRATION.md")
    for name in required:assert (HERE/name).is_file(),name
    write_json(HERE/"R21_LOCAL_VALIDATION.json",{"status":"LOCAL_DIAGNOSTIC_CHECKS_PASS_MX_PENDING",
        "checks":checks,"root_cause":"NOT_ESTABLISHED","production_patch":"NOT_GENERATED_PENDING_CAUSAL_EVIDENCE",
        "MX_capture":"NOT_EXECUTED","MX_causal_replay":"NOT_EXECUTED","MX_1000_step_smoke":"NOT_EXECUTED",
        "MX_5k_quality":"NOT_EXECUTED","MX_performance":"NOT_EXECUTED"})
    payloads=sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in ("R21_SHA256SUMS.txt",))
    (HERE/"R21_SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in payloads),encoding="utf-8",newline="\n")
    print(json.dumps({"status":"LOCAL_DIAGNOSTIC_CHECKS_PASS_MX_PENDING","checks":checks},indent=2))


if __name__=="__main__":main()
