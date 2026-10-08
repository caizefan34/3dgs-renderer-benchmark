import json, os, subprocess
HOT="/mnt/storage_pool/liaoyuanjun/fused_profile0/FUSED_HOTSPOT0"
req=["FUSED_HOTSPOT0_PREREG.json","FUSED_HOTSPOT0_BASELINE_FREEZE.json","FUSED_HOTSPOT0_TIMING_BOUNDARY.md",
     "FUSED_HOTSPOT0_CLEAN_TIMING.json","FUSED_HOTSPOT0_SOURCE_KERNEL_MAP.json","FUSED_HOTSPOT0_KERNEL_CENSUS.json",
     "FUSED_HOTSPOT0_COMPONENT_BREAKDOWN.json","FUSED_HOTSPOT0_WORKLOAD_STATS.json","FUSED_HOTSPOT0_KERNEL_CHARACTERIZATION.json",
     "FUSED_HOTSPOT0_SCALING.json","FUSED_HOTSPOT0_AMDAHL_ORACLE.json","FUSED_HOTSPOT0_REDUNDANCY_AUDIT.json",
     "FUSED_HOTSPOT0_PRIOR_ART_MAP.md","FUSED_HOTSPOT0_OPPORTUNITY_RANKING.json","FUSED_HOTSPOT0_VERDICT.json",
     "FUSED_HOTSPOT0_REPORT.md","MANIFEST.sha256"]
print("=== REQUIRED ARTIFACTS ===")
allok=True
for fn in req:
    p=os.path.join(HOT,fn); ok=os.path.exists(p)
    allok=allok and ok
    print(f"  {'OK ' if ok else 'MISSING'} {fn} ({os.path.getsize(p) if ok else 0} bytes)")
print("ALL_PRESENT", allok)
print("\n=== JSON PARSE CHECK ===")
for fn in req:
    if fn.endswith(".json"):
        p=os.path.join(HOT,fn)
        try: json.load(open(p)); print("  OK ", fn)
        except Exception as e: print("  FAIL", fn, e); allok=False
print("\n=== DIRS ===")
for d in ["raw","logs","profiler","scripts","source_map"]:
    print(f"  {d}: {len(os.listdir(os.path.join(HOT,d)))} entries")
print("\n=== MANIFEST VERIFY (sha256sum -c) ===")
r=subprocess.run(["bash","-c",f"cd {HOT} && sha256sum -c MANIFEST.sha256 2>&1 | tail -8"], capture_output=True, text=True)
print(r.stdout[-1500:])
print("MANIFEST_RC", r.returncode)
# count OK/FAIL
r2=subprocess.run(["bash","-c",f"cd {HOT} && sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK'"], capture_output=True, text=True)
r3=subprocess.run(["bash","-c",f"cd {HOT} && sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': FAILED'"], capture_output=True, text=True)
print("OK_COUNT", r2.stdout.strip(), "FAIL_COUNT", r3.stdout.strip())