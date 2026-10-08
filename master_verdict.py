import json, os
# Gather lane verdicts from authoritative sources
mx_oracle = "/mnt/storage_pool/liaoyuanjun/eb0_exact_backward_oracle_mx"
def rd(p):
    try: return json.load(open(p,encoding="utf-8"))
    except Exception as e: return {"_error":str(e)}
hgr0 = rd(os.path.join(mx_oracle,"HGR0_VERDICT.json"))
abc0 = rd(os.path.join(mx_oracle,"ABC0_VERDICT.json"))
hbr0 = rd(os.path.join(mx_oracle,"HBR0_VERDICT.json"))
eb0  = rd(os.path.join(mx_oracle,"EB0_COMPOSITION_ORACLE.json"))
sg    = rd("_sgor0/SGOR0_STATUS.json")

lanes = {
  "PSR0": {
    "verdict":"DROP_NO_E2E_HEADROOM",
    "status":"FINAL",
    "backward_speedup":"projection-backward recompute <8% production iteration AND <15% backward -> DROP per preregistered gate",
    "exactness":"n/a (dropped before exactness gate)",
    "projected_production_gain":"none (dropped)",
    "memory_overhead":"n/a",
    "engineering_complexity":"n/a",
    "scene_robustness":"n/a",
    "novelty_confidence":"n/a",
    "evidence":"goal-objective PSR0 gate; prior-turn resolution"
  },
  "SGOR0": {
    "verdict":"FAIL_BY_DILUTION (no-e2e-headroom)",
    "status":"FINAL",
    "backward_speedup":"scatter fraction |f|<=0.05 of SerialBatchBwd inner loop; Amdahl ceiling 1/(1-f)<=1.053x even with infinite scatter speedup; < 1.10x/scene and < 1.25x/pooled gates. scatter-side 1.274x signal is real but optimizes a <=5% slice.",
    "exactness":"G2 cos=1.0 (owner-reduction), extension-e2e cos=1.0 (4 groups) - PASS (but moot given G3 fail)",
    "projected_production_gain":"none (scatter is <=5% of backward; compute-bound inner loop dominates)",
    "memory_overhead":"scratch 64-159MB (<<40GB) - bounded, PASS (but moot)",
    "engineering_complexity":"high (split-build torch extension, two-pass integration) - for a moot target",
    "scene_robustness":"f contention-independent (contended==uncontended f~=-0.05); robust",
    "novelty_confidence":"moderate (selective per-Gaussian fanout owner-reduction is novel vs per-tile baselines, but the target slice is too small to matter)",
    "evidence":"_sgor0/SGOR0_DILUTION_ANALYSIS.md; SGOR0_DILUTION_RESULT.txt; SGOR0_STATUS.json"
  },
  "HGR0": {
    "verdict":"NO-GO_AUTHORITATIVE_TIMING_UNAVAILABLE (BLOCKED-EXTERNAL)",
    "status":"BLOCKED_EXTERNAL (oracle-pass, pending authoritative)",
    "backward_speedup":"oracle geomean 1.517x pooled (room1.367/garden1.975/bicycle1.294) - WOULD PASS gates, but authoritative timing unavailable",
    "exactness":"not confirmed vs real gsplat CUDA backward (prototype compositor canonical orthogonal to gsplat kernel)",
    "projected_production_gain":"oracle suggests positive, unconfirmed",
    "memory_overhead":"unknown (not measured)",
    "engineering_complexity":"unknown",
    "scene_robustness":"3-scene oracle all >1.1x",
    "novelty_confidence":"unknown",
    "evidence":"HGR0_VERDICT.json; BLOCKED on /root/epic05_p16rb (Permission denied) + exclusive A100"
  },
  "ABC0": {
    "verdict":"NO-GO_AUTHORITATIVE_TIMING_UNAVAILABLE (BLOCKED-EXTERNAL)",
    "status":"BLOCKED_EXTERNAL (oracle-pass, pending authoritative)",
    "backward_speedup":"oracle geomean 1.538x pooled (all scenes 1.538x) - WOULD PASS gates, but authoritative timing unavailable",
    "exactness":"not confirmed vs real gsplat CUDA backward (prototype compositor canonical orthogonal to gsplat kernel)",
    "projected_production_gain":"oracle suggests positive, unconfirmed",
    "memory_overhead":"unknown (not measured)",
    "engineering_complexity":"unknown",
    "scene_robustness":"3-scene oracle all 1.538x",
    "novelty_confidence":"unknown",
    "evidence":"ABC0_VERDICT.json; BLOCKED on /root/epic05_p16rb (Permission denied) + exclusive A100"
  },
  "HBR0": {
    "verdict":"NO-GO_AUTHORITATIVE_TIMING_UNAVAILABLE (BLOCKED-EXTERNAL, oracle-FAIL)",
    "status":"BLOCKED_EXTERNAL (oracle would FAIL)",
    "backward_speedup":"oracle geomean 1.0x (all scenes 1.0x) - would FAIL gates even if timed",
    "exactness":"not confirmed vs real gsplat CUDA backward",
    "projected_production_gain":"none (oracle 1.0x)",
    "memory_overhead":"unknown",
    "engineering_complexity":"unknown",
    "scene_robustness":"oracle 1.0x all scenes",
    "novelty_confidence":"low (prior-art risk flagged in goal; overlaps Gaussian-centric adaptive backward scheduling)",
    "evidence":"HBR0_VERDICT.json; held as fallback per goal; oracle would fail"
  }
}
n_confirmed_pass = sum(1 for v in lanes.values() if v["verdict"].startswith("PASS"))
n_oracle_pass_blocked = sum(1 for v in lanes.values() if "oracle-pass" in v["verdict"] or "oracle geomean" in str(v.get("backward_speedup","")) and "WOULD PASS" in str(v.get("backward_speedup","")))
master = {
  "deliverable":"EXACT_OPT_DEDUP_MASTER_VERDICT",
  "program":"EXACT_OPT_DEDUP_CONTINUATION",
  "date":"2026-10-03",
  "master_verdict":"NOT_ELIGIBLE (0 confirmed independent PASS)",
  "independent_pass_confirmed": n_confirmed_pass,
  "blocked_external_oracle_pass": ["HGR0","ABC0"],
  "composition_performed": False,
  "composition_reason":"No-rescue: a candidate must independently PASS before composition. 0 confirmed PASS. HGR0/ABC0 are oracle-pass but BLOCKED-EXTERNAL (inaccessible /root assets + no exclusive A100 + correctness-vs-real-kernel unconfirmed); they cannot be counted as PASS.",
  "lanes": lanes,
  "gate_reference": {"SGOR0_backward":"1.25x pooled & 1.10x every scene","PSR0_projection":"1.40x proj-bwd & 1.05x production","HGR0_ABC0_backward":"1.2x pooled & 1.1x every","HBR0_backward":"1.3x pooled & 1.15x every"},
  "external_blockers": {
    "root_assets":"/root/epic05_p16rb + /root/dfc1r1_20261002 gsplat-baseline = Permission denied (root-owned). HGR0/ABC0 SCENE_ROOT + sys.path locked here. No accessible copy in /mnt/storage_pool.",
    "exclusive_gpu":"all 8 A100-PCIE-40GB have root-owned evoseg/vllm RL tenant resident (idle as of 2026-10-03: vllm workers exited, 16 procs ~4GB/GPU 0% util, but 2 idle root procs/GPU remain -> not strictly exclusive). Authoritative timing requires exclusivity (fail-closed).",
    "ncu_profiling":"ERR_NVGPUCTRPERM - GPU perf counters root-restricted; ncu unusable by this user."
  },
  "resolution_path":"HGR0/ABC0 need (a) asset owner exposes /root/epic05_p16rb readably OR preregistered re-lock, AND (b) an exclusive A100 window, AND (c) correctness-vs-real-gsplat-kernel confirmation. None are locally actionable. SGOR0/PSR0/HBR0 are terminally resolved (no headroom / would-fail)."
}
json.dump(master, open("EXACT_OPT_DEDUP_MASTER_VERDICT.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
print("MASTER_VERDICT written; confirmed PASS =", n_confirmed_pass)
