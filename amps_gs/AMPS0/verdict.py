import json, datetime
v={
 "schema":"AMPS0_VERDICT","generated":datetime.date.today().isoformat(),
 "terminal_verdict":"DROP_INSUFFICIENT_OPTIMIZER_HEADROOM",
 "verdict_family":"DROP (fail-closed, P0 headroom gate)",
 "gate_triggered":"P0 hard headroom gate (objective section 5)",
 "fail_closed":True,
 "execution_stopped_after":"P0 (headroom gate + traffic model). P1 shadow numerics completed as supporting numerical evidence (CUDA-free, immediate-scope). P2/P3 NOT executed (fail-closed: do not skip a failed gate).",
 "evidence_summary":{
   "measured_optimizer_fraction_full_adam_M0_room15k":0.0594,
   "measured_sh_adam_wall_fraction_LAVOS0_all_scenes":0.14,
   "gate_threshold_drop_below":0.20,
   "all_scenes_below_threshold":True,
   "traffic_model_e2e_at_p0.14":{"C1":1.0315,"C2":1.0315,"C3":1.0344,"C4":1.0417},
   "traffic_gate_min":1.05,
   "even_C4_below_traffic_gate":True
 },
 "reasoning":[
   "The optimizer is only 5.94%-14% of iteration time on authoritative A100 strong-baseline measurements (full-Adam gsplat continuation room@15k; SH-Adam LAVOS0 all 3 scenes). Both are far below the 0.20 floor.",
   "The exact analytical traffic model shows that even the most aggressive candidate (C4, all m/v BF16, 28.6% optimizer HBM-traffic cut) yields only 1.017-1.042x E2E at measured p, below the 1.05x traffic gate.",
   "Reaching 1.08x (promote) would require p>=0.30; reaching 1.15x (strong) requires p>=0.40. The measured optimizer fraction cannot support this.",
   "A prior speed bound (M0_SPEED_BOUND) for the strictly stronger visible-skip optimizer variant (removes 100% of invisible-Gaussian optimizer work) reaches only 1.006x best-case. AMPS0 (BF16 state compression, strictly weaker than visible-skip) cannot exceed this.",
   "Conclusion: the central scientific premise of AMPS0 — that modern 3DGS training cost has shifted 'substantial' cost into Adam such that attribute-aware optimizer-state compression yields real end-to-end speedup — is NOT supported by measurement on the strong baseline. The optimizer is a minor fraction of iteration time."
 ],
 "what_would_change_the_verdict":[
   "A measured strong-baseline (Faster-GS fused) optimizer fraction >=0.30 on >=2 scenes/stages (none observed; 0.059-0.14 measured).",
   "Evidence that the fused backend materializes optimizer state in HBM per-iteration with measurable bandwidth that BF16 would halve (not measured; fused kernel may keep state in registers).",
   "A memory-capacity-bound regime (e.g., bicycle/garden at 4.7M Gaussians near A100 limits) where optimizer-state bytes dominate and force spilling — AMPS0 could then help capacity, but that is a different claim than speed."
 ],
 "novelty_status":"NOVELTY_NOT_ESTABLISHED (fresh literature audit required before any paper claim; prior-art audit classifies primary mechanism NO_DIRECT_GS_MATCH_FOUND but with PARTIAL_OVERLAP).",
 "do_not_claim":[
   "exact optimizer equivalence / identical trajectory / lossless compression / exact gradient (BF16 state storage is not mathematically exact Adam)",
   "'first mixed-precision Adam' or 'first compressed optimizer'",
   "any E2E speedup >=1.05x (not supported)"
 ],
 "allowed_wording_if_cited":[
   "mixed-precision optimizer-state storage",
   "FP32 update arithmetic with reduced-precision moment storage",
   "quality-preserving within measured benchmark tolerance (only if P1 numerics support it)"
 ],
 "artifacts":[
   "AMPS0_PREREG.json","AMPS0_PRIOR_ART_AUDIT.md","AMPS0_BASELINE_AUDIT.json",
   "AMPS0_HEADROOM_PROFILE.json","AMPS0_TRAFFIC_MODEL.json","AMPS0_SHADOW_NUMERICS.json",
   "AMPS0_VERDICT.json","AMPS0_REPORT.md"],
 "p1_supporting_note":"P1 shadow numerics (CUDA-free, immediate scope) tests whether BF16 moment storage is numerically viable in isolation. It does NOT rescue the headroom DROP; it characterizes the numerical ceiling of the mechanism for any future memory-capacity-bound lane."
}
json.dump(v, open("amps_gs/AMPS0/AMPS0_VERDICT.json","w"), indent=2)
print("VERDICT written:", v["terminal_verdict"])
