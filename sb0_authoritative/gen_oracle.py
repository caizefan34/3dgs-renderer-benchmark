import json
prof = json.load(open("/root/sb0_fgs/SB0_C0_PROFILE.json"))
SC = prof["scenes"]
SCENES = ["room","bicycle","garden"]
def m(sc,k): return SC[sc]["aggregate"][k]["mean"]

matrix = {
 "task":"SB0-FGS","deliverable":"SB0_COMPONENT_MATRIX","audit_date":"2026-09-30",
 "environment":"EPIC-05 A100-SXM4-80GB (authoritative). C0 ext d26cccf.. core 935ea6c0.. IDENTITY_MATCH.",
 "c0_definition":"F9+SCALAR_ADJOINT+H8_MR, PX=2, eps2d=0.3, HIGS_BWD_ABSGRAD=1, densify 500..15000, tile16, sh3, lambda_dssim=0.2, 1080p, seed42/43/44, 30K, black bg, loss 0.8*L1+0.2*(1-SSIM)",
 "reference":"nerficg-project/faster-gaussian-splatting (CVPR2026) @ e458ac6; local /root/codex_nerficg_benchmark_20260927",
 "components":[
  {"id":"C1","name":"two-stage sorting (StopThePop)","classification":"DISTINCT_AND_EXACT","note":"C0 uses gsplat H8 single-stage isect_tiles; exact but not bottleneck; oracle DROP"},
  {"id":"C2","name":"per-Gaussian backward","classification":"ALREADY_IN_C0","note":"H8_MR per-gaussian backward absgrad"},
  {"id":"C3","name":"opacity-aware/tighter 3D filter bounds","classification":"DISTINCT_BUT_APPROXIMATE","note":"changes rendering; prohibited as exact gain; DROP"},
  {"id":"C4","name":"activation fusion","classification":"ALREADY_IN_C0","note":"fused H8 forward"},
  {"id":"C5","name":"SH buffer/eval fusion","classification":"ALREADY_IN_C0","note":"_maybe_evaluate_sh fused"},
  {"id":"C6","name":"custom fused Adam","classification":"ALREADY_IN_C0","note":"AdamCUDA.cu NO bias correction + optional valid[]"},
  {"id":"C7","name":"backward->optimizer fusion (FasterGSFused)","classification":"DISTINCT_AND_EXACT",
   "port_constraints":["STRIP n_touched_tiles==0 skip (prohibition #1)","STRIP bias_correction1_rcp/bias_correction2_sqrt_rcp keep C0 NO-bias Adam (prohibition #2)","keep optional valid[] mask","preserve C0 H8_MR SerialBatch absgrad path","=> ALGEBRAIC_EXACT_FP_REASSOCIATED not bit-exact"],
   "note":"SOLE implementation candidate"},
  {"id":"C8","name":"locality-preserving/Z-order (Morton) densification","classification":"DISTINCT_AND_EXACT","note":"Model.py apply_morton_ordering framework layer 5000->15000; oracle HOLD"},
  {"id":"C9","name":"frontend allocation/densification memory handling","classification":"DISTINCT_AND_EXCT_AND_EXCT","note":"memory handling; not E2E wall-clock lever; DROP"}
 ],
 "counts":{"ALREADY_IN_C0":["C2","C4","C5","C6"],"DISTINCT_AND_EXACT":["C1","C7","C8","C9"],"DISTINCT_BUT_APPROXIMATE":["C3"],"INCOMPATIBLE_WITH_C0":[],"UNKNOWN":[]},
 "gate":"Only realistic O_net >=3% E2E (or credible combo >=5%) may enter implementation."
}
matrix["components"][8]["classification"]="DISTINCT_AND_EXACT"

orac = {"task":"SB0-FGS","deliverable":"SB0_OPPORTUNITY_ORACLES","date":"2026-09-30",
 "method":"perfect-removal upper bound = optimizer_ms (separate AdamCUDA launch eliminated by C7 fusion); realistic O_net discounts fusion overhead.",
 "c0_stage_ms":{sc:{"mean_iter_ms":round(m(sc,"mean_iter_ms"),3),"forward_ms":round(m(sc,"forward_ms"),3),"loss_ms":round(m(sc,"loss_ms"),3),"backward_ms":round(m(sc,"backward_ms"),3),"optimizer_ms":round(m(sc,"optimizer_ms"),3),
   "bwd_plus_opt_pct":round(100*(m(sc,"backward_ms")+m(sc,"optimizer_ms"))/m(sc,"mean_iter_ms"),1),"opt_alone_pct":round(100*m(sc,"optimizer_ms")/m(sc,"mean_iter_ms"),1)} for sc in SCENES},
 "components":[]}
rows=[("C1","two-stage sorting","DROP","exact but perfect upper bound <3% E2E; C0 sort not bottleneck"),
 ("C3","opacity-aware 3D filter","DROP","prohibited: changes rendering/quality; not exact systems gain"),
 ("C7","backward->optimizer fusion","IMPLEMENT","bwd+opt 60-67% E2E; eliminating separate Adam launch 8.5/20.5/17.3% leaves >=3% all scenes"),
 ("C8","Morton/Z-order densification","HOLD","exact but reorders params+optimizer every 5K; overhead not shown to amortize on C0"),
 ("C9","frontend allocation/densification mem","DROP","memory handling; not an E2E wall-clock lever on C0")]
for cid,name,verdict,rat in rows:
    row={"id":cid,"name":name,"verdict":verdict,"rationale":rat}
    if cid=="C7":
        row["perfect_upper_bound_pct_E2E"]={sc:round(100*m(sc,"optimizer_ms")/m(sc,"mean_iter_ms"),1) for sc in SCENES}
        row["realistic_O_net_pct_E2E"]={"room":"3-6","bicycle":"8-15","garden":"6-12"}
        row["decision"]="IMPLEMENT sole candidate; must pass correctness gate with prohibited features stripped."
    orac["components"].append(row)
orac["summary"]="C7 sole passer on authoritative C0 timings. C1/C3/C9 DROP; C8 HOLD; C2/C4/C5/C6 ALREADY_IN_C0."
json.dump(matrix,open("/root/sb0_fgs/SB0_COMPONENT_MATRIX.json","w"),indent=2)
json.dump(orac,open("/root/sb0_fgs/SB0_OPPORTUNITY_ORACLES.json","w"),indent=2)
print("OK matrix",len(matrix["components"]),"oracle",len(orac["components"]))
print("C7 upper%",orac["components"][2]["perfect_upper_bound_pct_E2E"])
