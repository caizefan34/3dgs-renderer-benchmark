import csv, json, platform, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = root / "artifacts" / "higs-f9-1"
out.mkdir(parents=True, exist_ok=True)
raw = {s: json.loads((out / f"{s}_raw.json").read_text()) for s in ("room", "bicycle", "garden")}

def dump(name, value):
    (out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

consumer_audit = {
 "blend": {"class":"BLEND_REQUIRED","reads":["means2d","conics","colors_eval","opacities_eval","tile_offsets","flatten_ids","render_alphas","last_ids"],"compact_master_required":False},
 "projection_vjp":{"class":"PROJECTION_VJP_REQUIRED","baseline_reads":["v_means","v_quats","v_scales"],"f9_reads":["means[visible_ids[g]]","quats[visible_ids[g]]","scales[visible_ids[g]]"],"writes":"master gradients via visible_ids"},
 "sh_vjp":{"class":"SH_VJP_REQUIRED","baseline_reads":["v_means","v_SH"],"f9_reads":["means[visible_ids[g]]","SH[visible_ids[g]]"],"writes":"master gradients via visible_ids"},
 "python":{"class":"PYTHON_ONLY","reads":["means2d_proxy","densification radii","visible_ids"]},
 "compact_geometry":{"class":"NOT_REQUIRED","in_f9":True},
}
dump("consumer_audit.json", consumer_audit)

state = {"baseline_compact_bytes_per_visible":236,"components":{"v_means":12,"v_quats":16,"v_scales":12,"v_opacities":4,"v_sh_degree3":192},"f9_compact_tensors":[],"per_scene":{}}
for s,d in raw.items():
 n=d["correctness"]["visible"][0]; state["per_scene"][s]={"n_visible":n,"bytes_removed":n*236,"mib_removed":n*236/2**20}
dump("saved_state_before_after.json",state)

forward = {}
backward = {}
rows=[]
for s,d in raw.items():
 t=d["timing"]; c=d["correctness"]
 bf=t["base_forward"]["median_ms"]; ff=t["f9_forward"]["median_ms"]
 bfb=t["base_fb"]["median_ms"]; ffb=t["f9_fb"]["median_ms"]
 bwd=bfb-bf; fbwd=ffb-ff
 forward[s]={"classification":"ALGEBRAIC_EXACT_FP_REASSOCIATED","frame":c["frame"],"alpha":c["alpha"],"support_mismatch":"not individually re-captured in F9-1; F9-0 oracle was zero","intersection_counts":c["n_isects"],"visible_ids":c["visible"],"structural_f4":{"tile_offsets":"not independently re-captured in F9-1 smoke","flatten_ids":"not independently re-captured in F9-1 smoke"}}
 backward[s]={"grads":c["grads"],"support_mismatch":"not measured","envelope_status":"not established against the separately frozen scalar-adjoint noise envelope","result":"diagnostic parity only"}
 for key,stat in t.items(): rows.append({"scene":s,"measurement":key,"kind":"F+B" if key.endswith("fb") else "forward","median_ms":stat["median_ms"],"mean_ms":stat["mean_ms"],"p10_ms":stat["p10_ms"],"p90_ms":stat["p90_ms"],"std_ms":stat["std_ms"],"samples":len(stat["samples_ms"])})
 rows.extend([{"scene":s,"measurement":"derived_backward_base","kind":"backward","median_ms":bwd},{"scene":s,"measurement":"derived_backward_f9","kind":"backward","median_ms":fbwd},{"scene":s,"measurement":"forward_gain_percent","kind":"derived","median_ms":(bf-ff)/bf*100},{"scene":s,"measurement":"fb_gain_percent","kind":"derived","median_ms":(bfb-ffb)/bfb*100}])
dump("forward_correctness.json",forward); dump("backward_correctness.json",backward)

with (out/"stage_timing.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

dump("timing_sanity.json",{"authoritative":"composed CUDA event around each complete path","historical_f9_0":{"room":{"isolated_median_sum_ms":0.512,"chain_median_ms":0.482304},"bicycle":{"isolated_median_sum_ms":0.588,"chain_median_ms":0.479232}},"explanation":["a median of isolated samples is not additive","the isolated F1/F2/F3 sets have distinct cache and allocator state","the composed event is therefore used for decisions"],"f9_1_protocol":{"warmup":5,"samples":20,"interleaved":True,"limitation":"early diagnostic run; it did not complete the requested 20 warmup, 100 samples, 5 repetitions"}})

with (out / "backward_access_variants.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant", "tested", "projection_access", "sh_access", "room_backward_delta_percent", "bicycle_backward_delta_percent", "garden_backward_delta_percent", "note"])
    w.writeheader()
    w.writerows([
        {"variant":"A_direct_master", "tested":True, "projection_access":"master[visible_ids[g]]", "sh_access":"master[visible_ids[g]]", "room_backward_delta_percent":1.2264, "bicycle_backward_delta_percent":0.6987, "garden_backward_delta_percent":2.3442, "note":"selected implementation"},
        {"variant":"B_backward_gather", "tested":False, "note":"not justified before completeness gate"},
        {"variant":"C_minimum_saved_state", "tested":False, "note":"not justified before completeness gate"},
    ])
dump("resources.json",{"sm80":{"f9_producer":{"registers_per_thread":43,"static_shared_bytes":0,"dynamic_shared_bytes":0,"local_bytes":0,"spills":0,"theoretical_occupancy":"75% (43 registers, A100 SM80)"},"projection_vjp_direct_master":{"registers_per_thread":96,"static_shared_bytes":0,"local_bytes":0,"spills":0,"occupancy":"~31.25% register limited"},"sh_vjp_direct_master":{"registers_per_thread":48,"static_shared_bytes":0,"local_bytes":0,"spills":0,"occupancy":"62.5%"}},"collection":"cuobjdump --dump-resource-usage","hardware":"A100-PCIE-40GB sm_80"})
dump("memory_traffic.json",{"logical":{"removed_compact_materialization_bytes_per_visible":236,"removed_compact_reads_forward_bytes_per_visible":236,"f9_final_projected_write_bytes_per_camera_visible":48,"master_direct_read_projection_bytes_per_camera_visible":40,"master_direct_read_sh3_bytes_per_camera_visible":204},"dram":{"measured":False,"reason":"no Nsight/CUPTI transaction collection in the diagnostic run"},"global_writes_eliminated":"236 B/visible compact state; projected-state writes remain"})
dump("densification_trace.json",{"status":"NOT_RUN","reason":"F9-1 is WEAK before the correctness and performance authorization gates; no topology-changing trace or 5K training is authorized.","known":{"visibility_ids":"counts equal in diagnostic fixtures","densification_radii":"not independently compared"}})
net={}
for s,d in raw.items():
 t=d["timing"];bf=t["base_forward"]["median_ms"];ff=t["f9_forward"]["median_ms"];bfb=t["base_fb"]["median_ms"];ffb=t["f9_fb"]["median_ms"]
 net[s]={"forward_base_ms":bf,"forward_f9_ms":ff,"forward_gain_percent":(bf-ff)/bf*100,"fb_base_ms":bfb,"fb_f9_ms":ffb,"fb_gain_percent":(bfb-ffb)/bfb*100,"derived_backward_delta_ms":(ffb-ff)-(bfb-bf)}
dump("net_analysis.json",{"scenes":net,"classification":"F9_1_WEAK","why":"end-to-end forward gain is below 10% on all scenes, despite small F+B gains; mandatory full exactness/densification protocol remains incomplete."})
dump("provenance.json",{"host":"mx / bms-39468022-001","gpu":"A100-PCIE-40GB, SM80","torch":"2.9.1+cu128","cuda":"12.8","source":"B2 freeze 77ab983ffe43420b2131669cb35776b883ca4c3c plus authoritative trainable overlay","worktree":"/tmp/f9-1-trainable-worktree","cuda_visible_devices":"3","date":"2026-09-21"})

(root/"reports/higs").mkdir(parents=True,exist_ok=True)
report=["# F9-1 — Trainable Gatherless Integration", "", "## Result", "", "**F9_1_WEAK.** The prototype integrates a direct-master F9 producer and gatherless VJP reads, but the diagnostic end-to-end forward gain is below 10% on room, bicycle, and garden. Exactness, densification, and the prescribed 5×100 protocol remain incomplete; production integration and 5K training are not authorized.", "", "## Audit and contract", "", "Blend backward consumes only projected/raster state. Projection VJP now resolves `gid = visible_ids[g]` before reading master means/quaternions/scales; SH VJP does the same for master means and SH coefficients. Master-gradient writes retain the existing `visible_ids` mapping. The F9 branch saves no compact `v_means`, `v_quats`, `v_scales`, `v_opacities`, or `v_SH`.", "", "The explicit F9 contract is FP32, pinhole, RGB, uncompressed degree-3 SH; unsupported modes fall back to the baseline path.", "", "## Diagnostic timings", "", "| scene | forward base → F9 (ms) | forward gain | F+B base → F9 (ms) | F+B gain | derived backward delta |", "|---|---:|---:|---:|---:|---:|"]
for s in ("room","bicycle","garden"):
 x=net[s];report.append(f"| {s} | {x['forward_base_ms']:.3f} → {x['forward_f9_ms']:.3f} | {x['forward_gain_percent']:.2f}% | {x['fb_base_ms']:.3f} → {x['fb_f9_ms']:.3f} | {x['fb_gain_percent']:.2f}% | {x['derived_backward_delta_ms']:+.3f} ms |")
report += ["", "## Correctness and limits", "", "F4 intersection counts and visible counts matched in the diagnostic runs (room 953,144; bicycle 1,412,193; garden 533,928). Render RGB/alpha matched exactly in these runs; gradient relative-L2 differences were small but this is only a diagnostic comparison, not the frozen SCALAR_ADJOINT/H2-BWD-2R envelope. Flatten IDs/tile offsets, densification trace, and 5K training were intentionally not claimed as passed.", "", "The F9-0 isolated chain discrepancy is explained by different cache/allocator state and non-additivity of medians; a complete-chain CUDA event is authoritative.", "", "See the JSON/CSV artifacts for raw samples, resource usage, consumer audit, and traffic accounting."]
(root/"reports/higs/f9-1-trainable-gatherless.md").write_text("\n".join(report)+"\n")
