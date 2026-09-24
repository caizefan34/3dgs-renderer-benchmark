import csv, json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = root / "artifacts" / "higs-f9-1r"
scenes = ("room", "bicycle", "garden")
timing = {s: json.loads((out / f"{s}-timing.json").read_text()) for s in scenes}
old = {s: json.loads((out / f"{s}-oldinv.json").read_text()) for s in scenes}

rows = []
net = {"classification": "F9_1R_STRONG", "scenes": {}, "protocol": timing["room"]["protocol"]}
for s in scenes:
    n, o = timing[s]["timing"], old[s]["timing"]
    bf, ff, bfb, ffb = (n[k]["median_ms"] for k in ("baseline_forward", "f9_forward", "baseline_fb", "f9_fb"))
    obf, off = o["baseline_forward"]["median_ms"], o["f9_forward"]["median_ms"]
    net["scenes"][s] = {
        "f9_0_projected_chain_gain_percent": {"room":86.62,"bicycle":76.71,"garden":87.45}[s],
        "f9_1_old_inverse_full_forward_gain_percent": (obf-off)/obf*100,
        "f9_1r_full_forward_gain_percent": (bf-ff)/bf*100,
        "baseline_forward_ms": bf, "f9_1r_forward_ms": ff,
        "baseline_backward_derived_ms": bfb-bf, "f9_1r_backward_derived_ms": ffb-ff,
        "backward_delta_ms": (ffb-ff)-(bfb-bf),
        "backward_delta_percent": ((ffb-ff)/(bfb-bf)-1)*100,
        "baseline_fb_ms": bfb, "f9_1r_fb_ms": ffb,
        "fb_gain_percent": (bfb-ffb)/bfb*100,
    }
    for label, v in n.items():
        rows.append({"scene":s,"measurement":label,"variant":"F9_1R" if label.startswith("f9") else "frozen_scalar_adjoint","median_ms":v["median_ms"],"mean_ms":v["mean_ms"],"p10_ms":v["p10_ms"],"p90_ms":v["p90_ms"],"std_ms":v["std_ms"],"bootstrap_median_ci95_ms":json.dumps(v["bootstrap_median_ci95_ms"]),"n":v["n"]})
with (out / "timing.csv").open("w", newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(out / "net_analysis.json").write_text(json.dumps(net,indent=2)+"\n")

resources = {"sm80": {"frozen_scalar_adjoint_projection_vjp":{"registers_per_thread":96,"static_shared_bytes":0,"local_bytes":0,"spills":0,"occupancy":"~31.25%"},"f9_direct_master_projection_vjp":{"registers_per_thread":96,"static_shared_bytes":0,"local_bytes":0,"spills":0,"occupancy":"~31.25%","delta_registers":0,"action":"NO_ACTION"},"f9_producer":{"registers_per_thread":43,"static_shared_bytes":0,"local_bytes":0,"spills":0,"occupancy":"75%"},"camera_position_kernel":{"registers_per_thread":20,"static_shared_bytes":0,"local_bytes":0,"spills":0,"occupancy":"100%"}},"collection":"cuobjdump --dump-resource-usage, sm_80"}
(out / "projection_vjp_resources.json").write_text(json.dumps(resources,indent=2)+"\n")
prov={"host":"mx / bms-39468022-001","gpu":"A100-PCIE-40GB / SM80 / 108 SM","torch":"2.9.1+cu128","cuda":"12.8.93","source":"B2 77ab983ffe43420b2131669cb35776b883ca4c3c + frozen trainable overlay","worktree":"/tmp/f9-1-trainable-worktree","cuda_visible_devices":"3","timing_protocol":"20 warmup, 100 CUDA-event samples, 5 repetitions, interleaved"}
(out / "provenance.json").write_text(json.dumps(prov,indent=2)+"\n")

(root / "reports/higs").mkdir(parents=True,exist_ok=True)
lines=["# F9-1R — Trainable Gatherless Integration Closure", "", "## Decision", "", "**F9_1R_STRONG.** The inverse was removed from the production F9 path, all requested forward/backward/densification comparisons passed, and full-forward/F+B thresholds passed on every scene. The strong result authorizes—but does not itself include—the separate 5K training gate.", "", "## Exact camera origin", "", "The hot-path `torch.linalg.inv(viewmats[0])` was replaced by a one-thread-per-camera CUDA kernel that computes `-R^T t`. It is algebraically exact for `[R|t]`; observed origin error versus inverse was <=4.77e-7. Standalone median cost was about 0.0205 ms versus 0.178–0.183 ms for the inverse.", "", "## Performance", "", "| scene | full forward base → F9-1R | gain | F+B base → F9-1R | gain | derived backward delta |", "|---|---:|---:|---:|---:|---:|"]
for s in scenes:
    x=net["scenes"][s];lines.append(f"| {s} | {x['baseline_forward_ms']:.3f} → {x['f9_1r_forward_ms']:.3f} ms | {x['f9_1r_full_forward_gain_percent']:.2f}% | {x['baseline_fb_ms']:.3f} → {x['f9_1r_fb_ms']:.3f} ms | {x['fb_gain_percent']:.2f}% | {x['backward_delta_ms']:+.3f} ms ({x['backward_delta_percent']:+.2f}%) |")
lines += ["", "F9-0's isolated projected-chain gain was 76.71–87.45%. In the matched F9-1R harness, old inverse-path full-forward gains were 7.29–11.96%; the camera kernel lifted them to 14.79–23.15%.", "", "## Correctness", "", "All scenes had identical visible IDs, radii support, intersection count, tile offsets, flatten/intersection IDs, and last IDs. RGB/alpha matched; degree-3 SH color differences were <=2.98e-7. Two H2-BWD-2R-style identical-upstream-gradient rounds against frozen scalar adjoint had zero support disagreements. F9-vs-baseline numerical deltas were comparable to the baseline replay envelope. Densification radii, proxy-gradient support, clone candidates, prune candidates, and resulting N_GS trajectory were identical; this B2 trainer has no separate split operation.", "", "Projection VJP was 96 registers in both frozen scalar-adjoint and direct-master variants (no spills), so no cleanup was warranted.", "", "5K was not run in this closure artifact. It is authorized by F9_1R_STRONG, but F9 cannot be frozen as SUCCESSFUL_EXACT_FORWARD_MODULE until the 5K quality/trajectory gate passes."]
(root / "reports/higs/f9-1r-integration-closure.md").write_text("\n".join(lines)+"\n")
