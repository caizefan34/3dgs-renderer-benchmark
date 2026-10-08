import json
from pathlib import Path
H0=Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_H0")
frozen=json.loads((H0/"H0_PERPIXEL_RHO.json").read_text())["cam_0"]
rows=[]
rows.append({"grouping":"frozen kNN target_size=64 (PREREG)","cell_m":None,"G":98804,"eff_groups_per_pixel":frozen["eff_groups_per_pixel_mean"],"rho":frozen["rho_per_pixel"],"arith_ceiling":frozen["arith_ceiling_per_pixel"],"is_frozen":True})
for s in ["0p25","0p5","1p0"]:
    d=json.loads((H0/("H0_VOXEL_RHO_cell%s.json"%s)).read_text())
    cell=d["cell"]
    kbar=d["per_pixel_canon"]/max(1.0,d["eff_groups_per_pixel"]*16*256/ (16*256))  # canon/eff_groups normalized
    kbar=d["per_pixel_canon"]/(d["eff_groups_per_pixel"]* (d["per_pixel_canon"]/d["per_pixel_canon"]))  # placeholder
    # proper: mean k = (canon/npix)/eff_groups ; npix unknown here, use canon/eff_groups/total_pix_proxy
    rows.append({"grouping":"voxel cell=%.2fm"%cell,"cell_m":cell,"G":d["G"],"eff_groups_per_pixel":d["eff_groups_per_pixel"],"rho":d["rho_per_pixel"],"arith_ceiling":d["arith_ceiling"],"is_frozen":False})
out={"schema":"comax0.h0_grouping_granularity_sweep.v1",
 "experiment":"COMAX0 H0 grouping-granularity sweep (FUTURE_HYPOTHESIS method-version exploration)",
 "date":"2026-10-05","scene":"room cam0",
 "method":"per-pixel record rejection (16x16 px, bbox-coverage, depth-sorted, per-group running-max). Replaces frozen kNN grouping with view-independent coarse VOXEL grouping (cell in meters).",
 "frozen_baseline":{"grouping":"kNN target_size=64 normal-angle 20deg","rho":frozen["rho_per_pixel"],"ceiling":frozen["arith_ceiling_per_pixel"],"eff_groups_per_pixel":frozen["eff_groups_per_pixel_mean"]},
 "sweep":rows,
 "finding":"rho and ceiling rise MONOTONICALLY as grouping coarsens (fewer groups/pixel -> higher per-ray overlap k -> higher record rejection). The record MECHANISM is NOT the bottleneck: coarser grouping reaches rho 0.65->0.83 (ceiling 2.88x->6.05x), exceeding the 2.0x promotion and 4.0x target gates. The frozen kNN grouping (target_size 64) is quality-preserving but caps rho at ~0.36.",
 "binding_constraint":"QUALITY-SPEED TRADEOFF controlled by grouping granularity. Coarser grouping strengthens the within-group correlation assumption (more gaussians modeled as perfectly correlated in coverage), which raises rejection but degrades the rendered image (larger deviation from ordinary GS). The 2.0x gate is reachable ONLY by sacrificing quality (unmeasured here; E1 quality gate would catch it).",
 "caveat":"FUTURE_HYPOTHESIS / method-version exploration. NOT the frozen preregistered COMAX. Master verdict unchanged (INCONCLUSIVE_BUDGET); frozen H0 verdict unchanged (MARGINAL_HEADROOM). The high-rho coarse groupings are a DIFFERENT method version with different quality characteristics and would require full E1 quality re-validation. NUMERICS/PROFILING ONLY; NOT E2 timing.",
 "future_hypothesis":"A quality-aware adaptive grouping (fine where coverage is decorrelated, coarse where correlated) could in principle raise rho toward 0.5+ while keeping PSNR/SSIM/LPIPS within the E1 gate -- an open method-design question, out of H0 scope."}
(H0/"H0_GROUPING_GRANULARITY_SWEEP.json").write_text(json.dumps(out,indent=2,allow_nan=False))
print("WROTE H0_GROUPING_GRANULARITY_SWEEP.json")
for r in rows: print(r["grouping"],"G",r["G"],"eff_g",round(r["eff_groups_per_pixel"],1),"rho",r["rho"],"ceil",r["arith_ceiling"],"FROZEN" if r["is_frozen"] else "")
