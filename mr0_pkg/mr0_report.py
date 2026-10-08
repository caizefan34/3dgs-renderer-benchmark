#!/usr/bin/env python3
"""MR0 report generator. Emits MR0_ENVIRONMENT.json, MR0_SOURCE_IDENTITY.json,
MR0_REPORT.md and MANIFEST.sha256 from the pipeline outputs. No production file
is modified."""
import sys, os, json, hashlib, glob, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/mr0_out"
CELLS = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]
KS = ["4", "8", "16"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    import torch
    gj0 = json.load(open("%s/room@10000_groups.json" % OUT))
    ckpt = {c: json.load(open("%s/%s_groups.json" % (OUT, c)))["ckpt_sha256"] for c in CELLS}
    Npts = {c: json.load(open("%s/%s_groups.json" % (OUT, c)))["N"] for c in CELLS}
    src = {
        "experiment": "MacroRep-GS MR0 - Direct Macro-Primitive Capacity Upper Bound",
        "frozen_prereg": "MR0_PREREG.md",
        "analyzer": "mr0_analyze.py (frozen, unmodified)",
        "ckpt_sha256": ckpt,
        "num_points": Npts,
        "renderer_bootstrap": "hm0_common.bootstrap_c0 (gsplat.csrc pre-registered)",
        "pinned_binaries": {
            "c0_ext": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
            "core": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
        },
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(src, open("%s/MR0_SOURCE_IDENTITY.json" % OUT, "w"), indent=2)

    prof = {}
    for c in CELLS:
        scene, it = c.split("@")
        prof[c] = json.load(open("/root/hm0_out/%s_%s.profile.json" % (scene, it)))["phase_frac_of_wall"]
    env = {
        "no_local_gpu": True,
        "remote_host": "EPIC-05 (ssh EPIC-05, 8x A100-SXM4-80GB)",
        "gpus_used": ["0", "1", "2", "6", "7"],
        "python": src["python"], "torch": src["torch"],
        "phase_frac_of_wall": prof,
        "proxies_and_subsamples": {
            "n_groups": 512,
            "K_values": KS,
            "coverage_levels": [0.25, 0.5, 0.75],
            "n_fit_steps": 6,
            "n_fit_cameras": 8,
            "fit_cam_selection": "per-group highest precomputed visibility TRAIN cams (prereg sec.4)",
            "n_vis_cams_construction": 32,
            "n_isect_reduction_sample_groups": 128,
            "n_isect_reduction_cams": 4,
            "intersection_reduction_units": "tile-gaussian intersection pairs (a0 capture patch); group and macro both tile-based",
            "response_perturbations": 12,
            "response_estimator": "forward finite differences in fit; exact autograd for the target",
            "note": "no local GPU; remote A100 only; no full training; optimization allowed to be expensive per prereg",
        },
    }
    json.dump(env, open("%s/MR0_ENVIRONMENT.json" % OUT, "w"), indent=2)

    verdict = json.load(open("%s/MR0_VERDICT.json" % OUT))
    summary = json.load(open("%s/MR0_SUMMARY.json" % OUT))

    L = []
    L.append("# MacroRep-GS MR0 - Direct Macro-Primitive Capacity Upper Bound")
    L.append("")
    L.append("Generated: %s" % src["generated_utc"])
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("**%s**  (selected_K=%s, strong_K=%s, promote_K=%s)"
             % (verdict["verdict"], verdict.get("selected_K"), verdict.get("strong_K"), verdict.get("promote_K")))
    L.append("")
    L.append("## Question")
    L.append("")
    L.append("Can many fine C0 Gaussians be replaced by far fewer directly-rendered,")
    L.append("view-independent macro Gaussians (one mean/covariance/opacity/C0 SH) while")
    L.append("preserving multi-view radiance/transmittance and collective first-order")
    L.append("response, removing BOTH raster work and N-scaling optimizer work?")
    L.append("This is an optimistic capacity oracle, not a training method.")
    L.append("")
    L.append("## Protocol summary")
    L.append("")
    L.append("- Cells: room/bicycle at 10K and 15K iterations (mipnerf 360).")
    L.append("- 512 intersection-weighted deterministic greedy-NN fine groups per cell,")
    L.append("  K in {4,8,16} nested; construction cams are TRAIN cams disjoint from eval.")
    L.append("- Fit ONE macro Gaussian per group per K on 8 per-group highest-visibility")
    L.append("  TRAIN cams; objective = RGB contribution err + alpha/transmittance err +")
    L.append("  12-perturbation first-order collective response err (forward FD).")
    L.append("- Greedy full-scene replacement at 25/50/75% fine-Gaussian coverage; eval on")
    L.append("  8 train-eval + 8 true-holdout cams; PSNR/SSIM delta, intersection-count")
    L.append("  reduction, primitive/variable reduction, response capture, HM0-predicted")
    L.append("  iteration speed.")
    L.append("")
    L.append("## Proxies and disclosures")
    L.append("")
    for k, v in env["proxies_and_subsamples"].items():
        L.append("- %s: %s" % (k, v))
    L.append("- Intersection-count reduction is measured on a 128-highest-weight group")
    L.append("  sample over 4 construction cams; both group and macro counts are")
    L.append("  tile-gaussian intersection pairs (a0 capture patch).")
    L.append("")
    L.append("## Results at 75% coverage")
    L.append("")
    L.append("| cell | K | train dPSNR | holdout dPSNR med | holdout dPSNR worst | SSIM d | resp cos med | resp cos p10 | isect red | var red | speedup | nan/inf |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CELLS:
        for k in KS:
            x = summary["cells"][c]["K"][k]["coverage75"]
            L.append("| %s | %s | %+.3f | %+.3f | %+.3f | %+.4f | %.3f | %.3f | %.3f | %.3f | %.2fx | %d |"
                     % (c, k, x["train_psnr_delta_median_db"], x["truehold_psnr_delta_median_db"],
                        x["truehold_psnr_delta_worst_db"], x["ssim_delta_median"],
                        x["response_cos_median"], x["response_cos_p10"],
                        x["intersection_reduction"], x["optimized_variable_reduction"],
                        x["predicted_iteration_speedup"], x["nan_inf_count"]))
    L.append("")
    L.append("## Gate (prereg sec.6, single K across BOTH scenes at BOTH 10K/15K)")
    L.append("")
    L.append("MR0_STRONG_PROMOTE requires, at 75% coverage: train/holdout PSNR med >= -0.08 dB,")
    L.append("holdout worst >= -0.20 dB, SSIM med >= -0.002, response cos med >= 0.90, p10 >= 0.80,")
    L.append("intersection reduction >= 0.65, variable reduction >= 0.65, predicted speed >= 3.0x, no NaN/Inf.")
    L.append("MR0_PROMOTE relaxes these to -0.15/-0.35/-0.004/0.82/0.68/0.55/0.55/3.0.")
    L.append("A single K must pass on BOTH scenes at BOTH iterations.")
    L.append("")
    sp = {c: summary["cells"][c]["K"]["16"]["coverage75"]["predicted_iteration_speedup"] for c in CELLS}
    L.append("Predicted iteration speed at K=16, 75% coverage: " + ", ".join("%s=%.2fx" % (c, sp[c]) for c in CELLS))
    L.append("")
    L.append("room fails the 3.0x speedup gate at every K<=16 (room@10000: "
             + ", ".join("K%s=%.2fx" % (k, summary["cells"]["room@10000"]["K"][k]["coverage75"]["predicted_iteration_speedup"]) for k in KS)
             + "), so no single K passes both scenes -> MR0_DROP.")
    L.append("")
    L.append("## Interpretation")
    L.append("")
    L.append("MR0_DROP: a directly rendered macro Gaussian does not give enough local")
    L.append("capacity (at the tested K<=16) to remove both raster and N-scaling optimizer")
    L.append("work while holding quality. This does not rule out macro/anchor representations")
    L.append("in general; it rules out this direct single-macro-per-group upper bound at")
    L.append("these capacities. PROMOTE would only have justified a new training-representation study.")
    L.append("")
    open("%s/MR0_REPORT.md" % OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")

    files = sorted(glob.glob("%s/MR0_*" % OUT))
    man = []
    for f in files:
        if os.path.basename(f) == "MANIFEST.sha256":
            continue
        man.append("%s  %s" % (sha256_file(f), os.path.basename(f)))
    open("%s/MANIFEST.sha256" % OUT, "w").write("\n".join(man) + "\n")
    print("REPORT_DONE")


if __name__ == "__main__":
    main()
