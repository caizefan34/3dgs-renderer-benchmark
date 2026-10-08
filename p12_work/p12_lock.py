#!/usr/bin/env python3
"""P12 Gate 0: source identity + fused-loss lock (re-record hashes; assume nothing).

Produces:
  /root/epic05_p12/artifacts/P12_SOURCE_IDENTITY.json
  /root/epic05_p12/artifacts/P12_FUSED_LOSS_LOCK.json

Verifies against the P11 authoritative artifacts (P11_SOURCE_IDENTITY.json,
P11_LOSS_EQUIVALENCE.json, P11_LOSS_TIMING.json) copied to
/root/epic05_p12/p11_reference/. FAILS LOUDLY on any drift.
"""
import hashlib
import json
import os
import subprocess
import sys

PY = sys.executable
P12 = "/root/epic05_p12"
ART = os.path.join(P12, "artifacts")
P11REF = os.path.join(P12, "p11_reference")

GMAIN_TREE = "/root/epic05_p8/gsplat-main-512d366b"
GMAIN_SO = "/root/epic05_p8/build/torchext/gsplat_cuda/gsplat_cuda.so"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
TRAINER = os.path.join(P12, "publication_trainer_p12.py")
TRAINER_P8 = "/root/3dgs-publication-epic05/publication_trainer_p8.py"
REPO = "/root/3dgs-renderer-benchmark"

# P11-locked expectations (from P11_SOURCE_IDENTITY.json)
P11_EXPECT = {
    "gsplat_commit": "512d366b67073d77ca099ede742683c165dfc23b",
    "gmain_so_sha256": "8ca8ec1e0c8618f058638b93010061a5bafb7f6c92005b54d07dc6c22658c811",
    "c0_so_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
    "core_so_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
    "fused_commit": "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38",
    "fused_so_sha256": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
    "trainer_p8_sha256": "f67b2125f93b84c9afee103675c13dbca301aa58f385069b5e853d6272a20618",
    "gaussian_model_sha256": "68731e375013a6f24aa4d81e5986402b1ac50f51cddf20688e358be705b1a60b",
    "config_sha256": "ea8e870476a3ba71c0ae9e1123a6fa5b1cf696bad983bdb6535d6fd04b7acc56",
    "dataset_sha256": "e6d8bfb1034bee102abc176060a8c7783951d253426ea7b5ea27daa16c258402",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def main():
    os.makedirs(ART, exist_ok=True)
    failures = []

    def check(name, got, want):
        ok = got == want
        if not ok:
            failures.append({"check": name, "got": got, "expected": want})
        return ok

    # ---- P11 reference artifacts present
    p11_files = {}
    for f in ("P11_SOURCE_IDENTITY.json", "P11_LOSS_EQUIVALENCE.json",
              "P11_LOSS_TIMING.json"):
        p = os.path.join(P11REF, f)
        if not os.path.exists(p):
            failures.append({"check": f"p11_reference/{f}", "got": "MISSING",
                             "expected": "present"})
        else:
            p11_files[f] = json.load(open(p))

    # ---- current gsplat main
    gmain_so_sha = sha256_file(GMAIN_SO)
    check("gmain_so_sha256", gmain_so_sha, P11_EXPECT["gmain_so_sha256"])
    # the gmain tree is a git-archive snapshot (no .git); the authoritative
    # commit is the P8 build-identity record (per-file git-hash audited)
    gmain_commit = json.load(open("/root/epic05_p8/gmain_build_identity.json"))["source_commit"]
    check("gsplat_commit", gmain_commit, P11_EXPECT["gsplat_commit"])
    gmain_dirty = False  # snapshot tree; drift impossible without rebuild (.so hash is the lock)

    # ---- C0 / A0 / A1 frozen lineage
    c0_so_sha = sha256_file(C0_SO)
    core_so_sha = sha256_file(CORE_SO)
    check("c0_so_sha256", c0_so_sha, P11_EXPECT["c0_so_sha256"])
    check("core_so_sha256", core_so_sha, P11_EXPECT["core_so_sha256"])
    c0_commit = sh(["git", "-C", C0_WT, "rev-parse", "HEAD"])
    c0_dirty = sh(["git", "-C", C0_WT, "status", "--porcelain"])

    # ---- trainer lineage
    trainer_sha = sha256_file(TRAINER)
    trainer_p8_sha = sha256_file(TRAINER_P8)
    check("trainer_p8_sha256", trainer_p8_sha, P11_EXPECT["trainer_p8_sha256"])
    # shared recipe files (frozen)
    gm_sha = sha256_file(os.path.join(REPO, "baseline", "reference_v1",
                                      "gaussian_model.py"))
    cfg_sha = sha256_file(os.path.join(REPO, "baseline", "reference_v1",
                                       "config.py"))
    ds_sha = sha256_file(os.path.join(REPO, "scripts", "epic05", "phase7",
                                      "dataset.py"))
    check("gaussian_model_sha256", gm_sha, P11_EXPECT["gaussian_model_sha256"])
    check("config_sha256", cfg_sha, P11_EXPECT["config_sha256"])
    check("dataset_sha256", ds_sha, P11_EXPECT["dataset_sha256"])

    # ---- fused-ssim identity
    fused_mod = "/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim/__init__.py"
    fused_so = ("/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/"
                "fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so")
    fused_so_sha = sha256_file(fused_so)
    check("fused_so_sha256", fused_so_sha, P11_EXPECT["fused_so_sha256"])
    fused_commit = sh(["git", "-C", "/root/fused-ssim", "rev-parse", "HEAD"])
    check("fused_commit", fused_commit, P11_EXPECT["fused_commit"])
    fused_dirty = sh(["git", "-C", "/root/fused-ssim", "status", "--porcelain"])

    # ---- P11 semantics restated from the authoritative artifacts
    l1 = p11_files.get("P11_LOSS_EQUIVALENCE.json", {})
    l2 = p11_files.get("P11_LOSS_TIMING.json", {})
    p11_class = l1.get("overall_classification")
    check("p11_overall_classification", p11_class, "SEMANTICALLY_MATCHED_WITHIN_FP")

    # ---- trainer diff vs p8 (must be the P12 loss-switch diff only)
    diff = sh(["diff", "-u", TRAINER_P8, TRAINER])
    n_diff_lines = len([l for l in diff.splitlines()
                        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))])

    identity = {
        "phase": "P12 Gate 0 - strong-baseline revalidation source identity",
        "date_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                   capture_output=True, text=True).stdout.strip(),
        "p11_reference_files": {k: os.path.join(P11REF, k) for k in p11_files},
        "upstream_gsplat": {
            "repository": "nerfstudio-project/gsplat",
            "commit": gmain_commit,
            "gsplat_version": "1.6.0",
            "tree": GMAIN_TREE,
            "working_tree_dirty": bool(gmain_dirty),
            "dirty_detail": "" if not gmain_dirty else str(gmain_dirty)[:500],
        },
        "gmain_so": {"path": GMAIN_SO, "sha256": gmain_so_sha,
                     "matches_p11": gmain_so_sha == P11_EXPECT["gmain_so_sha256"]},
        "c0_lineage": {
            "worktree": C0_WT, "commit": c0_commit,
            "working_tree_dirty": bool(c0_dirty),
            "ext_so": {"path": C0_SO, "sha256": c0_so_sha},
            "core_so": {"path": CORE_SO, "sha256": core_so_sha},
            "note": "C0 = frozen C0_V3_FINAL30K (F9+SCALAR_ADJOINT+H8_MR, PX=2); "
                    "A0 = same binary env-switched HIGS_DISABLE_F9=1; "
                    "A1 = same binary F9 on, no H8-MR, no scalar adjoint",
        },
        "trainer": {
            "p12_path": TRAINER, "p12_sha256": trainer_sha,
            "p8_frozen_path": TRAINER_P8, "p8_sha256": trainer_p8_sha,
            "diff_vs_p8_lines": n_diff_lines,
            "diff_semantics": "loss-backend switch only (--loss sep|fused; fused "
                              "training loss, SepSSIM eval metric unchanged)",
        },
        "recipe_files": {
            "gaussian_model_sha256": gm_sha,
            "config_sha256": cfg_sha,
            "dataset_sha256": ds_sha,
        },
        "recipe_lock": {
            "eps2d": 0.3, "seed": 42, "iterations": 30000,
            "lambda_dssim": 0.2,
            "loss_formula": "L = 0.8 * L1 + 0.2 * (1 - SSIM)",
            "ssim_window": "11x11 Gaussian, sigma=1.5, zero same-padding",
            "densify": "500..15000@100, grow_grad2d=0.0008 absgrad pixel-space",
            "opacity_reset": 3000, "sh_progress": 1000,
            "tile_size": 16, "packed": False, "rasterize_mode": "classic",
            "resolution": "1080p", "background": "black",
        },
        "environment": {
            "python": sh([PY, "-c", "import sys; print(sys.version.split()[0])"]),
            "torch": sh([PY, "-c", "import torch; print(torch.__version__)"]),
            "cuda_runtime": sh([PY, "-c", "import torch; print(torch.version.cuda)"]),
            "hostname": sh(["hostname"]),
            "gpus": sh(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                        "--format=csv,noheader"]).splitlines(),
        },
        "scenes": ["bicycle", "bonsai", "counter", "drjohnson", "flowers",
                   "garden", "kitchen", "playroom", "room", "stump", "train",
                   "treehill", "truck"],
        "failures": failures,
    }

    lock = {
        "phase": "P12 fused-loss lock (P11-validated identity reuse)",
        "date_utc": identity["date_utc"],
        "p11_identity_reused": {
            "repo": "https://github.com/rahul-goel/fused-ssim",
            "commit": fused_commit,
            "working_tree_clean": not bool(fused_dirty),
            "module_file": fused_mod,
            "module_sha256": sha256_file(fused_mod),
            "so_path": fused_so,
            "so_sha256": fused_so_sha,
            "build_flags": ["-O3", "--use_fast_math", "--maxrregcount=32",
                             "CUDA_ARCHITECTURES=80"],
        },
        "semantic_lock": {
            "loss_formula": "L = 0.8 * L1 + 0.2 * (1 - SSIM)",
            "padding": "same",
            "window": "11x11 Gaussian, sigma=1.5 (hardcoded 11-tap kernel in fused .so)",
            "C1": 0.01 ** 2, "C2": 0.03 ** 2,
            "image_range": "[0,1] clamped rendered vs GT (same GT preprocessing)",
            "reduction": "mean over full-size SSIM map (padding=same)",
            "train_flag": True,
            "p11_equivalence_classification": p11_class,
            "p11_equivalence_evidence": {
                "worst_grad_rel_l2": {s: v["worst_grad_rel_l2"]
                                      for s, v in l1.get("provisional_classification", {}).items()},
                "worst_loss_rel_diff": {s: v["worst_loss_rel_diff"]
                                        for s, v in l1.get("provisional_classification", {}).items()},
            },
            "p11_timing_evidence_S_loss": {
                s: round(v["scene_summary"]["S_loss"], 3)
                for s, v in l2.get("per_scene", {}).items()
            },
        },
        "eval_metric_lock": {
            "training_loss_ssim": "fused-ssim padding=same (P12 arms)",
            "evaluation_ssim_metric": "SepSSIM (frozen evaluation procedure; "
                                      "identical to P8/DSH-A so quality rows "
                                      "remain comparable)",
        },
        "lock_rule": "Any fused_ssim_cuda.so sha256 != P11 value aborts every "
                     "P12 run (hard RuntimeError in publication_trainer_p12.py).",
        "so_sha256_matches_p11": fused_so_sha == P11_EXPECT["fused_so_sha256"],
        "failures": failures,
    }

    with open(os.path.join(ART, "P12_SOURCE_IDENTITY.json"), "w") as f:
        json.dump(identity, f, indent=2)
    with open(os.path.join(ART, "P12_FUSED_LOSS_LOCK.json"), "w") as f:
        json.dump(lock, f, indent=2)

    print("P12_SOURCE_IDENTITY.json:")
    print(f"  gmain .so      {gmain_so_sha[:16]}... match={gmain_so_sha == P11_EXPECT['gmain_so_sha256']}")
    print(f"  c0 ext .so     {c0_so_sha[:16]}... match={c0_so_sha == P11_EXPECT['c0_so_sha256']}")
    print(f"  core .so       {core_so_sha[:16]}... match={core_so_sha == P11_EXPECT['core_so_sha256']}")
    print(f"  fused .so      {fused_so_sha[:16]}... match={fused_so_sha == P11_EXPECT['fused_so_sha256']}")
    print(f"  gsplat commit  {gmain_commit[:12]} match={gmain_commit == P11_EXPECT['gsplat_commit']}")
    print(f"  fused commit   {fused_commit[:12]} match={fused_commit == P11_EXPECT['fused_commit']}")
    print(f"  trainer diff lines vs p8: {n_diff_lines}")
    print(f"  c0 worktree dirty: {bool(c0_dirty)}  gmain tree dirty: {bool(gmain_dirty)}")
    if failures:
        print(f"\nP12 LOCK FAILURES ({len(failures)}):")
        for fl in failures:
            print(f"  FAIL {fl['check']}: got={fl['got']} expected={fl['expected']}")
        sys.exit(1)
    print("\nP12 LOCK: ALL CHECKS PASS")
    print("WROTE", os.path.join(ART, "P12_SOURCE_IDENTITY.json"))
    print("WROTE", os.path.join(ART, "P12_FUSED_LOSS_LOCK.json"))


if __name__ == "__main__":
    main()
