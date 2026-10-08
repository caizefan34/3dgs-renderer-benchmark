#!/usr/bin/env python3
"""AB0 blocker-state artifact generator (read-only verification outcome).

Produces the prereg-required artifacts for the AB0_CHECKPOINT_BLOCKER stop:
  AB0_SOURCE_IDENTITY.json, AB0_ENVIRONMENT.json, AB0_CHECKPOINT_INVENTORY.json,
  AB0_VERDICT.json, AB0_REPORT.md

No experiment is run. No checkpoint is substituted. All values come from the
read-only EPIC-05 verification performed 2026-09-30.
"""
import json, hashlib, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
GEN_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

IDENTITY = {
    "schema_version": "AB0-v1",
    "artifact": "AB0_SOURCE_IDENTITY.json",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "authoritative_host": "EPIC-05",
    "host_addr": "8.130.30.251:1024",
    "host_user": "root",
    "host_name": "dsw-933622-64cc845c7d-lqswc",
    "c0_extension": {
        "pinned_commit": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11e",
        "built_so_path": "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
        "built_so_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11e",
        "note": "built .so sha256 equals pinned C0 extension commit; identity PASS. c0_ext dir has no .git; identity rests on built .so hash (same method as SB0_REPORT.md:8/49).",
        "identity_match": True,
    },
    "core_gsplat_cuda_so": {
        "pinned_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
        "found_path": "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
        "found_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
        "identity_match": True,
    },
    "publication_trainer": {
        "path": "/root/3dgs-publication-epic05/publication_trainer.py",
        "sha256": "1b67171d93bc1632d94f4b63e0db9bc859c27309eb45d935bcd1d6f27c7dac4f",
    },
    "cameras_json_sha256": {
        "room": "84fcccbd3ae4ea6754bcb7838149a418b3a62275549a67a3e7a7afc5d2f83f45",
        "bicycle": "5f0ebcb6b33e415061a485ed9961fb277cf7e875586f0d82d88b746d9d9fd502",
    },
    "camera_dataset_counts": {"room": 311, "bicycle": 194},
    "identity_gate": "PASS",
}

ENVIRONMENT = {
    "schema_version": "AB0-v1",
    "artifact": "AB0_ENVIRONMENT.json",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "authoritative_host": "EPIC-05",
    "gpu": "NVIDIA A100-SXM4-80GB",
    "gpu_count_note": "verified 1 visible device in this session",
    "gpu_memory_total_mib": 81920,
    "driver_version": "570.133.20",
    "python": "3.12.13",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "gcc": "11.4.0",
    "execution_locus": "REMOTE_A100 (EPIC-05). NOT local GPU. Local machine is Win11/AMD64/Py3.13/torch2.13+cu130/RTX5070-Laptop-8GB/gsplat1.4.0 and lacks C0 identities + 5K/15K checkpoints.",
    "note": "SB0_REPORT.md:49 referenced torch2.9.1+cu128; live probe measured torch2.10.0+cu128 on EPIC-05. Recorded live-measured value.",
}

REQUIRED_CELLS = ["room@5000", "room@15000", "bicycle@5000", "bicycle@15000"]

INVENTORY = {
    "schema_version": "AB0-v1",
    "artifact": "AB0_CHECKPOINT_INVENTORY.json",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "authoritative_host": "EPIC-05",
    "required_cells": REQUIRED_CELLS,
    "prereg_rule": "AB0_PREREG.md section 1: 'If an exact checkpoint does not exist: AB0_CHECKPOINT_BLOCKER. STOP. Do not substitute.' + 'Recover all checkpoint / camera hashes from frozen manifests.'",
    "frozen_c0_publication_arm": {
        "location": "/root/3dgs-publication-epic05/full30k/c0_s4{2,3,4}_{room,bicycle}",
        "seeds": [42, 43, 44],
        "checkpoints_saved": ["ckpt_30000.pt (final only)"],
        "reason": "publication_trainer.py --checkpoints saves only config.iterations (line 658-664: torch.save ckpt_{config.iterations}.pt). No intermediate 5K/15K optimizer states saved.",
        "training_curve_5k_15k_rows": "eval-metric rows only (PSNR/SSIM/LPIPS), NOT saved optimizer states.",
        "has_5k": False,
        "has_15k": False,
    },
    "non_qualifying_checkpoints_found": [
        {
            "path": "/root/epic05_d0/runs/TRACE_{room,bicycle}/d0_ckpt_{5000,15000}.pt",
            "arm": "D0 / ReBudgetGS TRACE",
            "reason_disqualified": "already-densified state (d0_birth_records.npz, 1503506 birth records); camera_sequence differs from c0_s43 (57 12 142... vs 19 147 74...). Not frozen C0.",
        },
        {
            "path": "/root/epic05_n1/runs/n1snap_c0f_s42_{room,bicycle}/ckpt_15000.pt",
            "arm": "c0 (seed 42, 25000 iters)",
            "reason_disqualified": "different seed/iters/camera-order than frozen publication arm; only has 15K, NO 5K (room@5K and bicycle@5K still missing).",
        },
        {
            "path": "/root/epic05_grc3d/inputs/snapshots/snap_{bicycle,garden}_005000.pt",
            "arm": "gmain",
            "reason_disqualified": "arm=gmain, not C0.",
        },
    ],
    "global_search_result": "No pristine frozen-C0 5K or 15K checkpoint exists anywhere on EPIC-05 for room or bicycle.",
    "checkpoint_gate": "FAIL",
}

VERDICT = {
    "schema_version": "AB0-v1",
    "artifact": "AB0_VERDICT.json",
    "verdict": "AB0_CHECKPOINT_BLOCKER",
    "stage": "Stage -1 (pre-execution identity + checkpoint gate)",
    "identity_gate": "PASS",
    "checkpoint_gate": "FAIL",
    "analyzer_run": False,
    "metrics_collected": False,
    "reason": "Exact frozen-C0 5K/15K checkpoints for room/bicycle do not exist on the authoritative host (EPIC-05). Prereg section 1 mandates AB0_CHECKPOINT_BLOCKER and forbids substitution. Remaining 5K/15K checkpoints belong to other arms (D0-TRACE densified / N1-c0f seed42 only-15K / GRC3D arm=gmain) and cannot be substituted.",
    "experiments_run": 0,
    "rollouts_completed": 0,
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "generator_script_sha256": GEN_SHA,
    "next_action_required": "Research-lead decision: (a) authorize re-running frozen C0 training (same seed/recipe) and saving intermediate 5K/15K optimizer states, OR (b) keep AB0 blocked. Re-running is a deviation from the 'exact checkpoint / Do not substitute' hard gate and needs explicit authorization before A100 time is spent.",
}

REPORT = f"""# AB0 REPORT — AB0_CHECKPOINT_BLOCKER (Stage -1)

- Verdict: **AB0_CHECKPOINT_BLOCKER**
- Authoritative host: EPIC-05 (A100-SXM4-80GB, 8.130.30.251:1024, root)
- Execution locus: REMOTE A100. NOT local GPU.
- Identity gate: PASS. Checkpoint gate: FAIL. Analyzer: not run. Metrics: not collected.

## 1. Identity gate (PASS)
- core gsplat_cuda.so: pinned `935ea6c0...ff37e` == found `/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so` (exact SHA256).
- C0 extension `d26cccf...513f11`: built `.so` sha256 == pinned commit exactly (`/root/epic05_build_cache/c0_ext/.../experimental_gaussian_render_inference_scene_cuda.so`). c0_ext has no `.git`; identity rests on built `.so` hash (same method as SB0).
- publication_trainer.py sha256 `1b67171d...`; cameras.json room/bicycle hashes recorded.

## 2. Checkpoint gate (FAIL → blocker)
Required cells: room@5K, room@15K, bicycle@5K, bicycle@15K (frozen C0 semantics).
- Frozen C0 publication arm saved only `ckpt_30000.pt` (trainer `--checkpoints` saves final iteration only, line 658-664). No 5K/15K optimizer states.
- Available 5K/15K checkpoints are other arms and cannot be substituted:
  - D0/ReBudgetGS TRACE (densified state, different camera order).
  - N1 c0f (seed 42, 25000 iters, only 15K, different camera order).
  - GRC3D snapshots (arm=gmain).

## 3. Action
Per prereg section 1 and DSH handoff step 2, AB0 stops at Stage -1. No timing/integration/held-out metrics collected; analyzer not run.

Decision required from research lead: authorize frozen-C0 re-run to persist 5K/15K intermediates (deviation from exact-checkpoint gate), or keep blocked.
"""

for name, obj in [
    ("AB0_SOURCE_IDENTITY.json", IDENTITY),
    ("AB0_ENVIRONMENT.json", ENVIRONMENT),
    ("AB0_CHECKPOINT_INVENTORY.json", INVENTORY),
    ("AB0_VERDICT.json", VERDICT),
]:
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)

(OUT / "AB0_REPORT.md").write_text(REPORT, encoding="utf-8")
print("wrote AB0_REPORT.md")
print("generator sha256:", GEN_SHA)
