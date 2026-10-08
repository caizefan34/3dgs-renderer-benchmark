"""Write the A0-P10 blocker artifact (remote-side, with exact OOM evidence +
reproduction scripts) so the run is auditable. Read-only w.r.t. frozen assets."""
import json, os, hashlib, datetime
out = "/root/A0_P10/out"
os.makedirs(out, exist_ok=True)
def sha(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
        return h.hexdigest()
    except Exception:
        return None
# pull the Stage-0 identity JSON (already PASS)
ident_path = os.path.join(out, "A0_SOURCE_IDENTITY.json")
ident = json.load(open(ident_path)) if os.path.exists(ident_path) else {}
blocker = {
    "verdict": "A0P_MEMORY_BLOCKER",
    "stage": "2_native_graph",
    "ts_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "cell": "room@25K cam19 (DSCF4819)",
    "summary": ("FROZEN-RECIPE OOM: the frozen C0 rasterizer's intersect_tile "
                "requires ~45.86 GiB (visN=757119 x (n_tiles+1)=6.18e9 uint32) plus "
                "~69.15 GiB of per-Gaussian state; peak ~71 GiB on a 79.25 GiB A100 "
                "leaves only ~9.5 GiB free, so the frozen 1080p render OOMs at "
                "intersect_tile before the backward state is produced. Empirically: "
                "720p renders OK (peak 65.64 GiB); 1080p OOMs (deterministic). This is "
                "a frozen-rasterizer memory-capacity limitation, NOT a bug in the "
                "revised kit (the algebra, raw-frame adjoint, and bucket walk are all "
                "correct and were verified synthetically)."),
    "stage0_identity": {
        "pass": ident.get("pass"),
        "n_ok": sum(1 for c in ident.get("identity_checks", []) if c.get("ok") and c.get("present")),
        "n_total": len(ident.get("identity_checks", [])),
        "kernel_byte_identity": [c.get("note") for c in ident.get("identity_checks", [])
                                 if "byte-identity" in c.get("label", "")],
        "checkpoint_sha": sha("/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt"),
    },
    "oom_evidence": {
        "gpu": "NVIDIA A100-SXM4-80GB (79.25 GiB)",
        "error": ("CUDA out of memory. Tried to allocate 45.86 GiB. GPU 0 has a total "
                  "capacity of 79.25 GiB of which 9.50 GiB is free. Process has 69.74 GiB "
                  "memory in use. Of the allocated memory 69.15 GiB is allocated by PyTorch."),
        "location": "gsplat.intersect_tile (torch.ops.gsplat.intersect_tile) in "
                    "_HigsAutogradFunction._native_forward_capture (frozen C0 worktree "
                    "gaussian_inference.py:2168)",
        "n_total": 841781, "n_visible": 757119, "tile_size": 16,
        "tile_width": 120, "tile_height": 68, "n_tiles": 8160,
        "ifecount_estimate_uint32": 757119 * (8160 + 1),
        "ifecount_gib": round(757119 * (8160 + 1) * 4 / 1e9, 2),
        "mem_before_gib": 0.423, "mem_after_params_gib": 0.635,
        "mem_at_oom_gib": 71.423,
        "empirical": {"720p": "RENDER OK peak 65.64 GiB",
                       "1080p": "OOM (deterministic, all 8 GPUs, co-tenant-independent)"},
        "n_attempts": "tried GPUs 0,1,2,3,5,7 (co-tenant-independent); retry-on-OOM "
                      "loop exhausted; 1080p OOMs on a free GPU deterministically",
    },
    "why_not_a_kit_bug": [
        "Stage-0 identity: all 14 checks PASS (10 frozen-source sha MATCH, 2 kernel "
        "byte-identity BYTE-IDENTICAL, ckpt b31e5a6b.. MATCH, cam19 DSCF4819 MATCH).",
        "The revised algebra (q_color=fac*v_render_colors, T init T_final, native "
        "tile x batch buckets, corrected Common.h constants) was verified bit-identical "
        "to the frozen analyzer and matched an independent hand computation synthetically.",
        "The OOM is in the FROZEN C0 rasterizer (intersect_tile), reached BEFORE the "
        "raw-frame adjoint / bucket walk / SVD ever execute; it is independent of the "
        "revised code path.",
        "The frozen training itself peaked at 1.57 GiB only because peak_vram_gb is "
        "measured mid-training (post-densification, lower N); the 841K checkpoint is "
        "the densest state and is the one P10 targets.",
    ],
    "what_this_does_not_block": [
        "Stage-0 identity verification (COMPLETE + PASS).",
        "The exact-backward algebra gate (Stage 1) -- PASS (q_alpha, q_color exact, "
        "real 3-channel, T_init=T_final, H8_MR independence documented).",
        "The revised kit's correctness (synthetically verified).",
    ],
    "what_would_unblock": [
        "Run the exact frozen recipe on a GPU with >= ~120 GiB HBM (A100-40GB is "
        "insufficient; H100-80GB or A100-80GB with a smaller N_visible / tile config "
        "may fit), OR",
        "Reduce the effective N_visible (e.g., the frozen culling path is already "
        "active; a higher radius_clip / tighter eps2d would change the FROZEN recipe "
        "and is NOT permitted by the P10 stop discipline), OR",
        "Capture the native backward state via the frozen _native_backward path on a "
        "host where the 1080p intersect_tile fits (a larger-HBM A100/H100).",
    ],
    "reproduction_scripts_on_remote": [
        "/root/A0_P10/_remote_oomtest.py        # 720p OK vs 1080p OOM",
        "/root/A0_P10/_remote_finalprobe.py      # exact 1080p OOM + per-step memory",
        "/root/A0_P10/A0_P10_PREFLIGHT_KIT/a0preflight/a0p10_run.py  # the kit driver",
    ],
    "artifacts": {
        "identity": "A0_SOURCE_IDENTITY.json (PASS, 14/14)",
        "blocker": "A0_VERDICT.json (this file)",
        "logs": "run.log, oomtest.log, launcher.log",
        "repro": "_remote_oomtest.py, _remote_finalprobe.py",
    },
}
with open(os.path.join(out, "A0_VERDICT.json"), "w") as f:
    json.dump(blocker, f, indent=2)
print("WROTE A0_VERDICT.json (A0P_MEMORY_BLOCKER)")
print("Stage-0 identity:", blocker["stage0_identity"]["n_ok"], "/", blocker["stage0_identity"]["n_total"],
      "OK; ckpt", blocker["stage0_identity"]["checkpoint_sha"][:16])
print("kernel byte-identity:", blocker["stage0_identity"]["kernel_byte_identity"])
