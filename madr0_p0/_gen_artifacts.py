import json, datetime, hashlib, os

UTC = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
OUT = "madr0_p0"
os.makedirs(OUT, exist_ok=True)

# ---- MADR0_ENVIRONMENT.json (A100 dependency-check environment; no MADR0 kernels compiled yet) ----
env = {
    "artifact": "MADR0_ENVIRONMENT",
    "generated_utc": UTC,
    "host": "EPIC-05",
    "host_ip": "8.130.30.251",
    "gpu_count": 8,
    "gpus": [{"index": i, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920} for i in range(8)],
    "cross_a100_gate_gpus": [0, 1, 2, 3],
    "driver_version": "570.133.20",
    "cuda_runtime": "12.8",
    "torch": "2.9.1+cu128",
    "cudnn": "91002",
    "device_capability": "8.0",
    "device_name": "NVIDIA A100-SXM4-80GB",
    "nvcc": "12.8  (/usr/local/cuda-12.8/bin/nvcc)",
    "gcc": "11.4.0",
    "python": "3.10.21",
    "cub_version": "2.7.0",
    "cccl_version": "2.7.0",
    "binary_pins": {
        "c0_extension_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
        "core_gsplat_cuda_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
        "fused_ssim_sha256": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
    },
    "execution": "EPIC-05 A100 only; no local GPU",
    "note": "Dependency-check environment. MADR0 kernels were NOT compiled: run stopped at the DPC0 dependency gate before any MADR0 build."
}
json.dump(env, open(f"{OUT}/MADR0_ENVIRONMENT.json", "w"), indent=2)

# ---- MADR0_SOURCE_IDENTITY.json (all checkpoint + binary hashes verified matching) ----
ident = {
    "artifact": "MADR0_SOURCE_IDENTITY",
    "generated_utc": UTC,
    "host": "EPIC-05",
    "authority": "N1 canonical trajectory + HM0 frozen C0 + DPC0 packet (per MADR0 AUTHORITATIVE_ASSETS.json)",
    "checkpoint_hashes": {
        "room@10000": {"path": "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_10000.pt",
            "sha256": "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4",
            "expected": "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4", "match": True},
        "room@15000": {"path": "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_15000.pt",
            "sha256": "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
            "expected": "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd", "match": True},
        "bicycle@10000": {"path": "/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_10000.pt",
            "sha256": "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9",
            "expected": "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9", "match": True},
        "bicycle@15000": {"path": "/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_15000.pt",
            "sha256": "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
            "expected": "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7", "match": True}
    },
    "binary_hashes": {
        "c0_extension": {"sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
            "expected": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11", "match": True},
        "core_gsplat_cuda": {"sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
            "expected": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e", "match": True},
        "fused_ssim": {"sha256": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
            "expected": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb", "match": True}
    },
    "all_match": True,
    "source_identity_pass": True
}
json.dump(ident, open(f"{OUT}/MADR0_SOURCE_IDENTITY.json", "w"), indent=2)

# ---- MADR0_DPC_BASE.json (the dependency gate evidence) ----
dpc_base = {
    "artifact": "MADR0_DPC_BASE",
    "generated_utc": UTC,
    "host": "EPIC-05",
    "dependency": "DPC0 deterministic + correct primitive-centric backward base",
    "allowed_dpc0_states": ["DPC0_PROMOTE", "DPC0_STRONG_PROMOTE", "DPC0_TOO_SLOW(if correctness+determinism passed and only performance failed)"],
    "remote_dpc0_dir": "/root/epic05_dpc0",
    "dpc0_artifacts_found": {
        "DPC0_ENVIRONMENT.json": "present; pass",
        "DPC0_SOURCE_IDENTITY.json": "present; pass (all_match=true)",
        "DPC0_BUILD.json": "ABSENT",
        "DPC0_DETERMINISM.json": "ABSENT",
        "DPC0_CORRECTNESS.json": "ABSENT",
        "DPC0_DIRECTIONAL.json": "ABSENT",
        "DPC0_ONE_STEP.json": "ABSENT",
        "DPC0_TIMING.json": "ABSENT",
        "DPC0_SUMMARY.json": "ABSENT",
        "DPC0_VERDICT.json": "ABSENT"
    },
    "dpc0_candidate_runs_found": 0,
    "dpc0_candidate_determinism_pass": None,
    "dpc0_gradient_correctness_pass": None,
    "dpc0_verdict": None,
    "c0_baseline_run": {
        "tag": "c0_baseline",
        "file": "/root/epic05_dpc0/runs/c0_baseline_gpu0_room_10000_cam5.json",
        "deterministic": False,
        "n_unique_hashes": 8,
        "n_repeat": 8,
        "note": "Authoritative C0 atomic baseline; nondeterministic by design (pixel-centric float atomicAdd). Diagnostic only per DPC0 prereg 4C. This is NOT a DPC candidate result."
    },
    "dpc_base_semantics_pass": False,
    "block_reason": "No completed DPC0 base exists on EPIC-05. Only the atomic C0 baseline (nondeterministic by design) is present. Zero DPC0 candidate determinism/correctness results and no DPC0 verdict were found locally or on the remote. MADR0 prereq section 1: if the DPC base is incorrect or nondeterministic -> MADR0_BLOCKED_DPC_BASE, STOP.",
    "evidence_level": "fact",
    "confidence": "confirmed"
}
json.dump(dpc_base, open(f"{OUT}/MADR0_DPC_BASE.json", "w"), indent=2)

# ---- MADR0_SUMMARY.json (input to frozen analyzer) ----
summary = {
    "artifact": "MADR0_SUMMARY",
    "generated_utc": UTC,
    "host": "EPIC-05",
    "question": "Given a correct deterministic DPC-style tile-Gaussian partial representation, can the final Gaussian-gradient reduction be made materially faster by multiplicity-adaptive deterministic reduction without losing bitwise reproducibility or gradient correctness?",
    "dpc_base_semantics_pass": False,
    "gate": "dependency",
    "block_reason": dpc_base["block_reason"],
    "cells": {},
    "global": {}
}
json.dump(summary, open(f"{OUT}/MADR0_SUMMARY.json", "w"), indent=2)

print("wrote ENVIRONMENT, SOURCE_IDENTITY, DPC_BASE, SUMMARY")
