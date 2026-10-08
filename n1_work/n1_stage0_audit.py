#!/usr/bin/env python3
"""N1 Stage 0: source/checkpoint audit.

Verifies (DSH_TASK_N1 Stage 0):
- P12 trainer hash; C0 renderer/core hashes; fused-ssim lock;
- 15K/20K/25K checkpoints for room+bicycle exist (plus 10K stale-source);
- optimizer state is restorable EXACTLY (bit-exact, twice, incl. after a
  real optimizer step);
- camera sequence prefix identity vs the canonical p12a run;
- held-out render determinism;
- eval drift vs canonical (informational).

Writes N1_SOURCE_IDENTITY.json and N1_CHECKPOINT_IDENTITY.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import n1_core as C


def hash_tensor(t):
    import hashlib
    h = hashlib.sha256()
    h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def model_state_hashes(model):
    out = {}
    for n in C.PARAM_NAMES:
        out[f"param{n}"] = hash_tensor(getattr(model, n))
    for i, p in enumerate([getattr(model, n) for n in C.PARAM_NAMES]):
        st = model.optimizer.state[p]
        out[f"step{i}"] = int(st["step"].item())
        out[f"exp_avg{i}"] = hash_tensor(st["exp_avg"])
        out[f"exp_avg_sq{i}"] = hash_tensor(st["exp_avg_sq"])
    out["active_sh_degree"] = int(model.active_sh_degree)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "room"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES",
                          os.environ.get("N1_GPU", "0"))
    ident = C.bootstrap_c0()
    fused_loss = C.make_fused_loss()

    from dataset import GTDataset
    from config import ReferenceV1Config
    cfg = ReferenceV1Config()

    src = {
        "campaign": "N1_ReBudgetGS_mechanism_measurement",
        "stage": "0_source_checkpoint_audit",
        "frozen_date": "2026-09-28",
        "p12_trainer": {"path": C.P12_TRAINER,
                        "sha256": C.sha256_file(C.P12_TRAINER)},
        "n1_snapshot_trainer": {"path": C.N1_SNAP_TRAINER,
                                "sha256": C.sha256_file(C.N1_SNAP_TRAINER),
                                "edit": "save-only --snapshot-iterations flag; "
                                        "training semantics unchanged"},
        "c0_ext_so": {"path": ident["so_path"], "sha256": ident["so_sha256"]},
        "core_so": {"path": ident["core_path"] if "core_path" in ident
                    else C.CORE_SO, "sha256": ident["core_sha256"]},
        "fused_ssim_so": {"path": fused_loss.so_path,
                          "sha256": fused_loss.so_sha256,
                          "p11_lock": C.P11_FUSED_SO_SHA256,
                          "lock_ok": fused_loss.so_sha256 == C.P11_FUSED_SO_SHA256},
        "higs_env": ident["env"],
        "reference_v1": {
            "gaussian_model.py": C.sha256_file(os.path.join(C.REFERENCE_V1_DIR, "gaussian_model.py")),
            "config.py": C.sha256_file(os.path.join(C.REFERENCE_V1_DIR, "config.py")),
            "colmap_reader.py": C.sha256_file(os.path.join(C.REFERENCE_V1_DIR, "colmap_reader.py")),
            "phase7_dataset.py": C.sha256_file(os.path.join(C.PHASE7_DIR, "dataset.py")),
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "eps2d": C.EPS2D,
        "lambda_dssim": C.LAMBDA_DSSIM,
        "loss": "fused-ssim padding='same' train=True (P12 strong baseline)",
        "seed": cfg.seed,
        "config": cfg.to_dict(),
    }

    # snapshot-run provenance (as recorded by the trainer itself)
    snap_meta = {}
    for sc in (scene,):
        rj = os.path.join(C.SNAP_RUNS[sc], "results.json")
        r = __import__("json").load(open(rj))
        snap_meta[sc] = {
            "command": open(os.path.join(C.SNAP_RUNS[sc], "command.txt")).read().strip()
            if os.path.exists(os.path.join(C.SNAP_RUNS[sc], "command.txt")) else None,
            "binary_identity": r.get("binary_identity"),
            "loss_backend": {k: v for k, v in r.get("loss_backend", {}).items()
                             if k in ("name", "so_sha256", "padding", "commit")},
            "higs_env_all": r.get("higs_env_all"),
            "iterations": r.get("iterations"),
            "seed": r.get("seed"),
            "initial_N": r.get("initial_N"),
            "final_N": r.get("final_N"),
            "timing_grade": r.get("timing_grade"),
        }
    src["snapshot_runs"] = snap_meta

    # ---- camera sequence prefix identity -------------------------------
    seq_report = {}
    for sc in (scene,):
        new = np.load(os.path.join(C.SNAP_RUNS[sc], "camera_sequence.npy"))
        can = np.load(os.path.join(C.CANONICAL_RUNS[sc], "camera_sequence.npy"))
        n = len(new)
        same = bool((new == can[:n]).all())
        seq_report[sc] = {"n_new": int(n), "n_canonical": int(len(can)),
                          "prefix_identical": same}
    src["camera_sequence_prefix"] = seq_report

    # ---- checkpoints ----------------------------------------------------
    dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                        device="cuda", background="black")
    cams_gts = [dataset.get_item(i) for i in range(min(4, len(dataset)))]

    ck_report = {}
    for it in C.CKPTS:
        path = os.path.join(C.SNAP_RUNS[scene], f"ckpt_{it}.pt")
        rec = {"path": path, "bytes": os.path.getsize(path),
               "sha256": C.sha256_file(path)}
        ck = C.load_checkpoint(scene, it)
        rec["keys"] = sorted(ck.keys())
        rec["iteration"] = int(ck["iteration"])
        rec["N"] = int(ck["num_points"])
        rec["active_sh_degree"] = int(ck["active_sh_degree"])
        rec["scene"] = ck["scene"]
        rec["arm"] = ck["arm"]
        osd = ck["optimizer_state_dict"]
        rec["optimizer_param_groups"] = len(osd["param_groups"])
        rec["optimizer_state_keys"] = sorted(int(k) for k in osd["state"].keys())

        model = C.build_model(ckpt=ck)
        C.scene_reset()  # new tensor objects: release the renderer handle
        master = C.StateMaster(model)

        # structure + step values
        steps, ok_moments = [], True
        for i, p in enumerate([getattr(model, n) for n in C.PARAM_NAMES]):
            st = model.optimizer.state[p]
            steps.append(int(st["step"].item()))
            ok = (st["exp_avg"].shape == p.shape and
                  st["exp_avg_sq"].shape == p.shape)
            ok_moments = ok_moments and ok
        rec["adam_step_values"] = steps
        rec["adam_step_all_equal_iteration"] = all(s == it for s in steps)
        rec["moment_shapes_match_params"] = ok_moments
        rec["exp_avg_finite"] = bool(all(
            torch.isfinite(model.optimizer.state[p]["exp_avg"]).all()
            for p in [getattr(model, n) for n in C.PARAM_NAMES]))

        h0 = model_state_hashes(model)

        # (a) random perturbation -> restore -> bit-exact?
        gen = torch.Generator(device="cuda").manual_seed(1234)
        for n in C.PARAM_NAMES:
            p = getattr(model, n)
            p.data.add_(torch.rand(p.shape, generator=gen, device="cuda") * 0.01)
            st = model.optimizer.state[p]
            st["exp_avg"].add_(torch.rand(st["exp_avg"].shape, generator=gen,
                                          device="cuda"))
            st["step"].add_(1)
        master.restore()
        h1 = model_state_hashes(model)
        rec["restore_after_random_perturbation_bit_exact"] = (h0 == h1)

        # (b) one REAL A1-style action (fwd+loss+bwd+step) -> restore -> bit-exact?
        model.update_learning_rate(it + 1)
        cam, gt = dataset.get_item(0)
        img, radii, means2d = C.render_view(model, cam, model.active_sh_degree)
        loss = fused_loss(img, gt)
        loss.backward()
        assert hasattr(means2d, "absgrad"), "c0: means2d.absgrad missing"
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        h2 = model_state_hashes(model)
        rec["state_changed_by_real_action"] = (h2 != h0)
        master.restore()
        h3 = model_state_hashes(model)
        rec["restore_after_real_action_bit_exact"] = (h3 == h0)

        # (c) held-out render determinism + RESTORE-RENDER INVARIANCE:
        # render, run a real action, restore, render again -> must be bitwise
        # identical (StateMaster copy_ restore is seen live by the handle).
        a = C.heldout_eval(model, cams_gts, fused_loss)
        b = C.heldout_eval(model, cams_gts, fused_loss)
        rec["heldout_render_bit_deterministic"] = (a == b)
        rec["heldout_sample"] = a
        cam, gt = dataset.get_item(0)
        model.update_learning_rate(it + 1)
        img, _, means2d = C.render_view(model, cam, model.active_sh_degree)
        fused_loss(img, gt).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        master.restore()
        c = C.heldout_eval(model, cams_gts, fused_loss)
        rec["restore_render_invariance_bit_exact"] = (a == c)

        # (d) A1 replica determinism probe (3 quick replicas)
        deltas = []
        base = C.heldout_eval(model, cams_gts, fused_loss)["fused"]
        for _ in range(3):
            master.restore()
            model.update_learning_rate(it + 1)
            img, _, means2d = C.render_view(model, cam, model.active_sh_degree)
            loss = fused_loss(img, gt)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            deltas.append(C.heldout_eval(model, cams_gts, fused_loss)["fused"] - base)
        rec["a1_replica_heldout_fused_deltas"] = [float(d) for d in deltas]
        rec["optimizer_state_restorable_exactly"] = (
            rec["restore_after_random_perturbation_bit_exact"] and
            rec["restore_after_real_action_bit_exact"])
        del model, master, ck
        torch.cuda.empty_cache()
        ck_report[str(it)] = rec

    # (e) cross-checkpoint switch invariance: rebuild the FIRST checkpoint
    # from file after all later checkpoints were loaded -> held-out must be
    # bitwise identical to the recorded first-checkpoint sample.
    first_it = C.CKPTS[0]
    ck = C.load_checkpoint(scene, first_it)
    model = C.build_model(ckpt=ck)
    C.scene_reset()
    ho = C.heldout_eval(model, cams_gts, fused_loss)
    ck_report[str(first_it)]["switchback_heldout_bit_exact"] = (
        ho == ck_report[str(first_it)]["heldout_sample"])
    del model, ck
    torch.cuda.empty_cache()

    out = {"source": src, "checkpoints": ck_report}
    C.json_dump(out, os.path.join(C.ARTIFACTS, f"N1_AUDIT_RAW_{scene}.json"))

    # split into the two preregistered artifacts
    C.json_dump(src, os.path.join(C.ARTIFACTS, f"N1_SOURCE_IDENTITY_{scene}.json"))
    C.json_dump({"scene": scene, "checkpoints": ck_report},
                os.path.join(C.ARTIFACTS, f"N1_CHECKPOINT_IDENTITY_{scene}.json"))

    print(f"\n=== STAGE 0 AUDIT {scene} SUMMARY ===")
    for k, v in ck_report.items():
        print(f"ckpt {k}: N={v['N']:,} steps_ok={v['adam_step_all_equal_iteration']} "
              f"moments_ok={v['moment_shapes_match_params']} "
              f"restore_exact={v['optimizer_state_restorable_exactly']} "
              f"render_det={v['heldout_render_bit_deterministic']} "
              f"seq_prefix={src['camera_sequence_prefix'][scene]['prefix_identical']}")
    blocker = [k for k, v in ck_report.items()
               if not v["optimizer_state_restorable_exactly"]]
    print("N1_CHECKPOINT_BLOCKER" if blocker else "STAGE0_PASS")
    print("a1 replica deltas (first ckpt):",
          ck_report[str(C.CKPTS[0])]["a1_replica_heldout_fused_deltas"])


if __name__ == "__main__":
    main()
