#!/usr/bin/env python3
"""P13-Y: generate publication_trainer_p13y_capture.py from the frozen P12
trainer by STRICTLY ADDITIVE textual insertions, then emit the unified diff.

Guarantees:
  * source is the byte copy of publication_trainer_p12.py (sha256 asserted);
  * every insertion is anchored to an exact unique context block; the script
    fails loudly if any anchor is missing/ambiguous;
  * only insertions (no deletions/replacements of existing lines).
"""
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "publication_trainer_p12.py")
OUT = os.path.join(HERE, "publication_trainer_p13y_capture.py")
PATCH = os.path.join(HERE, "P13Y_TRAINER_DIFF.patch")
P12_SHA = "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5"

# --------------------------------------------------------------------------
HEADER_NOTE = '''P13-Y CAPTURE COPY (2026-09-27):
  Byte copy of /root/epic05_p12/publication_trainer_p12.py (sha256
  675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5) plus a
  strictly ADDITIVE P13-Y capture side channel (--p13y-* flags; see the
  "P13-Y capture side channel" block below and P13Y_TRAINER_DIFF.patch).
  With no --p13y-capture-iters / --p13y-parity-iters the training path is
  unchanged: the only additions are no-op set-membership tests at the top and
  bottom of the loop body and optional side-channel file writes (raw-state
  snapshots, camera log, per-iteration loss log, source identity). The side
  channel consumes no RNG, never mutates model/optimizer state, and does all
  disk I/O outside the phase-timing windows.

'''

FLAGS = '''    # ---------------- P13-Y capture flags (side channel; default off) -------
    ap.add_argument("--p13y-capture-dir", default=None,
                    help="save raw model state at --p13y-capture-iters below this dir")
    ap.add_argument("--p13y-capture-iters", default="",
                    help="comma-separated iterations at which to save raw model state")
    ap.add_argument("--p13y-parity-iters", default="",
                    help="comma-separated iterations at which to dump a parity probe "
                         "(trainer frame + loss + camera fingerprint; the raw state is "
                         "written by the same path as --p13y-capture-iters)")
    ap.add_argument("--p13y-parity-dir", default=None,
                    help="probe output dir (default: --p13y-capture-dir)")
    ap.add_argument("--p13y-cam-log", default=None,
                    help="write per-iteration {iter, cam_idx, sh_degree} JSON here")
    ap.add_argument("--p13y-iter-log", default=None,
                    help="write per-iteration {iter, loss, n_gs} JSON here")
    ap.add_argument("--p13y-source-identity", default=None,
                    help="write the P13-Y capture source identity JSON here")

'''

SIDE_CHANNEL = '''
    # ==================================================== P13-Y capture side channel
    # Additive only (P13Y_TRAINER_DIFF.patch): consumes no RNG, never mutates
    # model/optimizer state, all disk I/O outside the phase-timing windows.
    P13Y_P12_TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
    P13Y_P12_TRAINER_SHA256 = "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5"
    P13Y_C0_EXT_SHA256 = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
    P13Y_C0_CORE_SHA256 = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
    P13Y_FUSED_SO_SHA256 = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"

    p13y_capture_iters = ({int(x) for x in args.p13y_capture_iters.split(",") if x.strip()}
                          if args.p13y_capture_iters else set())
    p13y_parity_iters = ({int(x) for x in args.p13y_parity_iters.split(",") if x.strip()}
                         if args.p13y_parity_iters else set())
    p13y_state_iters = p13y_capture_iters | p13y_parity_iters
    p13y_active = bool(p13y_state_iters) or bool(args.p13y_cam_log) or bool(args.p13y_iter_log)
    if p13y_state_iters:
        if args.arm != "c0":
            raise SystemExit("P13-Y capture is implemented for arm=c0 only")
        if args.loss != "fused":
            raise SystemExit("P13-Y capture requires --loss fused")
        if args.p13y_capture_dir is None:
            raise SystemExit("--p13y-capture-dir required with capture/parity iters")
        os.makedirs(args.p13y_capture_dir, exist_ok=True)
    p13y_cam_log = [] if args.p13y_cam_log else None
    p13y_iter_log = [] if args.p13y_iter_log else None
    p13y_probe_pending = None

    def p13y_rng_fingerprint():
        """Hash of every RNG stream the training loop can consume."""
        h = hashlib.sha256()
        h.update(torch.get_rng_state().numpy().tobytes())
        if torch.cuda.is_available():
            h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
        st = np.random.get_state()
        h.update(st[0].encode())
        h.update(np.asarray(st[1]).tobytes())
        h.update(str(st[2:]).encode())
        h.update(repr(random.getstate()).encode())
        return h.hexdigest()

    def p13y_raw_tensors():
        """Raw optimization variables exactly as stored (unactivated)."""
        return [("xyz", model._xyz), ("scaling", model._scaling),
                ("rotation", model._rotation), ("opacity", model._opacity),
                ("shs", model._shs)]

    def p13y_state_fingerprint():
        h = hashlib.sha256()
        for name, t in p13y_raw_tensors():
            v = t.detach().contiguous().cpu().numpy()
            h.update(name.encode())
            h.update(str(v.dtype).encode())
            h.update(np.asarray(v.shape, dtype=np.int64).tobytes())
            h.update(v.tobytes())
        return h.hexdigest()

    def p13y_bytes_fingerprint(u):
        h = hashlib.sha256()
        h.update(str(u.dtype).encode())
        h.update(np.asarray(u.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(u).tobytes())
        return h.hexdigest()

    def p13y_camera_fingerprint(cam):
        m = (cam.viewmatrix.detach().cpu().numpy().tobytes()
             + cam.K.detach().cpu().numpy().tobytes()
             + f"{int(cam.image_width)}x{int(cam.image_height)}".encode())
        return hashlib.sha256(m).hexdigest()

    def p13y_save_state(iteration, out_dir, tag):
        """Save raw optimization variables (values as stored, no activations)."""
        os.makedirs(out_dir, exist_ok=True)
        st_torch, st_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        st_np, st_py = np.random.get_state(), random.getstate()
        h_before = p13y_rng_fingerprint()
        vers_before = [int(t._version) for _, t in p13y_raw_tensors()]
        t_d = time.perf_counter()
        raw = {name: t.detach().contiguous().cpu()
               for name, t in p13y_raw_tensors()}
        torch.save(raw, os.path.join(out_dir, "state.pt"))
        per_t = {name: p13y_bytes_fingerprint(t.numpy()) for name, t in raw.items()}
        sha_state = p13y_state_fingerprint()
        torch.set_rng_state(st_torch)
        torch.cuda.set_rng_state(st_cuda)
        np.random.set_state(st_np)
        random.setstate(st_py)
        h_after = p13y_rng_fingerprint()
        vers_after = [int(t._version) for _, t in p13y_raw_tensors()]
        meta = {
            "scene": args.scene, "arm": args.arm, "iteration": int(iteration),
            "tag": tag,
            "capture_point": ("top of the iteration body, after the sh-degree "
                              "progression step and before this iteration's camera "
                              "is loaded (= the state consumed by this iteration's "
                              "forward render)"),
            "n_gaussians": int(model._xyz.shape[0]),
            "active_sh_degree": int(model.active_sh_degree),
            "cam_idx_of_this_iteration": int(camera_sequence[int(iteration) - 1]),
            "optimizer_xyz_lr": float(model.optimizer.param_groups[0]["lr"]),
            "state_pt": os.path.join(out_dir, "state.pt"),
            "state_sha256": sha_state,
            "per_tensor_sha256": per_t,
            "tensor_names": [n for n, _ in p13y_raw_tensors()],
            "tensor_shapes": {n: list(t.shape) for n, t in p13y_raw_tensors()},
            "tensor_dtypes": {n: str(t.dtype) for n, t in p13y_raw_tensors()},
            "param_versions_before": vers_before,
            "param_versions_after": vers_after,
            "model_params_unchanged": bool(vers_before == vers_after),
            "rng_fingerprint_before": h_before,
            "rng_fingerprint_after_restore": h_after,
            "rng_restored_exactly": bool(h_before == h_after),
            "wall_ms": (time.perf_counter() - t_d) * 1000.0,
        }
        assert meta["model_params_unchanged"], "P13-Y capture mutated model params"
        assert meta["rng_restored_exactly"], "P13-Y capture perturbed an RNG stream"
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    def p13y_write_probe(iteration, state_meta, frame_t, loss_t, cam, sh_degree):
        """Parity probe: trainer frame + loss + camera fingerprint (+ state ref)."""
        out_dir = os.path.join(args.p13y_parity_dir or args.p13y_capture_dir,
                               f"probe_i{iteration:05d}")
        os.makedirs(out_dir, exist_ok=True)
        frame_cpu = frame_t.detach().float().cpu().contiguous()
        loss_val = float(loss_t.item())
        torch.save({"frame": frame_cpu}, os.path.join(out_dir, "trainer_frame.pt"))
        rec = {
            "scene": args.scene, "arm": args.arm, "iteration": int(iteration),
            "probe_point": ("same iteration as the state capture; frame/loss are the "
                            "training loop's own outputs for this iteration"),
            "trainer_frame_pt": os.path.join(out_dir, "trainer_frame.pt"),
            "trainer_frame_sha256": p13y_bytes_fingerprint(frame_cpu.numpy()),
            "trainer_frame_shape": list(frame_cpu.shape),
            "trainer_frame_dtype": str(frame_cpu.dtype),
            "trainer_loss": loss_val,
            "cam_idx": int(camera_sequence[int(iteration) - 1]),
            "camera_sha256": p13y_camera_fingerprint(cam),
            "sh_degree": int(sh_degree),
            "n_gaussians": int(model._xyz.shape[0]),
            "state_dir": state_meta["state_pt"] if state_meta else None,
            "state_sha256": state_meta["state_sha256"] if state_meta else None,
            "capture_meta": state_meta,
        }
        with open(os.path.join(out_dir, "probe_meta.json"), "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  [P13-Y probe @{iteration}] frame_sha={rec['trainer_frame_sha256'][:16]} "
              f"loss={loss_val:.8f} cam={rec['cam_idx']} -> {out_dir}", flush=True)

    if args.p13y_source_identity or p13y_active:
        _src_sha = sha256_file(os.path.abspath(__file__))
        ident_rec = {
            "phase": "P13-Y capture side channel", "date": "2026-09-27",
            "p12_trainer": {"path": P13Y_P12_TRAINER,
                            "sha256": sha256_file(P13Y_P12_TRAINER)
                            if os.path.exists(P13Y_P12_TRAINER) else None,
                            "expected_sha256": P13Y_P12_TRAINER_SHA256},
            "capture_trainer": {"path": os.path.abspath(__file__), "sha256": _src_sha},
            "c0_extension": {"path": C0_SO, "sha256": ident.get("so_sha256"),
                             "expected_sha256": P13Y_C0_EXT_SHA256},
            "c0_core": {"path": CORE_SO, "sha256": ident.get("core_sha256"),
                        "expected_sha256": P13Y_C0_CORE_SHA256},
            "fused_ssim": {"so_path": loss_backend.get("so_path"),
                           "so_sha256": loss_backend.get("so_sha256"),
                           "expected_sha256": P13Y_FUSED_SO_SHA256},
            "gsplat_file": __import__("gsplat").__file__,
            "c0_worktree": C0_WT,
            "recipe_identity": {
                "arm": args.arm, "loss": args.loss, "seed": config.seed,
                "resolution": "1080p", "eps2d": eps2d_eff,
                "scene": args.scene, "iterations": config.iterations,
                "n_cameras": int(n_cameras),
            },
            "capture_config": {
                "capture_iters": sorted(p13y_capture_iters),
                "parity_iters": sorted(p13y_parity_iters),
                "capture_dir": args.p13y_capture_dir,
                "cam_log": args.p13y_cam_log, "iter_log": args.p13y_iter_log,
            },
            "higs_env": {k: v for k, v in os.environ.items() if k.startswith("HIGS_")},
        }
        ident_rec["p12_trainer"]["match"] = bool(
            ident_rec["p12_trainer"]["sha256"] == P13Y_P12_TRAINER_SHA256)
        ident_rec["c0_extension"]["match"] = bool(
            ident_rec["c0_extension"]["sha256"] == P13Y_C0_EXT_SHA256)
        ident_rec["c0_core"]["match"] = bool(
            ident_rec["c0_core"]["sha256"] == P13Y_C0_CORE_SHA256)
        assert ident_rec["p12_trainer"]["match"], "P12 trainer source drifted"
        assert ident_rec["c0_extension"]["match"], "C0 extension hash mismatch"
        assert ident_rec["c0_core"]["match"], "C0 core hash mismatch"
        if args.loss == "fused":
            assert ident_rec["fused_ssim"]["so_sha256"] == P13Y_FUSED_SO_SHA256, \\
                "fused SSIM hash mismatch"
        if args.p13y_source_identity:
            with open(args.p13y_source_identity, "w") as f:
                json.dump(ident_rec, f, indent=2)
            print(f"  P13-Y source identity -> {args.p13y_source_identity}", flush=True)
'''

LOOP_TOP = '''        if p13y_state_iters and iteration in p13y_state_iters:
            p13y_state_meta = p13y_save_state(
                iteration,
                os.path.join(args.p13y_capture_dir, f"i{iteration:05d}"),
                tag="parity" if iteration in p13y_parity_iters else "capture")
        else:
            p13y_state_meta = None
        if p13y_parity_iters and iteration in p13y_parity_iters:
            p13y_probe_pending = (iteration, p13y_state_meta)

'''

CAM_LOG = '''        if p13y_cam_log is not None:
            p13y_cam_log.append([int(iteration), int(cam_idx),
                                 int(model.active_sh_degree)])

'''

PROBE_STASH = '''        if p13y_probe_pending is not None:
            p13y_probe_frame = image
            p13y_probe_loss = loss
            p13y_probe_cam = cam
            p13y_probe_shd = int(model.active_sh_degree)

'''

LOOP_BOTTOM = '''        if p13y_iter_log is not None:
            p13y_iter_log.append([int(iteration), float(loss.item()),
                                  int(model._xyz.shape[0])])
        if p13y_probe_pending is not None:
            p13y_write_probe(p13y_probe_pending[0], p13y_probe_pending[1],
                             p13y_probe_frame, p13y_probe_loss, p13y_probe_cam,
                             p13y_probe_shd)
            p13y_probe_pending = None

'''

FINAL_DUMPS = '''    if p13y_cam_log is not None:
        _align_ok = all(r[0] == i + 1 and r[1] == int(camera_sequence[i])
                        for i, r in enumerate(p13y_cam_log))
        with open(args.p13y_cam_log, "w") as f:
            json.dump({
                "scene": args.scene, "arm": args.arm, "seed": config.seed,
                "iterations": int(config.iterations),
                "camera_sequence_sha256": sha256_file(
                    os.path.join(args.outdir, "camera_sequence.npy")),
                "alignment_rule": ("cam_log[i][0] == i+1 and cam_log[i][1] == "
                                   "camera_sequence[i] for every logged iteration"),
                "alignment_holds": bool(_align_ok),
                "cam_log": p13y_cam_log,
            }, f)
        assert _align_ok, "P13-Y cam log misaligned with camera_sequence.npy"
        print(f"  P13-Y camera log ({len(p13y_cam_log)} iters, aligned={_align_ok}) "
              f"-> {args.p13y_cam_log}")
    if p13y_iter_log is not None:
        with open(args.p13y_iter_log, "w") as f:
            json.dump({
                "scene": args.scene, "arm": args.arm, "seed": config.seed,
                "iterations": config.iterations,
                "camera_sequence_sha256": sha256_file(
                    os.path.join(args.outdir, "camera_sequence.npy")),
                "final_model_param_sha256": p13y_state_fingerprint(),
                "final_N": int(model._xyz.shape[0]),
                "iter_log": p13y_iter_log,
            }, f)
        print(f"  P13-Y iter log -> {args.p13y_iter_log}")

'''

# (anchor, position, text, expected_count)
OPS = [
    ('"""PUBLICATION unified trainer: reference_v1 frozen recipe, multi-arm.\n\n',
     "after", HEADER_NOTE, 1),
    ("    args = ap.parse_args()\n", "before", FLAGS, 1),
    ("    eval_indices = list(range(0, n_cameras, max(1, n_cameras // 10)))\n"
     "    eval_iterations = set(i for i in config.eval_iterations if i <= args.iterations)\n",
     "after", SIDE_CHANNEL, 1),
    ("        cam_idx = int(camera_sequence[iteration - 1])\n"
     "        cam, gt_image = dataset.get_item(cam_idx)\n",
     "before", LOOP_TOP, 1),
    ("        cam_idx = int(camera_sequence[iteration - 1])\n"
     "        cam, gt_image = dataset.get_item(cam_idx)\n",
     "after", CAM_LOG, 1),
    ("        pev[2].record()\n", "after", PROBE_STASH, 1),
    ("        iter_times.append((time.perf_counter() - t0) * 1000)\n",
     "after", LOOP_BOTTOM, 1),
    ('    print(f"  WROTE {out_path} + final_status.json, timing.json, quality.json, "\n'
     '          f"training_results.json, training_curve.csv")\n',
     "after", FINAL_DUMPS, 1),
]


def main():
    src = open(SRC, encoding="utf-8").read()
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    assert sha == P12_SHA, f"source drift: {sha}"
    out = src
    for anchor, pos, text, expected in OPS:
        n = out.count(anchor)
        if n != expected:
            sys.exit(f"ANCHOR ERROR ({n} != {expected}): {anchor[:70]!r}")
        out = out.replace(anchor, (anchor + text) if pos == "after" else (text + anchor), 1)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    diff = "".join(difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="publication_trainer_p12.py", tofile="publication_trainer_p13y_capture.py"))
    with open(PATCH, "w", encoding="utf-8", newline="\n") as f:
        f.write(diff)
    n_add = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    n_del = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    print(f"WROTE {OUT}\n  src_sha256={sha}\n  dst_sha256="
          f"{hashlib.sha256(out.encode('utf-8')).hexdigest()}\n"
          f"  diff: +{n_add} -{n_del} lines (additive-only: {n_del == 0})")
    assert n_del == 0, "non-additive edit detected"
    sys.path.insert(0, HERE)
    import ast
    ast.parse(out)
    print("AST parse OK")


if __name__ == "__main__":
    main()
