"""P15-P0 Step 3 - one-cell EdgeBudget export (NPZ + summary JSON).

The export is correctness-only: it reuses the certified P14 code paths in
place and writes (i) the progress proxy S_i and (ii) the exact per-Gaussian
intersection cost c_i at the frozen 16K initial state, after passing the
P14 reproduction gate. No training, no CUDA changes, no optimizer steps.

FROZEN definitions (preregistered before any P15-P0 result):

  progress proxy : P14 selector energy S_i (float64 [N]) - exact reuse of
                   p14_selector.selector_for_cell(ctx, cluster_cameras);
                   the certified P14 scripts are never edited.
  cost           : c_i = exact Gaussian-tile intersection count at the 16K
                   initial state, summed over the cell's 8 cluster cameras,
                   produced by the exact C0 stream path (cull -> gatherless
                   projected producer -> intersect_tile) via
                   p14_common.SceneCtx.stream_pieces. Frozen settings:
                   eps2d=0.3, tile=16, pinhole, 1080p, radius_clip=0.0.
  cost certificate: the cost pass runs twice and both c arrays must be
                   bit-identical (sha256), else P0_EXPORT_INVALID.

  P14 reproduction gate, per q in {0.10,0.20,0.25,0.30,0.50}:
    G1  n_active_new == n_active_p14                  (exact)
    G2  |E_new(q) - E_p14(q)| <= 5e-4                 (absolute)
    G3  Jaccard(active_new, active_p14) >= 0.95
    G4  symmetric-difference energy share <= 1e-5
  gate_pass = G1..G4 hold for every q; any failing cell stops the queue
  with P0_EXPORT_INVALID.

  GPU contamination: full nvidia-smi inventory (all GPUs + compute apps)
  before and after each cell is recorded in the summary; foreign apps on
  the measurement GPU are flagged (correctness work does not require an
  idle GPU, but the ledger must be complete).

Exit codes: 0 valid, 2 gate/certificate invalid, 3 unexpected error.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

Q5 = (0.10, 0.20, 0.25, 0.30, 0.50)
GATE = {"dE_abs": 5.0e-4, "jaccard": 0.95, "symdiff_energy_share": 1.0e-5}


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def jwrite(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[p15] wrote {path}", flush=True)


def _run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"ERR:{e}"


def gpu_inventory(measure_gpu, my_pid):
    snap = _run(["nvidia-smi",
                 "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                 "--format=csv,noheader"])
    idx_uuid = {}
    for line in snap.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            idx_uuid[parts[0]] = parts[1]
    uuid_idx = {v: k for k, v in idx_uuid.items()}
    apps_raw = _run(["nvidia-smi",
                     "--query-compute-apps=gpu_uuid,pid,process_name,"
                     "used_memory", "--format=csv,noheader"])
    apps = []
    for line in apps_raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        apps.append({"gpu_index": uuid_idx.get(parts[0], "?"),
                     "pid": parts[1], "process_name": parts[2],
                     "used_memory": parts[3]})
    my_pid_s = str(my_pid)
    foreign = [a for a in apps
               if a["gpu_index"] == str(measure_gpu)
               and a["pid"] != my_pid_s]
    return {"snapshot": snap, "apps": apps, "my_pid": my_pid_s,
            "measure_gpu": str(measure_gpu),
            "foreign_apps_on_measure_gpu": foreign,
            "n_foreign_on_measure_gpu": len(foreign)}


def cost_pass(ctx, cams, N):
    """One exact cost pass: c_i = sum over cameras of per-Gaussian
    intersection-entry counts from the exact C0 stream path."""
    import numpy as np
    c = np.zeros(N, dtype=np.int64)
    per_cam = []
    for ci in cams:
        cam, _ = ctx.cam_gt(int(ci))
        pieces = ctx.stream_pieces(cam)
        if pieces.get("empty"):
            per_cam.append({"camera": int(ci), "n_isects": 0,
                            "n_visible": 0})
            continue
        mf = pieces["master_flat"]
        c += np.bincount(mf, minlength=N).astype(np.int64)
        per_cam.append({"camera": int(ci),
                        "n_isects": int(pieces["n_isects"]),
                        "n_visible": int(pieces["n_visible"])})
    return c, per_cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p14-scripts", default="/root/epic05_p14/scripts")
    ap.add_argument("--p14-root", default="/root/epic05_p14")
    ap.add_argument("--p15-root", default="/root/epic05_p15/p0")
    ap.add_argument("--scene", required=True, choices=("room", "bicycle"))
    ap.add_argument("--cell", required=True, type=int, choices=(0, 1, 2))
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, args.p14_scripts)
    import numpy as np
    import p14_common as C
    import p14_selector as S14

    summary = {
        "phase": "P15-P0", "step": "export",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scene": args.scene, "cell": args.cell, "gpu_str": str(args.gpu),
        "p14_root": args.p14_root, "p14_scripts": args.p14_scripts,
        "p15_root": args.p15_root,
        "frozen": {"q_grid": list(Q5), "gate": GATE,
                   "cost_definition": ("exact Gaussian-tile intersection "
                                       "count, 8 cluster cameras, 16K "
                                       "initial state, cull->producer->"
                                       "intersect_tile"),
                   "progress_proxy": "P14 selector energy S_i"},
        "verdict": None,
    }
    my_pid = os.getpid()
    t_start = time.time()
    inv_before = gpu_inventory(args.gpu, my_pid)
    summary["gpu"] = {"before": inv_before}
    print(f"[p15:{args.scene} c{args.cell}] GPU inventory before:\n"
          f"{inv_before['snapshot']}\nforeign-on-measure="
          f"{inv_before['n_foreign_on_measure_gpu']}", flush=True)

    try:
        # ---- P14 source records --------------------------------------
        clusters = json.load(open(os.path.join(
            args.p14_root, "artifacts", "P14_CAMERA_CLUSTERS.json")))
        cellrec = clusters["scenes"][args.scene]["cells"][args.cell]
        selector = json.load(open(os.path.join(
            args.p14_root, "artifacts", "P14_SELECTOR.json")))
        screc = selector["scenes"][args.scene]
        cell_sel = screc["cells"][args.cell]
        cams = [int(x) for x in cellrec["cluster_cameras"]]
        assert cams == [int(x) for x in cell_sel["cluster_cameras"]], \
            "cluster cameras mismatch P14 selector record"
        npz_path = screc["active_npz"]["path"]
        npz_sha = sha256_file(npz_path)
        assert npz_sha == screc["active_npz"]["sha256"], \
            "P14 active npz hash mismatch"
        npz = np.load(npz_path)

        ckpt = C.CKPT[args.scene]
        p14_ident = json.load(open(os.path.join(
            args.p14_root, "artifacts", "P14_SOURCE_IDENTITY.json")))
        ckpt_expected = p14_ident["checkpoints_16k"][args.scene]["sha256"]
        ckpt_sha = sha256_file(ckpt)
        assert ckpt_sha == ckpt_expected, "16K checkpoint hash mismatch"
        summary["source_chain"] = {
            "ckpt": {"path": ckpt, "sha256": ckpt_sha,
                     "matches_p14_identity": True},
            "active_npz": {"path": npz_path, "sha256": npz_sha,
                           "matches_p14_record": True},
            "clusters_json_sha256": sha256_file(os.path.join(
                args.p14_root, "artifacts", "P14_CAMERA_CLUSTERS.json")),
            "selector_json_sha256": sha256_file(os.path.join(
                args.p14_root, "artifacts", "P14_SELECTOR.json")),
            "p14_common_sha256": sha256_file(os.path.join(
                args.p14_scripts, "p14_common.py")),
            "p14_selector_sha256": sha256_file(os.path.join(
                args.p14_scripts, "p14_selector.py")),
        }

        # ---- P14 anchors (H=32 Stage A) ------------------------------
        anchors = {}
        for q in Q5:
            qq = int(round(q * 100))
            fd = json.load(open(os.path.join(
                args.p14_root, "P14_O1_UNITS",
                f"{args.scene}_cell{args.cell}_f_q{qq:02d}_h32.json")))
            dd = json.load(open(os.path.join(
                args.p14_root, "P14_O1_UNITS",
                f"{args.scene}_cell{args.cell}_d1_h32.json")))
            g_f = float(fd["G_train"])
            g_d = float(dd["G_train"])
            anchors[str(q)] = {
                "R_train": (g_f / g_d) if g_d > 0 else None,
                "G_train_F": g_f, "G_train_D1": g_d,
                "E_ref_p14": float(cell_sel["E_q"][str(q)]),
                "n_active_p14": int(cell_sel["active_counts"][str(q)]),
            }
        summary["p14_anchor"] = anchors

        # ---- scene context --------------------------------------------
        t0 = time.time()
        ctx = C.SceneCtx(args.scene, gpu=str(args.gpu))
        ctx.restore_initial()
        t_ctx = time.time() - t0
        N = int(ctx.N)
        summary["N"] = N
        summary["it0"] = int(ctx.it0)
        summary["seed_camera"] = int(cellrec["seed_camera"])
        summary["cluster_cameras"] = cams
        summary["audit_cameras"] = [int(x)
                                    for x in cellrec["audit_cameras"]]

        # ---- selector (progress proxy) + gate -------------------------
        t0 = time.time()
        Sarr, order, cam_recs, e_curve, total = S14.selector_for_cell(
            ctx, cams)
        t_sel = time.time() - t0

        per_q = {}
        gate_pass = True
        for q in Q5:
            K = S14.k_of_q(q, N)
            act_new = np.sort(order[:K]).astype(np.int64)
            act_p14 = npz[f"cell{args.cell}_q{int(round(q*100)):02d}"] \
                .astype(np.int64)
            inter = np.intersect1d(act_new, act_p14, assume_unique=False)
            union = np.union1d(act_new, act_p14)
            jac = float(inter.size) / float(union.size) if union.size else 1.0
            sym = np.setxor1d(act_new, act_p14, assume_unique=False)
            sym_energy = float(Sarr[sym].sum()) / float(Sarr.sum())
            E_new = float(e_curve[str(q)])
            E_p14 = float(cell_sel["E_q"][str(q)])
            dE = E_new - E_p14
            ok = (int(act_new.size) == int(act_p14.size)
                  and abs(dE) <= GATE["dE_abs"]
                  and jac >= GATE["jaccard"]
                  and sym_energy <= GATE["symdiff_energy_share"])
            per_q[str(q)] = {
                "K": int(K), "n_act_new": int(act_new.size),
                "n_act_p14": int(act_p14.size),
                "jaccard": jac, "n_symdiff": int(sym.size),
                "symdiff_energy_share": sym_energy,
                "E_new": E_new, "E_p14": E_p14, "dE": dE,
                "active_sha256_new": sha256_bytes(act_new.tobytes()),
                "active_sha256_p14_record":
                    cell_sel.get("active_sha256", {}).get(str(q)),
                "pass": bool(ok),
            }
            gate_pass = gate_pass and ok
        summary["gate"] = {"rules": GATE, "per_q": per_q,
                           "gate_pass": bool(gate_pass),
                           "S_total_new": float(total),
                           "S_total_p14": float(cell_sel["S_total"]),
                           "S_sha256_new": sha256_bytes(Sarr.tobytes()),
                           "S_sha256_p14": cell_sel.get("S_sha256")}
        for q in Q5:
            r = per_q[str(q)]
            print(f"[p15:{args.scene} c{args.cell}] gate q={q}: "
                  f"n_act {r['n_act_new']}/{r['n_act_p14']} "
                  f"jac={r['jaccard']:.6f} dE={r['dE']:+.2e} "
                  f"symE={r['symdiff_energy_share']:.2e} "
                  f"pass={r['pass']}", flush=True)
        summary["timings_s"] = {"ctx_init": t_ctx, "selector": t_sel,
                                "selector_per_cam_losses": [
                                    float(r["loss"]) for r in cam_recs]}

        # ---- exact cost (two passes, bitwise certificate) --------------
        t0 = time.time()
        c1, per_cam1 = cost_pass(ctx, cams, N)
        t_c1 = time.time() - t0
        t0 = time.time()
        c2, per_cam2 = cost_pass(ctx, cams, N)
        t_c2 = time.time() - t0
        repro_ok = bool(np.array_equal(c1, c2)
                        and per_cam1 == per_cam2)
        Ctot = int(c1.sum())
        summary["cost"] = {
            "definition": ("exact Gaussian-tile intersection entries summed "
                           "over the 8 cluster cameras at the 16K initial "
                           "state"),
            "n_isects_total": Ctot,
            "per_camera": per_cam1,
            "n_gaussians_zero_cost": int((c1 == 0).sum()),
            "n_gaussians_zero_energy": int((Sarr <= 0).sum()),
            "c_sha256_pass1": sha256_bytes(c1.tobytes()),
            "c_sha256_pass2": sha256_bytes(c2.tobytes()),
            "repro_ok": repro_ok,
            "gpu_seconds": {"pass1": t_c1, "pass2": t_c2},
        }
        summary["timings_s"]["cost1"] = t_c1
        summary["timings_s"]["cost2"] = t_c2

        # ---- P14 reference points in P15 cost space -------------------
        order_S = np.argsort(-Sarr, kind="stable")
        Stot = float(Sarr.sum())
        q_ref = {}
        for q in Q5:
            K = S14.k_of_q(q, N)
            sel = order_S[:K]
            q_ref[str(q)] = {
                "K": int(K),
                "C_ref": float(c1[sel].sum()) / float(Ctot) if Ctot else None,
                "E_ref": float(Sarr[sel].sum()) / Stot,
            }
        summary["q_ref_points"] = q_ref

        # ---- write NPZ -------------------------------------------------
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez_compressed(
            args.out,
            S=Sarr.astype(np.float64), c=c1.astype(np.int64),
            N=np.int64(N), cell=np.int64(args.cell),
            q_grid=np.array(Q5, dtype=np.float64),
            E_ref_q=np.array([q_ref[str(q)]["E_ref"] for q in Q5],
                             dtype=np.float64),
            C_ref_q=np.array([q_ref[str(q)]["C_ref"] for q in Q5],
                             dtype=np.float64))
        npz_out_sha = sha256_file(args.out)
        summary["npz"] = {"path": args.out, "sha256": npz_out_sha,
                          "bytes": os.path.getsize(args.out),
                          "keys": ["S", "c", "N", "cell", "q_grid",
                                   "E_ref_q", "C_ref_q"]}

        inv_after = gpu_inventory(args.gpu, my_pid)
        summary["gpu"]["after"] = inv_after
        summary["timings_s"]["total"] = time.time() - t_start
        valid = bool(gate_pass and repro_ok)
        summary["verdict"] = ("P0_EXPORT_VALID" if valid
                              else "P0_EXPORT_INVALID")
        summary["stop_reason"] = (None if valid else
                                  ("gate_failed" if not gate_pass
                                   else "cost_not_reproducible"))
        jwrite(args.summary, summary)
        print(f"[p15:{args.scene} c{args.cell}] verdict={summary['verdict']} "
              f"n_isects={Ctot:,} npz={summary['npz']['bytes']/1e6:.1f}MB "
              f"sha={npz_out_sha[:16]} total={summary['timings_s']['total']:.0f}s",
              flush=True)
        if inv_after["n_foreign_on_measure_gpu"]:
            print(f"[p15-warn] foreign apps on measurement GPU after: "
                  f"{inv_after['foreign_apps_on_measure_gpu']}", flush=True)
        sys.exit(0 if valid else 2)

    except SystemExit:
        raise
    except Exception:
        summary["verdict"] = "P0_EXPORT_ERROR"
        summary["error"] = traceback.format_exc()
        try:
            jwrite(args.summary, summary)
        except Exception:
            pass
        print(f"[p15:{args.scene} c{args.cell}] EXPORT ERROR\n"
              f"{summary['error']}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
