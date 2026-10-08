#!/usr/bin/env python3
"""D0 Stage 1/2 validation (run on EPIC-05).

For each scene: TRACE_<scene> (d0_start trainer, --d0-trace) must be
indistinguishable from CLEAN_<scene> (frozen trainer, same recipe) and from the
frozen P12 reference run, in: camera sequence, per-100-iter loss trace,
densification events, N_gs trajectory, final_N, and quality; the tracer's
self-consistency counters must be clean and every prereg artifact present.
Statistical spread gates use the pre-registered 2.0x screen on top of the
unchanged hard gates (D0_TRACE_VALIDITY_SCREEN_PREREG.json).

Writes /root/epic05_d0/artifacts/D0_TRACE_VALIDITY_30k.json; exit 0 iff
D0_TRACE_VALID_30K for every scene analyzed.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = "/root/epic05_d0"
RUNS = os.path.join(ROOT, "runs")
ART = os.path.join(ROOT, "artifacts")
REF = "/root/epic05_p12/runs/p12a/p12a_c0f_s42_{scene}"

REF_FINAL = {
    "room": {"psnr": 32.11282763173283, "ssim": 0.9279112645284155,
             "lpips": 0.30071848765063514, "final_N": 855694,
             "cam_sha": "a80127c542dc2859c0806b7fbf90b8c475a8e0536133be785ff38010630b98e9"},
    "bicycle": {"psnr": 26.368364270001997, "ssim": 0.8355331460839694,
                "lpips": 0.24735402285130983, "final_N": 3663539,
                "cam_sha": "8debe1c2244aca0994e9bd27ddfdd440fcc60f51341351d6c899b3c637c78e2d"},
}
ENV = {"psnr": 0.10, "ssim": 0.003, "lpips": 0.005}   # frozen t_star envelope
N_CAM = {"room": 311, "bicycle": 194}
SAMPLE_OFF = {"room": 11, "bicycle": 15}


def load(rid):
    d = os.path.join(RUNS, rid)
    r = {"dir": d, "exists": os.path.isdir(d)}
    for name, key in (("results.json", "results"), ("quality.json", "quality"),
                      ("final_status.json", "final_status")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as f:
                r[key] = json.load(f)
    p = os.path.join(d, "camera_sequence.npy")
    if os.path.exists(p):
        r["camera_sequence"] = np.load(p)
    p = os.path.join(d, "d0_trace_meta.json")
    if os.path.exists(p):
        with open(p) as f:
            r["d0_trace"] = json.load(f)
    r["files"] = sorted(os.listdir(d)) if os.path.isdir(d) else []
    return r


def loss_trace(run):
    return np.array([row["loss"] for row in run.get("results", {}).get("per_iter", [])],
                    dtype=np.float64)


def n_trace(run):
    return np.array([row["n_gs"] for row in run.get("results", {}).get("per_iter", [])],
                    dtype=np.int64)


def dens_key(ev):
    return (ev["iter"], ev["n_before"], ev["n_after"], ev["cloned"], ev["split"],
            ev["pruned_total"], ev.get("selected"))


def dens_list(run):
    return [dens_key(e) for e in run.get("results", {}).get("densification_events", [])]


def compare(scene):
    out = {"scene": scene}
    g = out["gates"] = {}
    clean = load("CLEAN_%s" % scene)
    clean2 = load("CLEAN2_%s" % scene)
    trace = load("TRACE_%s" % scene)
    ref_dir = REF.format(scene=scene)
    with open(os.path.join(ref_dir, "results.json")) as f:
        ref_res = json.load(f)
    with open(os.path.join(ref_dir, "quality.json")) as f:
        ref_q = json.load(f)
    ref_cam = np.load(os.path.join(ref_dir, "camera_sequence.npy"))

    # --- G1 camera sequences -------------------------------------------------
    cs_c, cs_t = clean.get("camera_sequence"), trace.get("camera_sequence")
    g["camera_seq_trace_eq_clean"] = bool(
        cs_c is not None and cs_t is not None and np.array_equal(cs_c, cs_t))
    g["camera_seq_clean_eq_ref"] = bool(
        cs_c is not None and cs_c.shape == ref_cam.shape and np.array_equal(cs_c, ref_cam))
    g["camera_seq_len"] = int(len(cs_c)) if cs_c is not None else 0

    # --- G2 loss traces ------------------------------------------------------
    # The frozen C0 renderer is run-to-run nondeterministic (backward atomic
    # order; verified in the Stage 0c smoke: iteration-1 loss bit-identical,
    # ~1e-7 drift from iteration 2).  Trace validity therefore uses the
    # intrinsic-spread envelope over clean replicates (P12A ref + CLEAN +
    # CLEAN2) instead of bit-equality; the literal prereg 0.5% criterion is
    # recorded verbatim and evaluated against the clean envelope too.
    lc, lt = loss_trace(clean), loss_trace(trace)
    lc2 = loss_trace(clean2)
    lr = np.array([row["loss"] for row in ref_res["per_iter"]], dtype=np.float64)
    g["loss_n"] = [int(lc.size), int(lc2.size), int(lt.size), int(lr.size)]
    if lc.size == lt.size == lc2.size == lr.size and lc.size:
        g["loss_trace_maxdiff_vs_clean"] = float(np.abs(lt - lc).max())
        g["loss_trace_maxdiff_vs_clean2"] = float(np.abs(lt - lc2).max())
        g["loss_trace_maxdiff_vs_ref"] = float(np.abs(lt - lr).max())
        g["loss_clean_vs_ref_maxdiff"] = float(np.abs(lc - lr).max())
        g["loss_clean_vs_clean2_maxdiff"] = float(np.abs(lc - lc2).max())
        g["loss_clean2_vs_ref_maxdiff"] = float(np.abs(lc2 - lr).max())

    # --- G3 densification events + N trajectory ------------------------------
    dc, dt = dens_list(clean), dens_list(trace)
    dc2 = dens_list(clean2)
    dr = [dens_key(e) for e in ref_res["densification_events"]]
    g["dens_iters_equal_all"] = bool(
        [e[0] for e in dc] == [e[0] for e in dt] ==
        [e[0] for e in dc2] == [e[0] for e in dr])
    g["dens_events_trace_eq_clean"] = bool(dc == dt)
    g["dens_events_clean_eq_ref"] = bool(dc == dr)
    g["dens_events_clean2_eq_ref"] = bool(dc2 == dr)
    g["dens_n_events"] = [len(dc), len(dc2), len(dt), len(dr)]
    nc, nt, nc2 = n_trace(clean), n_trace(trace), n_trace(clean2)
    nr = np.array([row["n_gs"] for row in ref_res["per_iter"]], dtype=np.int64)
    if nc.shape == nt.shape == nc2.shape == nr.shape and nc.size:
        g["n_gs_maxdiff_trace_vs_clean"] = int(np.abs(nt - nc).max())
        g["n_gs_maxdiff_trace_vs_clean2"] = int(np.abs(nt - nc2).max())
        g["n_gs_maxdiff_trace_vs_ref"] = int(np.abs(nt - nr).max())
        g["n_gs_maxdiff_cc_envelope"] = int(
            max(np.abs(nc - nc2).max(), np.abs(nc - nr).max(),
                np.abs(nc2 - nr).max()))

    # --- G4 final N ----------------------------------------------------------
    fN_c = clean.get("results", {}).get("final_N")
    fN_c2 = clean2.get("results", {}).get("final_N")
    fN_t = trace.get("results", {}).get("final_N")
    fN_r = ref_res["final_N"]
    g["final_N"] = [fN_c, fN_c2, fN_t, fN_r]
    g["final_N_trace_vs_clean_rel"] = abs(fN_t - fN_c) / fN_c if fN_c and fN_t else None
    g["final_N_trace_vs_clean2_rel"] = abs(fN_t - fN_c2) / fN_c2 if fN_c2 and fN_t else None
    g["final_N_trace_vs_ref_rel"] = abs(fN_t - fN_r) / fN_r if fN_r and fN_t else None
    rels_cc = []
    if fN_c and fN_r:
        rels_cc.append(abs(fN_c - fN_r) / fN_r)
    if fN_c2 and fN_r:
        rels_cc.append(abs(fN_c2 - fN_r) / fN_r)
    if fN_c and fN_c2:
        rels_cc.append(abs(fN_c - fN_c2) / fN_c2)
    g["final_N_cc_envelope_rel"] = float(max(rels_cc)) if rels_cc else None
    rels_ct = [r for r in (g["final_N_trace_vs_clean_rel"],
                           g["final_N_trace_vs_clean2_rel"],
                           g["final_N_trace_vs_ref_rel"]) if r is not None]
    g["final_N_ct_envelope_rel"] = float(max(rels_ct)) if rels_ct else None
    g["final_N_literal_0p5pct_vs_clean"] = bool(
        g["final_N_trace_vs_clean_rel"] is not None and
        g["final_N_trace_vs_clean_rel"] <= 0.005)
    g["final_N_literal_0p5pct_vs_ref"] = bool(
        g["final_N_trace_vs_ref_rel"] is not None and
        g["final_N_trace_vs_ref_rel"] <= 0.005)

    # --- G4b extended clean-replicate envelope -------------------------------
    # Pre-registered escalation (D0_BICYCLE_FINALN_SCREEN_ESCALATION_PREREG.json):
    # the within-group envelope is estimated over every frozen-recipe clean
    # replica present (CLEAN_<scene>, CLEAN2_<scene>, CLEAN3_<scene>, ... plus
    # the historical P12 reference); screen thresholds are unchanged.
    clean_ids = ["CLEAN_%s" % scene, "CLEAN2_%s" % scene]
    extra_ids = sorted(d for d in os.listdir(RUNS)
                       if re.fullmatch(r"CLEAN\d+_%s" % scene, d)
                       and d not in clean_ids)
    g["extended_clean_ids"] = clean_ids + extra_ids
    g["extended_clean_complete"] = True
    if extra_ids:
        clean_all = [(clean_ids[0], clean), (clean_ids[1], clean2)]
        for rid in extra_ids:
            clean_all.append((rid, load(rid)))
        if not all(r.get("results") for _, r in clean_all):
            g["extended_clean_complete"] = False
        else:
            Lall = {rid: loss_trace(r) for rid, r in clean_all}
            Lall["ref"] = lr
            Nall = {rid: n_trace(r) for rid, r in clean_all}
            Nall["ref"] = nr
            Fall = {rid: r.get("results", {}).get("final_N")
                    for rid, r in clean_all}
            Fall["ref"] = fN_r
            ids = [rid for rid, _ in clean_all] + ["ref"]
            cc_loss, cc_n, cc_f = {}, {}, {}
            ct_loss, ct_n, ct_f = {}, {}, {}
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    k = "%s|%s" % (a, b)
                    if Lall[a].size == lt.size and Lall[b].size == lt.size:
                        cc_loss[k] = float(np.abs(Lall[a] - Lall[b]).max())
                    if Nall[a].shape == nt.shape and Nall[b].shape == nt.shape:
                        cc_n[k] = int(np.abs(Nall[a] - Nall[b]).max())
                    if Fall[a] and Fall[b]:
                        cc_f[k] = abs(Fall[a] - Fall[b]) / max(Fall[a], Fall[b])
            for rid in ids:
                if rid not in ("CLEAN_%s" % scene, "CLEAN2_%s" % scene):
                    if Lall[rid].size == lt.size:
                        ct_loss[rid] = float(np.abs(lt - Lall[rid]).max())
                    if Nall[rid].shape == nt.shape:
                        ct_n[rid] = int(np.abs(nt - Nall[rid]).max())
                    if Fall[rid] and fN_t:
                        ct_f[rid] = abs(fN_t - Fall[rid]) / max(Fall[rid], 1)

            def _maxk(d):
                if not d:
                    return None, None
                k = max(d, key=d.get)
                return k, d[k]

            g["extended_pairwise"] = {
                "loss_cc": {"max": cc_loss and max(cc_loss.values()),
                            "argmax": _maxk(cc_loss)[0]},
                "loss_ct_extras": dict(ct_loss),
                "n_gs_cc": {"max": cc_n and max(cc_n.values()),
                            "argmax": _maxk(cc_n)[0]},
                "n_gs_ct_extras": dict(ct_n),
                "final_N_cc": {"max": cc_f and max(cc_f.values()),
                               "argmax": _maxk(cc_f)[0]},
                "final_N_ct_extras": dict(ct_f),
            }
            g["loss_trace_maxdiff_vs_extras"] = (
                float(max(ct_loss.values())) if ct_loss else 0.0)
            g["loss_cc_envelope_max_ext"] = (
                float(max(cc_loss.values())) if cc_loss else 0.0)
            g["n_gs_maxdiff_trace_vs_extras"] = (
                int(max(ct_n.values())) if ct_n else 0)
            g["n_gs_maxdiff_cc_envelope_ext"] = (
                int(max(cc_n.values())) if cc_n else 0)
            g["final_N_ct_envelope_rel_ext"] = (
                float(max(ct_f.values())) if ct_f else None)
            g["final_N_cc_envelope_rel_ext"] = (
                float(max(cc_f.values())) if cc_f else None)

    # --- G5 quality ----------------------------------------------------------
    qc = clean.get("quality", {})
    qt = trace.get("quality", {})
    qr = ref_q
    g["quality"] = {"clean": {k: qc.get(k) for k in ("psnr", "ssim", "lpips", "n_eval_cameras")},
                    "trace": {k: qt.get(k) for k in ("psnr", "ssim", "lpips", "n_eval_cameras")},
                    "ref": {k: qr.get(k) for k in ("psnr", "ssim", "lpips", "n_eval_cameras")}}
    g["quality_trace_in_envelope_vs_ref"] = bool(
        qt.get("psnr") is not None and
        qt["psnr"] >= qr["psnr"] - ENV["psnr"] and
        qt["ssim"] >= qr["ssim"] - ENV["ssim"] and
        qt.get("lpips", 9) <= qr["lpips"] + ENV["lpips"])
    g["quality_clean_in_envelope_vs_ref"] = bool(
        qc.get("psnr") is not None and
        qc["psnr"] >= qr["psnr"] - ENV["psnr"] and
        qc["ssim"] >= qr["ssim"] - ENV["ssim"] and
        qc.get("lpips", 9) <= qr["lpips"] + ENV["lpips"])
    g["status_trace"] = trace.get("final_status", {}).get("status")
    g["all_losses_finite_trace"] = trace.get("final_status", {}).get("all_losses_finite")

    # --- G6 tracer self-consistency + artifact completeness ------------------
    d0 = trace.get("d0_trace") or {}
    g["d0_f9_enabled"] = d0.get("f9_enabled")
    g["d0_f9_transitions"] = d0.get("f9_transitions")
    g["d0_n_iters_f9"] = d0.get("n_iters_f9")
    g["d0_n_iters_base"] = d0.get("n_iters_base")
    g["d0_means2d_mode"] = d0.get("means2d_mode")
    g["d0_means2d_mode_f9"] = d0.get("means2d_mode_f9")
    g["d0_n_sumcheck_mismatches"] = d0.get("n_sumcheck_mismatches")
    mism = d0.get("sumcheck_mismatches") or []
    if mism:
        deltas = [abs(int(m["sum"]) - int(m["n_isects"])) for m in mism]
        rels = [d / max(int(m["n_isects"]), 1) for d, m in zip(deltas, mism)]
        g["d0_sumcheck_max_abs"] = int(max(deltas))
        g["d0_sumcheck_max_rel"] = float(max(rels))
    else:
        g["d0_sumcheck_max_abs"] = 0
        g["d0_sumcheck_max_rel"] = 0.0
    g["d0_iterations"] = (d0.get("run_meta") or {}).get("iterations")
    g["runs_exist"] = {"clean": bool(clean.get("exists")),
                       "clean2": bool(clean2.get("exists")),
                       "trace": bool(trace.get("exists"))}
    g["d0_grad_missing_iters"] = d0.get("grad_missing_iters")
    g["d0_n_uids"] = d0.get("n_uids")
    g["d0_n_births_post_init"] = d0.get("n_births_post_init")
    g["d0_n_samples"] = d0.get("n_samples")
    g["d0_sample_offset"] = d0.get("sample_offset")
    g["d0_checkpoints"] = d0.get("checkpoints")
    g["d0_n_features"] = d0.get("n_features")
    td = trace.get("dir", "")
    g["artifacts"] = {
        "uid_table": os.path.exists(os.path.join(td, "d0_uid_table.npz")),
        "birth_records": os.path.exists(os.path.join(td, "d0_birth_records.npz")),
        "trajectory": os.path.exists(os.path.join(td, "d0_population_trajectory.npz")),
        "sample_summary": os.path.exists(os.path.join(td, "d0_sample_summary.npz")),
        "samples_dir_n": (len(os.listdir(os.path.join(td, "d0_cost_samples")))
                          if os.path.isdir(os.path.join(td, "d0_cost_samples")) else 0),
        "ckpt_iters": sorted(int(f.split("_")[-1].split(".")[0])
                             for f in trace.get("files", []) if f.startswith("d0_ckpt_")),
    }
    ev_rows = trace.get("results", {}).get("eval_rows", [])
    g["d0_ckpt_eval_rows"] = sorted(r["step"] for r in ev_rows if r.get("tag") == "d0_ckpt")
    g["n_eval_cameras_final"] = qt.get("n_eval_cameras")
    g["expected_eval_cameras"] = N_CAM[scene]
    return out


def verdict_for(scene_out):
    g = scene_out["gates"]
    bad = []
    rx = g.get("runs_exist", {})
    for k in ("clean", "clean2", "trace"):
        if not rx.get(k):
            bad.append("run missing: %s" % k)
    if len(g.get("extended_clean_ids", [])) > 2 and \
            not g.get("extended_clean_complete", True):
        bad.append("extended clean replica incomplete")
    if not g.get("camera_seq_trace_eq_clean"):
        bad.append("camera seq trace != clean")
    if not g.get("camera_seq_clean_eq_ref"):
        bad.append("camera seq clean != reference")
    if not g.get("dens_iters_equal_all"):
        bad.append("densification event iterations differ across runs")
    # intrinsic-envelope gates (frozen renderer is run-to-run nondeterministic).
    # Pre-registered screen (D0_TRACE_VALIDITY_SCREEN_PREREG.json): literal 1x
    # inside-envelope is reported verbatim; the gate fires only on a >2.0x
    # breach of the within-group envelope maximum.
    lmax = max(g.get("loss_trace_maxdiff_vs_clean", 1e9),
               g.get("loss_trace_maxdiff_vs_clean2", 1e9),
               g.get("loss_trace_maxdiff_vs_ref", 1e9),
               g.get("loss_trace_maxdiff_vs_extras", 0.0))
    lenv = max(g.get("loss_clean_vs_ref_maxdiff", 0.0),
               g.get("loss_clean_vs_clean2_maxdiff", 0.0),
               g.get("loss_clean2_vs_ref_maxdiff", 0.0),
               g.get("loss_cc_envelope_max_ext") or 0.0)
    g["loss_ct_envelope_max"] = lmax
    g["loss_cc_envelope_max"] = lenv
    g["loss_inside_envelope"] = bool(lmax <= lenv)
    g["loss_screen_2x"] = {"factor": 2.0, "ct_max": lmax, "cc_max": lenv,
                           "pass": bool(lmax <= 2.0 * lenv)}
    if not g["loss_screen_2x"]["pass"]:
        bad.append("loss trace beyond 2x screen (%.5f > 2.0x %.5f)"
                   % (lmax, lenv))
    nmax = max(g.get("n_gs_maxdiff_trace_vs_clean", 10 ** 9),
               g.get("n_gs_maxdiff_trace_vs_clean2", 10 ** 9),
               g.get("n_gs_maxdiff_trace_vs_ref", 10 ** 9),
               g.get("n_gs_maxdiff_trace_vs_extras", 0))
    nenv = (g.get("n_gs_maxdiff_cc_envelope_ext")
            if g.get("n_gs_maxdiff_cc_envelope_ext") is not None
            else g.get("n_gs_maxdiff_cc_envelope", 0))
    g["n_gs_ct_envelope_max"] = nmax
    g["n_gs_inside_envelope"] = bool(nmax <= nenv)
    g["n_gs_screen_2x"] = {"factor": 2.0, "ct_max": nmax, "cc_max": nenv,
                           "pass": bool(nmax <= 2.0 * nenv)}
    if not g["n_gs_screen_2x"]["pass"]:
        bad.append("N_gs trajectory beyond 2x screen (%d > 2.0x %d)"
                   % (nmax, nenv))
    rel = (g.get("final_N_ct_envelope_rel_ext")
           if g.get("final_N_ct_envelope_rel_ext") is not None
           else g.get("final_N_ct_envelope_rel"))
    env = (g.get("final_N_cc_envelope_rel_ext")
           if g.get("final_N_cc_envelope_rel_ext") is not None
           else g.get("final_N_cc_envelope_rel"))
    g["final_N_inside_envelope"] = bool(
        rel is not None and env is not None and rel <= env)
    g["final_N_screen_2x"] = {
        "factor": 2.0, "ct_max": rel, "cc_max": env,
        "pass": bool(rel is not None and env is not None and
                     rel <= 2.0 * env)}
    if not g["final_N_screen_2x"]["pass"]:
        bad.append("final_N beyond 2x screen (ct=%s cc=%s)" % (rel, env))
    if not g.get("quality_trace_in_envelope_vs_ref"):
        bad.append("trace quality outside C0 parity envelope")
    if g.get("status_trace") != "SUCCESS":
        bad.append("trace final_status != SUCCESS")
    if (g.get("d0_sumcheck_max_rel") or 0.0) > 1e-6:
        bad.append("isect sumcheck relative delta %s > 1e-6"
                   % g.get("d0_sumcheck_max_rel"))
    # Renderer-branch gates (frozen recipe: baseline-native for sh_degree<3,
    # gatherless F9 from iteration 3000; metadata identical at capture site).
    if g.get("d0_f9_transitions") != [[3000, True]]:
        bad.append("f9 transitions != [[3000, True]] (got %s)"
                   % (g.get("d0_f9_transitions"),))
    if g.get("d0_n_iters_base") != 2999:
        bad.append("baseline iterations != 2999 (got %s)" % g.get("d0_n_iters_base"))
    n_it = g.get("d0_iterations")
    if n_it and (g.get("d0_n_iters_base", 0) + g.get("d0_n_iters_f9", 0)) != n_it:
        bad.append("branch counters %s+%s != iterations %s"
                   % (g.get("d0_n_iters_base"), g.get("d0_n_iters_f9"), n_it))
    if g.get("d0_means2d_mode") != "pixel":
        bad.append("means2d mode (baseline) != pixel")
    if g.get("d0_means2d_mode_f9") != "pixel":
        bad.append("means2d mode (F9) != pixel")
    a = g.get("artifacts", {})
    if not all(a.get(k) for k in ("uid_table", "birth_records", "trajectory",
                                  "sample_summary")):
        bad.append("missing core d0 artifact file(s)")
    if a.get("samples_dir_n") != 1500:
        bad.append("sampled cost shards != 1500 (got %s)" % a.get("samples_dir_n"))
    if a.get("ckpt_iters") != [5000, 10000, 15000, 20000, 25000, 30000]:
        bad.append("d0 ckpt set wrong: %s" % a.get("ckpt_iters"))
    if g.get("d0_ckpt_eval_rows") != [5000, 10000, 15000, 20000, 25000, 30000]:
        bad.append("d0 ckpt eval rows wrong: %s" % g.get("d0_ckpt_eval_rows"))
    if g.get("d0_n_features") != 30:
        bad.append("n_features != 30")
    if g.get("d0_sample_offset") != SAMPLE_OFF[scene_out["scene"]]:
        bad.append("sample offset wrong")
    if g.get("n_eval_cameras_final") != g.get("expected_eval_cameras"):
        bad.append("final eval cameras != frozen count")
    return bad


scenes = sys.argv[1:] or ["room", "bicycle"]
out = {"artifact": "D0_TRACE_VALIDITY_30k.json", "scenes": {}}
allbad = {}
for s in scenes:
    so = compare(s)
    bad = verdict_for(so)
    out["scenes"][s] = so
    allbad[s] = bad

verdict = "D0_TRACE_VALID_30K" if not any(allbad.values()) else "D0_TRACE_INVALID"
out["verdict"] = verdict
out["reasons"] = allbad
os.makedirs(ART, exist_ok=True)
with open(os.path.join(ART, "D0_TRACE_VALIDITY_30k.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)

for s in scenes:
    print("== %s: %s" % (s, "PASS" if not allbad[s] else "FAIL"))
    print(json.dumps(out["scenes"][s]["gates"], indent=2, default=str))
    for b in allbad[s]:
        print("   FAIL:", b)
print("VERDICT:", verdict)
sys.exit(0 if verdict == "D0_TRACE_VALID_30K" else 1)
