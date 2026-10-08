#!/usr/bin/env python3
"""P13-Y report: render P13Y_REPORT.md (OBSERVED / EVIDENCE / HYPOTHESIS /
TEST / CONCLUSION) from the frozen artifacts. Read-only over the artifacts:
this script never recomputes a gate, it only formats recorded decisions.
"""
import json
import os

P13Y = "/root/epic05_p13y"
ART = f"{P13Y}/artifacts"
SCENES = ["room", "bicycle", "garden"]


def load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    verdict = load_json(f"{ART}/P13Y_VERDICT.json", {})
    agg = load_json(f"{ART}/P13Y_Y23_AGG.json", {})
    ident = load_json(f"{ART}/P13Y_SOURCE_IDENTITY.json", {})
    seq = load_json(f"{ART}/P13Y_SEQUENCE_INDEXING.json", {})
    neum = load_json(f"{ART}/P13Y_CAPTURE_NEUTRALITY.json", {})
    cont = load_json(f"{ART}/P13Y_CONTAMINATION.json", {})
    man = load_json(f"{ART}/MANIFEST_P13Y.json", {})

    L = []
    A = L.append
    A("# P13-Y — Persistent Linearization / Stale-Jacobian Oracle: report")
    A("")
    A(f"Date: 2026-09-27. Host: EPIC-05. Decision class: **{verdict.get('class')}**.")
    A("")
    A("Scope discipline: 8 pairs/scene is discovery only (no paper claim); no "
      "training-speedup claim; no novelty claim; gates frozen before results.")
    A("")
    # ---------------- OBSERVED ----------------
    A("## OBSERVED")
    A("")
    A("Forward Taylor linearity and stale-Jacobian backward were measured on real "
      "training revisits of the C0 arm, in the topology-frozen window "
      "15000 <= t <= 17000 (room / bicycle / garden).")
    A("")
    for s in SCENES:
        sc = agg.get("scenes", {}).get(s, {})
        A(f"- **{s}**: {sc.get('n_joint_pass')}/8 joint pass, "
          f"{sc.get('n_Y2_pass')}/8 Y2 pass, {sc.get('n_Y3_pass')}/8 Y3 pass, "
          f"{sc.get('n_catastrophic')} catastrophic, {sc.get('n_invalid')} invalid "
          f"-> scene class **{sc.get('scene_class')}**.")
    A("")
    A(f"Adapter parity gate: **{agg.get('adapter_parity_gate_status')}**. "
      f"Capture neutrality: **{neum.get('criterion', {}).get('pass')}**. "
      f"Source identity: **{ident.get('status')}**.")
    A("")
    # ---------------- EVIDENCE ----------------
    A("## EVIDENCE")
    A("")
    A("### E1. Source identity (Sec.1)")
    A("")
    if ident:
        ck = ident.get("checks", {})
        rows = []
        for name in ("p12_trainer", "c0_extension_so", "c0_core_so",
                     "fused_ssim_cuda_so"):
            e = ck.get(name, {})
            rows.append([name, e.get("sha256"), e.get("expected_sha256"),
                         "match" if e.get("match") else (e.get("error") or "MISMATCH")])
        A(table(rows, ["object", "sha256", "expected", "status"]))
        A("")
        h = ck.get("higs_render_source", {})
        A(f"HIGS render source: `{h.get('path')}` sha256 `{h.get('sha256')}`; "
          f"documented gsplat pin `{h.get('documented_pin')}` "
          f"({h.get('note')}).")
        cap = ck.get("capture_trainer", {})
        A(f"Capture trainer: sha256 `{cap.get('sha256')}` "
          f"(base P12 sha `{cap.get('p12_base_sha256')}`), additive diff sha256 "
          f"`{cap.get('additive_diff_sha256')}`.")
    else:
        A("_P13Y_SOURCE_IDENTITY.json missing._")
    A("")
    A("### E2. Sequence indexing proof (Sec.3)")
    A("")
    if seq:
        lines = seq.get("static_evidence", {}).get("lines", {})
        n_anchor = sum(1 for v in lines.values() if v.get("line_numbers"))
        A(f"- L1 static: `camera_sequence.npy[i]` is the camera of training "
          f"iteration `i+1` (1-based loop header, consumption index "
          f"`iteration-1`); {n_anchor}/{len(lines)} source anchors located in "
          f"the frozen P12 trainer.")
        A(f"- L2 regeneration: bitwise re-derivation with `random.Random(42)` "
          f"pop-from-stack matches the frozen P12 30K files for all scenes: "
          f"{verdict.get('sequence_indexing_conclusion', {}).get('regeneration_matches_all_scenes')}.")
        A(f"- L3 empirical: in-run cam-log alignment assertion "
          f"(`cam_log[i] == (i+1, camera_sequence[i])`) + parity probe per scene.")
        for s in SCENES:
            ps = seq.get("per_scene", {}).get(s, {})
            rg = ps.get("regeneration", {})
            A(f"  - {s}: npy `{ps.get('npy')}` dtype={ps.get('dtype')} "
              f"shape={ps.get('shape')}; regeneration bitwise "
              f"`{rg.get('matches_bitwise')}`; every-camera-once-per-cycle "
              f"`{ps.get('every_camera_once_per_cycle')}`; empirical "
              f"`{ps.get('empirical')}`.")
    else:
        A("_P13Y_SEQUENCE_INDEXING.json missing._")
    A("")
    A("### E3. Revisit distribution + frozen pair plan (Sec.3)")
    A("")
    for s in SCENES:
        dist = load_json(f"{ART}/P13Y_REVISIT_DISTRIBUTION_{s}.json")
        plan = load_json(f"{ART}/P13Y_PAIR_PLAN_{s}.json")
        if dist:
            lw = dist.get("lag_quantile_table_in_window", {})
            A(f"- {s}: {dist.get('n_iterations')} iterations, {dist.get('n_cameras')} "
              f"cameras; adjacent revisits {dist.get('n_adjacent_revisits_in_window')} "
              f"in window {dist.get('window')} (full: "
              f"{dist.get('n_adjacent_revisits_full')}); in-window lag "
              f"min/median/max = {fmt(lw.get('min'))}/"
              f"{fmt((lw.get('quantiles') or {}).get('0.5'))}/{fmt(lw.get('max'))}.")
        if plan:
            rows = []
            for e in plan.get("quantile_selection", []):
                p = e.get("pair", {})
                rows.append([fmt(e.get("quantile")), fmt(e.get("lag_target")),
                             p.get("camera_id"), p.get("t0"), p.get("t1"),
                             p.get("lag")])
            A("")
            A(f"**{s} frozen pair plan** (`frozen={plan.get('frozen')}`, "
              f"date={plan.get('date')}, sequence sha "
              f"`{(plan.get('sequence_sha256') or '')[:16]}`):")
            A("")
            A(table(rows, ["quantile", "lag_target", "cam", "t0", "t1", "lag"]))
            A("")
    A("### E4. Capture neutrality (Sec.4)")
    A("")
    if neum:
        A(f"- CPU/NumPy/Torch-CPU/Torch-CUDA RNG saved+restored around every "
          f"capture (`rng_restored_exactly` asserted in-run per capture).")
        A(f"- Trio runs: {json.dumps(neum.get('runs', {}), default=str)}")
        A(f"- camera_sequence identical in all three: "
          f"{neum.get('camera_sequence_identical_all_three')}; final params "
          f"OFF1==OFF2: {neum.get('final_params_identical_off1_off2')}, "
          f"ON==OFF1: {neum.get('final_params_identical_on_off1')}.")
        A(f"- noise floor (OFF1 vs OFF2): "
          f"{json.dumps(neum.get('noise_floor_off1_vs_off2', {}), default=str)}")
        A(f"- capture effect (ON vs OFF1): "
          f"{json.dumps(neum.get('capture_effect_on_vs_off1', {}), default=str)}")
        A(f"- Criterion pass: **{neum.get('criterion', {}).get('pass')}** "
          f"({neum.get('criterion', {}).get('rule')}) checks="
          f"{json.dumps(neum.get('criterion', {}).get('checks', {}), default=str)}.")
        if neum.get("full_replay_vs_p12_original"):
            fr = neum["full_replay_vs_p12_original"]
            A(f"- Full capture replay vs P12 30K prefix: n_rows={fr.get('n_iter_rows_compared')}, "
              f"n_loss_differing={fr.get('n_loss_differing')}, "
              f"max_abs_loss_diff={fmt(fr.get('max_abs_loss_diff'))}, "
              f"first_differing_iter={fr.get('first_differing_iter')}, "
              f"n_gs_equal_all={fr.get('n_gs_equal_all_rows')}, "
              f"camera_seq_match={fr.get('camera_sequence_sha256_match')}.")
    else:
        A("_P13Y_CAPTURE_NEUTRALITY.json missing._")
    A("")
    A("#### E4b. Neutrality diagnosis (post-hoc, appended after the frozen verdict)")
    A("")
    sh = load_json(f"{ART}/P13Y_CAPTURE_NEUTRALITY_SHORT.json")
    ens = load_json(f"{ART}/P13Y_NEUTRALITY_ENSEMBLE.json")
    A("The frozen criterion is a pointwise single-sample comparison against one "
      "OFF replay; it was evaluated before this diagnosis and is NOT edited here. "
      "What the diagnosis adds:")
    A("")
    A("- Cross-process float nondeterminism in the frozen C0 recipe seeds at "
      "iteration ~14 with NO instrumentation at all (OFF1 vs OFF2 diverge at iter "
      f"{fmt((neum.get('noise_floor_off1_vs_off2') or {}).get('first_differing_iter'), 0)}); "
      "it was already documented in the P13-X preflight as C_vs_D ~ D_vs_D2. "
      "The same seed iteration is therefore present in every compared pair, so "
      "the trio comparison is a chaotic-amplification comparison, not a "
      "deterministic reproduction test.")
    if sh:
        nf = sh.get("noise_floor_off1_vs_off2", {})
        ce = sh.get("capture_effect_on_vs_off1", {})
        A(f"- Short trio (400 iters): noise floor max|dL|={fmt(nf.get('max_abs_loss_diff'))} "
          f"(first diff iter {fmt(nf.get('first_differing_iter'), 0)}) vs capture "
          f"effect max|dL|={fmt(ce.get('max_abs_loss_diff'))} "
          f"(first diff iter {fmt(ce.get('first_differing_iter'), 0)}).")
    if ens:
        d = ens.get("distribution", {})
        mo = d.get("max_abs_loss_diff", {})
        A(f"- Five-OFF ensemble vs one ON (400 iters): OFF-OFF max|dL| spans "
          f"{fmt(mo.get('off_off_min'))}..{fmt(mo.get('off_off_max'))} "
          f"(median {fmt(mo.get('off_off_median'))}); ON-OFF spans "
          f"{fmt(min(mo.get('on_off_values', [0])))}..{fmt(mo.get('on_off_max'))} "
          f"-> ON lies OUTSIDE the OFF-OFF envelope "
          f"(envelope_ok={mo.get('on_off_within_off_off_envelope')}), "
          f"i.e. beyond replay-to-replay spread.")
        q = ens.get("end_state_quality", {})
        for k in ("psnr", "ssim", "lpips"):
            e = q.get(k, {})
            if e:
                A(f"  - end-state {k}: OFF {e.get('off_min'):.9g}..{e.get('off_max'):.9g}, "
                  f"ON {e.get('on_value'):.9g} (inside envelope "
                  f"{e.get('on_inside_off_envelope')}).")
        A(f"- OUTCOME (recorded, no gate): the short-horizon (400-iter) instrumented "
          f"run is a distributional outlier -- every ON-OFF max|dL| is larger than "
          f"every OFF-OFF max|dL| ({fmt(max(mo.get('on_off_values', [0])))} vs "
          f"{fmt(mo.get('off_off_max'))}, ~28x), while the ON end-state quality "
          f"deviations are in the 1e-4 dB / 3e-5 relative range. At the longer "
          f"horizon (3500 iters) ON-OFF max|dL|={fmt((neum.get('capture_effect_on_vs_off1') or {}).get('max_abs_loss_diff'))} "
          f"is BELOW the OFF-OFF max "
          f"{fmt((neum.get('noise_floor_off1_vs_off2') or {}).get('max_abs_loss_diff'))}, "
          f"so the outlier pattern does not persist at 3500 iterations. Both "
          f"observations are recorded as-is; neither alters the frozen criterion "
          f"verdict (FAIL) nor the sec.9/sec.10 class.")
    A(f"- Reading (recorded, not a gate): {ens.get('reading') if ens else 'n/a'}")
    A("- Consequence for this phase: Y2/Y3 compare theta0 and theta1 taken from the "
      "SAME run along the SAME trajectory, so the oracle measurement itself is "
      "internally valid; but the capture side channel is NOT proven pointwise-"
      "neutral, so any future design that assumes provable capture neutrality must "
      "re-establish it (e.g. state capture from a deterministic replay checkpoint, "
      "or an out-of-process re-render) before that assumption is used.")
    A("")
    A("### E5. Adapter parity (Sec.5)")
    A("")
    for s in SCENES:
        p = agg.get("adapter_parity", {}).get(s)
        if p:
            A(f"- {s}: status={p.get('status')}, n_probes={p.get('n_probes')}, "
              f"max|I_A-I_B|={fmt(p.get('max_abs_diff'))}, "
              f"max loss rel err={fmt(p.get('max_loss_rel_err'))}")
    A("")
    A("### E6. Y1 graph stability (Sec.6, descriptive only)")
    A("")
    for s in SCENES:
        sc = agg.get("scenes", {}).get(s, {})
        rows = [[fmt(sc.get("median_Y1_visible_jaccard")),
                 fmt(sc.get("median_Y1_membership_jaccard")),
                 fmt(sc.get("median_Y1_order_inversion_rate")),
                 fmt(sc.get("median_Y1_unchanged_full_order_tile_fraction"))]]
        A(f"- {s} medians:")
        A("")
        A(table(rows, ["visible_jaccard", "membership_jaccard",
                       "order_inversion_rate", "unchanged_full_order_frac"]))
        A("")
    A("### E7. Y2 forward linearity (Sec.7)")
    A("")
    for s in SCENES:
        sc = agg.get("scenes", {}).get(s, {})
        rows = [[fmt(sc.get("median_E_delta")), fmt(sc.get("median_E_I")),
                 fmt(sc.get("median_psnr_lin")), fmt(sc.get("median_L_rel_err")),
                 fmt(sc.get("median_h_consistency_max_rel"))]]
        A(f"- {s} medians:")
        A("")
        A(table(rows, ["E_delta", "E_I", "PSNR(I_lin,I1)", "L_rel_err",
                       "h_consistency_max_rel"]))
        A("")
    A("### E8. Y3 stale-Jacobian backward (Sec.8)")
    A("")
    for s in SCENES:
        sc = agg.get("scenes", {}).get(s, {})
        rows = [[fmt(sc.get("median_cos_means")), fmt(sc.get("median_E_rel_means")),
                 fmt(sc.get("median_cos_opacity")), fmt(sc.get("median_cos_scales")),
                 fmt(sc.get("median_cos_quats")), fmt(sc.get("median_cos_sh0")),
                 fmt(sc.get("median_cos_shN"))]]
        A(f"- {s} medians:")
        A("")
        A(table(rows, ["cos_means", "E_rel_means", "cos_opacity", "cos_scales",
                       "cos_quats", "cos_sh0", "cos_shN"]))
        A("")
    A("### E9. Per-pair table")
    A("")
    for s in SCENES:
        sc = agg.get("scenes", {}).get(s, {})
        rows = []
        for r in sc.get("pair_table", []):
            v = r.get("verdict") or {}
            rows.append([r.get("idx"), r.get("cam"), r.get("t0"), r.get("t1"),
                         r.get("lag"), fmt(r.get("E_delta")),
                         fmt(r.get("L_rel_err")), fmt(r.get("cos_means")),
                         fmt(r.get("E_rel_means")), fmt(r.get("cos_opacity")),
                         fmt(r.get("Y1_vis_jaccard")),
                         "J" if v.get("joint_pass") else ("INV" if v.get("invalid")
                                                          else "f")])
        A(f"**{s}**")
        A("")
        A(table(rows, ["i", "cam", "t0", "t1", "lag", "E_delta", "L_rel",
                       "cos_mu", "E_rel_mu", "cos_opa", "Y1_visJ", "v"]))
        A("")
    A("### E10. Contamination ledger (Sec.12)")
    A("")
    A(f"- units logged: {cont.get('n_units')}; "
      f"{[(u['unit'], u['rc']) for u in cont.get('units', [])]}")
    A(f"- policy: {cont.get('policy')}")
    A("")
    A("### E11. Manifest")
    A("")
    A(f"- {man.get('n_files')} files, {man.get('total_bytes', 0) / 1e9:.2f} GB under "
      f"`{man.get('root')}` (sha256 per file in MANIFEST_P13Y.json).")
    A("")
    # ---------------- HYPOTHESIS ----------------
    A("## HYPOTHESIS")
    A("")
    A("H1 (forward): for a fixed real training camera c and adjacent true revisits "
      "t0->t1 in the post-densification window, the render is locally affine in the "
      "raw parameter vector: `R_c(theta1) ~ R_c(theta0) + J_c(theta0)(theta1-theta0)` "
      "with E_delta <= 0.10.")
    A("")
    A("H2 (backward): with one frozen adjoint lambda1 = dL/dI at t1, the stale "
      "Jacobian transpose delivers the same parameter gradient: "
      "`J_c(theta0)^T lambda1 ~ J_c(theta1)^T lambda1`, cos >= 0.995 on "
      "means+opacity (secondary groups cos >= 0.990), E_rel <= 0.05 (0.10).")
    A("")
    A("H3 (compositional): if H1+H2 hold with a high joint-pass rate across scenes, "
      "a persistent linearization oracle could skip relinearization for a bounded "
      "reuse lifetime; otherwise the mechanism dies and the program moves to "
      "Spectral / Compressed Adjoint without rescue.")
    A("")
    A("H1/H2 as stated are the operands of the frozen gates; no gate was edited "
      "after any result.")
    A("")
    # ---------------- TEST ----------------
    A("## TEST")
    A("")
    A("- Sec.7 Y2: d = theta1-theta0; central FD at h=0.01 (primary), crosschecked "
      "at h=0.005/0.02 with NO post-hoc h selection; I_lin = I0 + J0 d; primary "
      "E_delta = ||I_lin-I1||/||I1-I0||; Y2 valid iff "
      "||I1-I0||/||I1|| >= 1e-6 and h-consistency <= 0.05; Y2 PASS iff "
      "E_delta <= 0.10 and |L(I_lin)-L(I1)|/|L(I1)| <= 0.005.")
    A("- Sec.8 Y3: at theta1, lambda1 = dL(I1,GT)/dI1 (fused training loss); "
      "g_true = J(theta1)^T lambda1, g_stale = J(theta0)^T lambda1 (same lambda1, "
      "fresh graphs); per-group cosine / relative L2 / norm ratio / top-10%-magnitude "
      "sign agreement; primary means+opacity cos >= 0.995 and E_rel <= 0.05, "
      "secondary scales/quats/sh0/shN cos >= 0.990 and E_rel <= 0.10; a "
      "two-step-vs-direct gradient check validates the split path.")
    A("- Sec.9 scene gate and Sec.10 global classes were evaluated exactly as "
      "frozen (no post-hoc edits).")
    A("- Y1 (Sec.6) is descriptive only and did not decide linearization; Y2/Y3 "
      "ran on ALL selected pairs.")
    A("")
    # ---------------- CONCLUSION ----------------
    A("## CONCLUSION")
    A("")
    A(f"**{verdict.get('class')}**")
    A("")
    A(f"- scene classes: {json.dumps(verdict.get('scene_classes'), default=str)}")
    A(f"- adapter parity gate: {verdict.get('adapter_parity_gate')}")
    A(f"- capture neutrality pass: {verdict.get('capture_neutrality_pass')}")
    A(f"- next step: {verdict.get('next_step')}")
    A("")
    if verdict.get("class") == "PRIMARY_BREAKTHROUGH_CANDIDATE":
        A("Sec.10 constraints now attach to any Y4 design: composed executable oracle "
          "must count full relinearization frequency, stale-linear application cost, "
          "cache refresh cost, loss cost, optimizer cost, memory traffic, cache "
          "footprint, and densification/other fixed costs; CUDA implementation only "
          "if S_E2E,conservative >= 4.0x; test 25th-percentile reuse lifetime >= 3x; "
          "if persistent cache > 40% of A100 80GB -> HOLD/DROP unless streamed cache "
          "overhead <= 10%.")
        A("")
    else:
        A("Per Sec.13 the program stops here for this mechanism and moves to "
          "Spectral / Compressed Adjoint; persistent linearization is NOT rescued.")
        A("")
    A("### Failures / anomalies preserved")
    A("")
    n_inv = sum(agg.get("scenes", {}).get(s, {}).get("n_invalid", 0) for s in SCENES)
    n_cat = sum(agg.get("scenes", {}).get(s, {}).get("n_catastrophic", 0) for s in SCENES)
    A(f"- invalid pairs: {n_inv}; catastrophic pairs: {n_cat} (per-pair JSONs and "
      f"tracebacks kept verbatim under pair_results/).")
    A(f"- contaminated units (if any) are retained verbatim in "
      f"logs/P13Y_CONTAMINATION.log and P13Y_CONTAMINATION.json.")
    A(f"- no production persistent CUDA cache was written; no 30K full training was "
      f"started; no external speedup/novelty claim is made; no git commit.")
    A("")

    out = os.path.join(P13Y, "P13Y_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("WROTE", out, f"({len(L)} lines)")


if __name__ == "__main__":
    main()
