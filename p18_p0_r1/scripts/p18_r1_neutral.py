#!/usr/bin/env python3
"""P18-P0-R1 instrumentation-neutrality probe (within-process A/B).

Base pipeline is NOT bit-reproducible across processes (flaky fp16-ulp-scale
differences on isolated cams), so cross-process hash equality is the wrong
criterion. This probe removes process noise: ONE process renders the full probe
sequence four times, toggling the trace instrumentation:

    off -> on -> off2 -> on2

All passes must be bit-identical per cam (max|delta| == 0, nz == 0). The two
armed passes must additionally agree on live-population histograms. Writes
P18_R1_NEUTRALITY.json.
"""
import json
import sys
import time

sys.path.insert(0, "/root/epic05_p18_r1/scripts")
import numpy as np
import torch

from p18_r1_capture import (CTA_WARPS, IDX_GATE, LOCAL_CAP, MAX_CTAS, OUT,
                            TRACE_EXT, bootstrap, hash_tensor, load_model,
                            pack_scene, render_cam)

PASSES = [("off", False), ("on", True), ("off2", False), ("on2", True)]


def main():
    exp = bootstrap(TRACE_EXT)
    exp.p18_trace_set(0, 0, 0, 0, 255)
    torch.cuda.synchronize()

    rec = torch.full((MAX_CTAS * CTA_WARPS * LOCAL_CAP * 16, 8), -1,
                     dtype=torch.int32, device="cuda")
    hist = torch.zeros(17, dtype=torch.int32, device="cuda")

    camset = json.load(open("/root/epic05_p18/P18_CAMERA_SET.json"))
    result = {"passes": [p for p, _ in PASSES],
              "hist_totals": {}, "hashes": {}, "pairs": {}, "verdict": {}}

    for scene in ["room", "bicycle"]:
        model = load_model(scene)
        obj = pack_scene(model, scene)
        scores = camset["scenes"][scene]
        cams = list(scores["probe_ids"])

        # warmup (settle allocator/cache state) with tracing off
        render_cam(obj, scores["camera_records"][str(cams[0])])
        torch.cuda.synchronize()
        print(f"[neutral] {scene}: warmup done", flush=True)

        renders = {}
        hashes = {}
        hists = {}
        for pname, armed in PASSES:
            if armed:
                exp.p18_trace_set(rec.data_ptr(), hist.data_ptr(),
                                  MAX_CTAS, LOCAL_CAP, IDX_GATE)
            else:
                exp.p18_trace_set(0, 0, 0, 0, 255)
            torch.cuda.synchronize()
            tag = f"{scene}.{pname}"
            hs, hh = {}, {}
            t0 = time.time()
            for ci in cams:
                if armed:
                    rec.fill_(-1)
                    hist.zero_()
                r, a, W, H = render_cam(obj, scores["camera_records"][str(ci)])
                torch.cuda.synchronize()
                h, _, _ = hash_tensor(r)
                hs[str(ci)] = h
                if armed:
                    hh[str(ci)] = int(hist.sum().item())
                renders[(pname, ci)] = r
            hashes[pname] = hs
            if armed:
                hists[pname] = hh
            print(f"[neutral] {scene} pass={pname} finished in {time.time()-t0:.1f}s",
                  flush=True)

        # pairwise tensor comparison across passes
        pairs = {}
        for i in range(len(PASSES)):
            for j in range(i + 1, len(PASSES)):
                pi, pj = PASSES[i][0], PASSES[j][0]
                worst = {"max_abs": 0.0, "nz": 0, "mean_abs": 0.0, "cam_max_abs": {}}
                n_all = 0
                for ci in cams:
                    a = renders[(pi, ci)].float()
                    b = renders[(pj, ci)].float()
                    d = (a - b).abs()
                    mx = float(d.max().item())
                    nz = int((d > 0).sum().item())
                    worst["cam_max_abs"][str(ci)] = mx
                    worst["max_abs"] = max(worst["max_abs"], mx)
                    worst["nz"] += nz
                    worst["mean_abs"] += float(d.sum().item())
                    n_all += int(d.numel())
                worst["mean_abs"] /= max(n_all, 1)
                worst["hash_identical"] = (pi == pj) or all(
                    hashes[pi][str(ci)] == hashes[pj][str(ci)] for ci in cams)
                pairs[f"{pi}__vs__{pj}"] = worst
                print(f"[neutral] {scene} {pi} vs {pj}: max_abs={worst['max_abs']:.3e} "
                      f"nz={worst['nz']} hash_identical={worst['hash_identical']}",
                      flush=True)

        hist_eq = None
        if "on" in hists and "on2" in hists:
            hist_eq = hists["on"] == hists["on2"]
            print(f"[neutral] {scene} hist(on)==hist(on2): {hist_eq}", flush=True)

        result["hashes"][scene] = hashes
        result["hist_totals"][scene] = hists
        result["pairs"][scene] = pairs
        result["verdict"][scene] = {
            "bit_identical_all_passes": all(
                v["max_abs"] == 0.0 and v["nz"] == 0 for v in pairs.values()),
            "hist_on_eq_on2": hist_eq,
        }
        del model, obj, renders
        torch.cuda.empty_cache()

    ok = all(v["bit_identical_all_passes"] and v["hist_on_eq_on2"]
             for v in result["verdict"].values())
    result["overall"] = "PASS" if ok else "FAIL"
    (OUT / "P18_R1_NEUTRALITY.json").write_text(json.dumps(result, indent=2))
    print(f"[neutral] OVERALL={result['overall']} -> P18_R1_NEUTRALITY.json",
          flush=True)


if __name__ == "__main__":
    main()
