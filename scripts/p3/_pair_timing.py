#!/usr/bin/env python3
"""TASK 2 (subprocess fallback): paired PROD-vs-V0 timing.

Same-process importlib load of both probe .so FAILED with:
  ImportError: generic_type: type "GaussianInferenceRenderer" is already registered!
This is a pybind11 per-interpreter CLASS-registry collision (both .so export
py::class_<GaussianInferenceRenderer>), orthogonal to the torch 'experimental'
namespace (which our rename p3h_probe_* already solved). So we fall back to
one-subprocess-per-probe as the task spec permits, running the IDENTICAL timing
protocol (warmup 20, reps 5, samples 100) per probe, sequentially, on the same
GPU. The only remaining confound is separate processes (label it).
"""
import json, os, subprocess, sys, time

PROD_SO = "/mnt/storage_pool/liaoyuanjun/p3h_probe_cache/probe_prod/experimental_p3h_prod_cuda/experimental_p3h_prod_cuda.so"
V0_SO   = "/mnt/storage_pool/liaoyuanjun/p3h_probe_cache/probe_v0/experimental_p3h_v0_cuda/experimental_p3h_v0_cuda.so"
WORKER  = "/mnt/storage_pool/liaoyuanjun/p3h_probe/_pair_worker.py"

SCENES = ["room", "bicycle", "garden"]
REPS, SAMPLES, WARMUP = 5, 100, 20

def run_probe(key, so, out_json):
    env = dict(os.environ)
    env["PROBE_KEY"] = key
    env["PROBE_SO"] = so
    env["SCENES"] = ",".join(SCENES)
    env["REPS"] = str(REPS)
    env["SAMPLES"] = str(SAMPLES)
    env["WARMUP"] = str(WARMUP)
    env["OUT_JSON"] = out_json
    env["PATH"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:" + env.get("PATH", "")
    env["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"
    t0 = time.time()
    p = subprocess.run([env["PATH"].split(":")[0] + "/python", WORKER],
                       env=env, capture_output=True, text=True)
    dt = time.time() - t0
    print("PROBE %s exit=%d wall=%.1fs" % (key, p.returncode, dt), flush=True)
    if p.returncode != 0:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
        raise SystemExit("worker %s failed" % key)
    return json.loads(Path(out_json).read_text())

from pathlib import Path  # noqa: E402

def main():
    # ensure worker exists
    if not os.path.exists(WORKER):
        raise SystemExit("worker missing: " + WORKER)
    prod_res = run_probe("prod", PROD_SO, "/mnt/storage_pool/liaoyuanjun/p3h_probe/prod_result.json")
    v0_res   = run_probe("v0",   V0_SO,   "/mnt/storage_pool/liaoyuanjun/p3h_probe/v0_result.json")
    print("\n=== PAIRED SUMMARY (subprocess, same GPU, same protocol) ===")
    out = []
    for scene in SCENES:
        pr = prod_res[scene]  # list of per-rep ms
        vr = v0_res[scene]
        import statistics as st
        med_p = st.median(pr); med_v = st.median(vr)
        ratios = [v / p for p, v in zip(pr, vr)]
        out.append({
            "scene": scene,
            "prod_reps_ms": [round(x,4) for x in pr],
            "v0_reps_ms":   [round(x,4) for x in vr],
            "median_prod_ms": round(med_p,4),
            "median_v0_ms":   round(med_v,4),
            "v0_over_prod_ratio": round(med_v/med_p,5),
            "median_paired_ratio": round(st.median(ratios),5),
            "mean_paired_delta_pct": round((st.mean(ratios)-1)*100,3),
            "max_paired_delta_pct": round((max(ratios)-1)*100,3),
        })
        print(json.dumps(out[-1], indent=2))
    Path("/mnt/storage_pool/liaoyuanjun/p3h_probe/pair_summary.json").write_text(
        json.dumps(out, indent=2))
    print("DONE_TASK2_SUBPROCESS")

if __name__ == "__main__":
    main()