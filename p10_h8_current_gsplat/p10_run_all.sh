#!/bin/bash
# P10 H8-MR gate runner (EPIC-05).
#
# GPU policy: this host is shared. GPUs 0-3 were idle when the phase started and
# GPU 0 was used for H2; GPUs 4-7 belong to an unrelated job and are never
# touched. Before any timed run, check that no foreign process is resident --
# p10_h3_timing.py bench refuses unless --allow-shared is passed, and
# p10_h3_timing.py bench REFUSES with exit code 3 on a busy GPU.
#
# CPU-only steps (H0, H4 static, H4 ncu probe) can run at any time.
set -euo pipefail
cd /root/epic05_p10
source /root/miniforge3/etc/profile.d/conda.sh
conda activate gsplat
export CUDA_VISIBLE_DEVICES=0
mkdir -p dumps

H8SO=/root/epic05_p10/build/gsplat_cuda_p10_h8/gsplat_cuda_p10_h8.so
PRISO=/root/epic05_p10/build/gsplat_cuda_p10_pristine/gsplat_cuda_p10_pristine.so
CTLSO=/root/epic05_p10/build/gsplat_cuda_p10_pristine_ctl/gsplat_cuda_p10_pristine_ctl.so
export H8SO PRISO CTLSO
SCENES="${SCENES:-room bicycle garden}"
ONLY="${ONLY:-all}"

if [ "$ONLY" = "all" ] || [ "$ONLY" = "h0" ]; then
echo "=== H0 source identity (CPU) ==="
python p10_h0_identity.py --out dumps/EPIC05_P10_SOURCE_IDENTITY.json
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "h2" ]; then
echo "=== H2 GPU gradient equivalence ==="
for s in $SCENES; do
  python p10_h2_equivalence.py dump --binary pristine --scene "$s" --gpu 0 --absgrad --out "dumps/h2_pri_$s.pt"
  python p10_h2_equivalence.py dump --binary pristine --scene "$s" --gpu 0 --absgrad --out "dumps/h2_pri_${s}_p2.pt"
  python p10_h2_equivalence.py dump --binary h8       --scene "$s" --gpu 0 --absgrad --out "dumps/h2_h8_$s.pt"
  # --pristine2 supplies the cross-process same-build null for the inertness
  # control: the atomic-scattered buffers are reproducible inside a process but
  # not across processes, so a within-process pool would make the control fail
  # spuriously on heavy-tailed tensors (see P10_H8_CURRENT_GSPLAT_REPORT.md).
  python p10_h2_equivalence.py compare --scene "$s" \
      --pristine "dumps/h2_pri_$s.pt" --pristine2 "dumps/h2_pri_${s}_p2.pt" \
      --h8 "dumps/h2_h8_$s.pt" --out "dumps/EPIC05_P10_EQUIVALENCE_$s.json"
done
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "h3" ]; then
echo "=== H3 direct backward microbenchmark (clean GPU required) ==="
for s in $SCENES; do
  python p10_h3_timing.py bench --tag h8off --binary h8       --arm off --scene "$s" --gpu 0 --out "dumps/h3_$s.json"
  python p10_h3_timing.py bench --tag h8on  --binary h8       --arm on  --scene "$s" --gpu 0 --out "dumps/h3_$s.json"
  python p10_h3_timing.py bench --tag pri   --binary pristine --arm off --scene "$s" --gpu 0 --out "dumps/h3_$s.json"
  python p10_h3_timing.py parse --scene "$s" "dumps/h3_$s.json" --out "dumps/h3_table_$s.json"
done
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "h4" ]; then
echo "=== H4 mechanism profile ==="
python p10_h4_profile.py ncu_probe --gpu 0 --out dumps/h4_ncu_probe.json
python p10_h4_sass_from_extract.py --dir sass_extract/h8 --so "$H8SO" --arm off --marker Lb0E \
    --out dumps/h4_sass_h8_off.json
python p10_h4_sass_from_extract.py --dir sass_extract/h8 --so "$H8SO" --arm on  --marker Lb1E \
    --out dumps/h4_sass_h8_on.json
python p10_h4_sass_from_extract.py --dir sass_extract/pristine --so "$PRISO" --arm pristine_crosscheck \
    --out dumps/h4_sass_pristine.json
python p10_sass_cmp.py --pri sass_extract/pristine --h8 sass_extract/h8 --arm off --detail \
    --out dumps/h4_sass_fp_cmp_off.json
python p10_sass_cmp.py --pri sass_extract/pristine --h8 sass_extract/h8 --arm on  --detail \
    --out dumps/h4_sass_fp_cmp_on.json
# same-source control: a second compile of the pristine tree. Establishes how much
# opcode-level difference is build noise before the pristine-vs-patched deltas are
# read as a mechanism signal.
if [ -f "$CTLSO" ] && [ ! -d sass_extract/pristine_ctl ]; then
  python p10_sass_extract.py "$CTLSO" sass_extract/pristine_ctl
fi
if [ -f "$CTLSO" ] && [ -d sass_extract/pristine_ctl ]; then
  python p10_sass_cmp.py --pri sass_extract/pristine --h8 sass_extract/pristine_ctl \
      --arm off --detail --out dumps/h4_sass_fp_cmp_control.json
else
  echo "NOTE: pristine control build/extract missing; skipped the noise-floor comparison"
fi
for s in $SCENES; do
  python p10_h4_profile.py torchprof --scene "$s" --gpu 0 --out "dumps/h4_prof_$s.json"
done
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "assemble" ]; then
echo "=== assemble ==="
python - <<'PY'
import json, glob, os
eq = {}
for p in sorted(glob.glob("dumps/EPIC05_P10_EQUIVALENCE_*.json")):
    d = json.load(open(p)); eq[d["scene"]] = d
classes = {s: d["verdict"]["classification"] for s, d in eq.items()}
json.dump({"scenes": eq,
           "per_scene_classification": classes,
           "verdict": {"classification": "PASS_NUMERICALLY_EQUIVALENT" if all(
               c == "PASS_NUMERICALLY_EQUIVALENT" for c in classes.values())
               else "NOT_ALL_PASS",
               "non_pass_scenes": [s for s, c in classes.items()
                                   if c != "PASS_NUMERICALLY_EQUIVALENT"],
               "passed_scenes": [s for s, c in classes.items()
                                 if c == "PASS_NUMERICALLY_EQUIVALENT"]}},
          open("dumps/EPIC05_P10_EQUIVALENCE.json", "w"), indent=2)

t = {}
for p in sorted(glob.glob("dumps/h3_table_*.json")):
    d = json.load(open(p)); t[os.path.basename(p)[len("h3_table_"):-len(".json")]] = d
json.dump(t, open("dumps/EPIC05_P10_TIMING.json", "w"), indent=2)

prof = json.load(open("dumps/h4_prof_room.json"))
for arm in ("off", "on"):
    try:
        prof[f"sass_{arm}"] = json.load(open(f"dumps/h4_sass_h8_{arm}.json"))
    except FileNotFoundError:
        pass
try:
    prof["sass_pristine_crosscheck"] = json.load(open("dumps/h4_sass_pristine.json"))
except FileNotFoundError:
    pass
for arm in ("off", "on", "control"):
    try:
        prof[f"fp_dataflow_cmp_{arm}"] = json.load(open(f"dumps/h4_sass_fp_cmp_{arm}.json"))
    except FileNotFoundError:
        pass
prof["ncu_probe"] = json.load(open("dumps/h4_ncu_probe.json"))
all_tables = {}
for p in sorted(glob.glob("dumps/h4_prof_*.json")):
    s = os.path.basename(p)[len("h4_prof_"):-len(".json")]
    all_tables[s] = json.load(open(p))["kernels"]
prof["all_scene_kernel_tables"] = all_tables
ko, kn = prof["kernels"]["off"], prof["kernels"]["on"]
rows = []
for k in sorted(set(ko) | set(kn), key=lambda x: -(ko.get(x, {}).get("cuda_ms_per_iter", 0))):
    a = ko.get(k, {}).get("cuda_ms_per_iter")
    b = kn.get(k, {}).get("cuda_ms_per_iter")
    rows.append({"kernel": k, "off_ms": a, "on_ms": b,
                 "delta_ms": (b - a) if (a is not None and b is not None) else None,
                 "delta_pct": ((b - a) / a * 100) if (a and b is not None) else None})
prof["kernel_time_table_room"] = rows
prof["total_cuda_ms_per_iter_room"] = {
    "off": sum(v["cuda_ms_per_iter"] for v in ko.values()),
    "on": sum(v["cuda_ms_per_iter"] for v in kn.values()),
}
prof["note"] = ("ncu hardware counters unavailable on this host (driver resource blocked, "
                "see ncu_probe); evidence = full-stream cuobjdump static analysis "
                "(p10_sass_extract + p10_sass_cmp: opcode census, FP opcode-multiset deltas, "
                "with a same-source pristine control build as the noise floor) "
                "+ torch.profiler CUPTI per-kernel CUDA time")
json.dump(prof, open("dumps/EPIC05_P10_KERNEL_PROFILE.json", "w"), indent=2)
print("assembled dumps/EPIC05_P10_EQUIVALENCE.json / _TIMING.json / _KERNEL_PROFILE.json")
PY
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "verdict" ]; then
echo "=== verdict ==="
python p10_verdict.py \
  --h2 dumps/EPIC05_P10_EQUIVALENCE_room.json dumps/EPIC05_P10_EQUIVALENCE_bicycle.json dumps/EPIC05_P10_EQUIVALENCE_garden.json \
  --h3-h8 dumps/h3_verdict_room.json dumps/h3_verdict_bicycle.json dumps/h3_verdict_garden.json \
  --h3-pristine dumps/h3_verdict_room_pri.json dumps/h3_verdict_bicycle_pri.json dumps/h3_verdict_garden_pri.json \
  --h4 dumps/EPIC05_P10_KERNEL_PROFILE.json \
  --out dumps/EPIC05_P10_VERDICT.json
fi

echo "=== DONE ==="
