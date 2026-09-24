# Reproducibility commands

Run on an idle A100 (the selected final run used physical GPU 5):

```bash
cd /home/liaoyuanjun/warp_emit_a0
CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/liaoyuanjun/warp_emit_a0/candidate_pkg \
TORCH_EXTENSIONS_DIR=/home/liaoyuanjun/warp_emit_a0/torch_extensions \
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=10 \
/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python warp_emit_pipeline.py
```

The production-baseline comparison is retained in the frozen P2I component
measurement (`../../../p2i/raw/p2i_opportunity_gate.json`). The independent
serial-vs-warp test is:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONNOUSERSITE=1 \
/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python warp_emit_w1.py
```

For the two-camera parity gate, first use the installed baseline package, then
the isolated candidate package:

```bash
python warp_emit_correctness.py baseline
PYTHONPATH=/home/liaoyuanjun/warp_emit_a0/candidate_pkg \
TORCH_EXTENSIONS_DIR=/home/liaoyuanjun/warp_emit_a0/torch_extensions \
python warp_emit_correctness.py candidate
```

The candidate package is a scratch copy of gsplat. `csrc.so` was retained under
the reversible name `csrc.so.baseline_binary`; only `IntersectTile.cu` was
patched, using the four patch files adjacent to this experiment.
