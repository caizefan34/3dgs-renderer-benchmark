# P2I P0 command record

Remote host: `mx` (`bms-39468022-001`), A100 GPU 4.

```bash
export PYTHONNOUSERSITE=1
export PATH=/home/liaoyuanjun/miniforge3/envs/anysplat/bin:$PATH
export CUDA_VISIBLE_DEVICES=4
export P2I_REMOTE_ROOT=/home/liaoyuanjun/p2i_fusion_measure
export MAX_JOBS=2
/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python p2i_opportunity_gate.py
```

Accepted P0 evidence is JSON, CSV, environment JSON, and `run.log` beside this file.
