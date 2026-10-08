#!/bin/bash
cd /root/epic05_p13x
/root/miniforge3/envs/gsplat/bin/python -m py_compile p13x_diag.py publication_trainer_p13x.py p13x_preflight.py && echo "COMPILE_OK"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
