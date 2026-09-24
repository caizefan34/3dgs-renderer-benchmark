#!/bin/bash
# Robust launcher for the FINAL-30K scheduler (survives ssh disconnect).
exec /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python \
    /mnt/storage_pool/liaoyuanjun/final30k_scheduler.py \
    --max-wait-hours 48 --max-total-hours 96 --poll-s 60 \
    >> /mnt/storage_pool/liaoyuanjun/final30k_scheduler_stdout.log 2>&1
