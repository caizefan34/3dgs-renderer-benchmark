#!/bin/bash
# Identify the co-tenant and check load stability.
echo "=== compute apps ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
echo "=== any foreign python? ==="
ps aux 2>/dev/null | grep -v grep | grep -E "python|train" | grep -v vos_train_driver | head -8
echo "=== load ==="
uptime
echo "=== my jobs ==="
ps aux 2>/dev/null | grep vos_train_driver | grep -v grep | awk '{print $2, $14, $16}' | head -4
