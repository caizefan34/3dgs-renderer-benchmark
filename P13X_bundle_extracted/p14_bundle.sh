#!/bin/bash
set -e
cd /root/epic05_p14
tar czf /root/epic05_p14_bundle.tar.gz \
    --exclude='raw/P14_VC_room.npz' \
    --exclude='raw/P14_VC_bicycle.npz' \
    artifacts P14_O1_UNITS P14_O2_UNITS logs raw scripts
ls -la /root/epic05_p14_bundle.tar.gz
tar tzf /root/epic05_p14_bundle.tar.gz | wc -l
