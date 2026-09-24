#!/bin/bash
# Remote R3 completion monitor - runs as background job on mx
REPO="$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)"
R3DIR="$REPO/results/reference_v1/r3"
MARKER="$R3DIR/COMPLETE.marker"
LOG="$R3DIR/monitor.log"

echo "Monitor started at $(date)" >> "$LOG"

while true; do
    if [ -f "$MARKER" ]; then
        echo "Already complete at $(date)" >> "$LOG"
        exit 0
    fi
    if ! pgrep -f "r3_certificate_runner" > /dev/null 2>&1; then
        sleep 30
        if ! pgrep -f "r3_certificate_runner" > /dev/null 2>&1; then
            echo "Process finished at $(date)" >> "$LOG"
            sleep 10
            for W in 5000 15000 30000; do
                if [ ! -f "$R3DIR/$W/runner.log" ]; then
                    echo "WARNING: $W/runner.log missing" >> "$LOG"
                fi
            done
            touch "$MARKER"
            echo "COMPLETE marker created at $(date)" >> "$LOG"
            exit 0
        fi
    fi
    sleep 60
done
