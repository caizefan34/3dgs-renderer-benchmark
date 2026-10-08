# Attempt 001 startup abort (2026-09-27)

- The frozen author training command was launched on physical A100 GPU3 UUID `GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72` at 2026-09-27T09:16:58.502337Z.
- Supervisor terminated only the newly launched process group after 6.056 s (phase wall); child return code `-15` (SIGTERM). No training iteration completed: `logs/train.log` is empty; output contains only `input.ply`, `cfg_args`, and `cameras.json` from scene initialization.
- Stop reason was a monitor PID attribution error: `nvidia-smi` reported compute PID 2197703 (434 MiB), while the container process handle reported PID 2600732. EPIC-05 host/container PID namespaces differ, so the monitor incorrectly classified the CUDA context as a foreign process. At that sample GPU memory was ~423 MiB, far below the 73,728 MiB memory gate. Device returned to 0 MiB/0% after exit.
- This was a supervisor false positive, not a training/resource failure. Original output, logs, samples, and hashes are retained under `/root/sadgs_room_author_3k_20260927`; no files there will be overwritten by attempt 002.
