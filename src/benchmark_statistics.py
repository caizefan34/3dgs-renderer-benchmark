"""Repeat-level statistics for benchmark throughput measurements."""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


_T_CRITICAL_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def summarize_repeat_throughput(repeats: Sequence[Sequence[float]]) -> dict:
    """Summarize FPS with repeats, rather than correlated frames, as replicates."""
    if not repeats or not repeats[0]:
        raise ValueError("at least one non-empty repeat is required")
    frames_per_repeat = len(repeats[0])
    if any(len(repeat) != frames_per_repeat for repeat in repeats):
        raise ValueError("all repeats must contain the same number of frames")

    normalized = [[float(value) for value in repeat] for repeat in repeats]
    if any(not math.isfinite(value) or value <= 0 for repeat in normalized for value in repeat):
        raise ValueError("frame times must be finite positive numbers")

    repeat_fps = [frames_per_repeat * 1000.0 / sum(repeat) for repeat in normalized]
    repeat_mean = statistics.mean(repeat_fps)
    pooled_fps = len(normalized) * frames_per_repeat * 1000.0 / sum(
        sum(repeat) for repeat in normalized
    )
    if len(repeat_fps) > 1:
        repeat_stdev = statistics.stdev(repeat_fps)
        critical = _T_CRITICAL_95.get(len(repeat_fps), 1.96)
        margin = critical * repeat_stdev / math.sqrt(len(repeat_fps))
        repeat_cv = repeat_stdev / repeat_mean
    else:
        margin = 0.0
        repeat_cv = 0.0

    return {
        "pooled_fps": pooled_fps,
        "repeat_mean_fps": repeat_mean,
        "repeat_median_fps": statistics.median(repeat_fps),
        "repeat_cv": repeat_cv,
        "fps_ci95_low": max(0.0, repeat_mean - margin),
        "fps_ci95_high": repeat_mean + margin,
        "ci95_method": "student_t_over_repeat_fps",
        "repeat_count": len(repeat_fps),
        "frames_per_repeat": frames_per_repeat,
    }
