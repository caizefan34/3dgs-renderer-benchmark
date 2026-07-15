"""Optional hardware profiling helpers."""
from __future__ import annotations

from typing import Mapping


def summarize_hardware_profile(profile: Mapping) -> dict:
    """Normalize optional profiler metrics without affecting core benchmark timing."""
    return {
        "schema_version": 1,
        "optional": True,
        "sm_utilization_pct": profile.get("sm_utilization_pct"),
        "occupancy_pct": profile.get("occupancy_pct"),
        "dram_throughput_gbps": profile.get("dram_throughput_gbps"),
        "l2_hit_rate_pct": profile.get("l2_hit_rate_pct"),
        "kernel_breakdown": profile.get("kernel_breakdown", []),
    }

