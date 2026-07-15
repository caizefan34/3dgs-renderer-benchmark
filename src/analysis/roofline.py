"""Optional roofline-style workload classification."""
from __future__ import annotations


def roofline_classification(
    arithmetic_intensity: float,
    memory_bandwidth_utilization: float,
    compute_utilization: float,
) -> dict:
    if arithmetic_intensity < 0 or memory_bandwidth_utilization < 0 or compute_utilization < 0:
        raise ValueError("Roofline inputs must be non-negative")
    if memory_bandwidth_utilization >= 70 and compute_utilization < 70:
        classification = "memory-bound"
    elif compute_utilization >= 70 and memory_bandwidth_utilization < 70:
        classification = "compute-bound"
    else:
        classification = "mixed"
    return {
        "schema_version": 1,
        "arithmetic_intensity": arithmetic_intensity,
        "memory_bandwidth_utilization_pct": memory_bandwidth_utilization,
        "compute_utilization_pct": compute_utilization,
        "classification": classification,
    }

