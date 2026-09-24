# Optional Research Extensions

Hardware profiling and roofline analysis are optional. They must remain
separate from core timing so profiler overhead does not bias published FPS.

Supported optional fields:

- SM utilization;
- occupancy;
- DRAM throughput;
- L2 hit rate;
- kernel breakdown;
- arithmetic intensity;
- memory bandwidth utilization;
- compute utilization;
- roofline class: memory-bound, compute-bound, or mixed.

Example roofline classification:

```text
python src/scripts/roofline_analysis.py \
  --arithmetic-intensity 4.2 \
  --memory-bandwidth-utilization 78 \
  --compute-utilization 45 \
  --output results/roofline/example.json
```

