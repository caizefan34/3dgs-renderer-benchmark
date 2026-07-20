"""Low-overhead NVML process-memory sampling."""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone


class NvmlProcessMemorySampler:
    def __init__(self, device_index: int = 0, interval_seconds: float = 0.005):
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.available = False
        self.baseline_mb = None
        self.peak_mb = None
        self.peak_delta_mb = None
        self.samples = []
        self._start_perf_ns = None
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None

    def _device_handle(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            return self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if self.device_index >= len(devices):
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not expose the requested device")
        physical = devices[self.device_index]
        if physical.isdigit():
            return self._nvml.nvmlDeviceGetHandleByIndex(int(physical))
        return self._nvml.nvmlDeviceGetHandleByUUID(physical)

    def _usage_mb(self):
        processes = []
        for function_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            function = getattr(self._nvml, function_name, None)
            if function is not None:
                try:
                    processes.extend(function(self._handle))
                except self._nvml.NVMLError:
                    pass
        values = [
            process.usedGpuMemory / (1024 * 1024)
            for process in processes
            if process.pid == os.getpid()
            and process.usedGpuMemory not in (None, getattr(self._nvml, "NVML_VALUE_NOT_AVAILABLE", None))
        ]
        return max(values) if values else 0.0

    def start(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = self._device_handle()
            self.baseline_mb = self._usage_mb()
        except Exception:
            return self
        self.available = True
        self._start_perf_ns = time.perf_counter_ns()
        self.peak_mb = self.baseline_mb
        self._record_sample(self.baseline_mb)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                usage_mb = self._usage_mb()
                self.peak_mb = max(self.peak_mb, usage_mb)
                self._record_sample(usage_mb)
            except Exception:
                self.available = False
                return

    def _record_sample(self, usage_mb=None):
        """Retain one auditable raw process-memory observation."""
        now_perf_ns = time.perf_counter_ns()
        if self._start_perf_ns is None:
            self._start_perf_ns = now_perf_ns
        self.samples.append({
            "relative_ms": (now_perf_ns - self._start_perf_ns) / 1_000_000.0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "used_gpu_memory_mib": float(self._usage_mb() if usage_mb is None else usage_mb),
        })

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        if self.available:
            usage_mb = self._usage_mb()
            self.peak_mb = max(self.peak_mb, usage_mb)
            self._record_sample(usage_mb)
            self.peak_delta_mb = max(0.0, self.peak_mb - self.baseline_mb)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return self.peak_delta_mb
