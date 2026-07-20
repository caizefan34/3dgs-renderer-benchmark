import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_framework.nvml import NvmlProcessMemorySampler


class NvmlSamplerEvidenceTest(unittest.TestCase):
    def test_visible_cuda_index_resolves_to_physical_nvml_device(self):
        sampler = NvmlProcessMemorySampler()
        sampler._nvml = mock.Mock()

        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3,5"}):
            sampler._device_handle()

        sampler._nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(3)

    def test_visible_cuda_uuid_resolves_with_nvml_uuid(self):
        sampler = NvmlProcessMemorySampler()
        sampler._nvml = mock.Mock()

        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-example"}):
            sampler._device_handle()

        sampler._nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-example")

    def test_sample_records_auditable_timestamp_pid_and_mib(self):
        sampler = NvmlProcessMemorySampler()
        sampler._start_perf_ns = 1_000_000_000
        sampler._usage_mb = mock.Mock(return_value=321.5)

        with mock.patch("benchmark_framework.nvml.time.perf_counter_ns", return_value=1_012_500_000), \
             mock.patch("benchmark_framework.nvml.os.getpid", return_value=4242):
            sampler._record_sample()

        sample = sampler.samples[0]
        self.assertEqual(sample["relative_ms"], 12.5)
        self.assertEqual(sample["pid"], 4242)
        self.assertEqual(sample["used_gpu_memory_mib"], 321.5)
        self.assertIsNotNone(datetime.fromisoformat(sample["timestamp_utc"]).tzinfo)


if __name__ == "__main__":
    unittest.main()
