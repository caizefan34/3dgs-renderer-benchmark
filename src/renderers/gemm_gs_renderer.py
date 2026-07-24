"""Adapter for the pinned GEMM-GS diff-gaussian-compatible extension."""

import json
from importlib import metadata

from .diff_gaussian_renderer import DiffGaussianRenderer


class GemmGSRenderer(DiffGaussianRenderer):
    name = "gemm_gs"
    implementation = "shieldforever/GEMM-GS Tensor Core blending"
    source_url = "https://github.com/shieldforever/GEMM-GS"
    source_commit = "aca61f897f58964ff7204e1e3c6485995b5f212c"

    def is_available(self) -> bool:
        if self._available is None:
            try:
                direct_url = metadata.distribution(self.package_name).read_text("direct_url.json")
                source = json.loads(direct_url or "{}").get("url", "").lower()
                self._available = "gemm" in source
            except (metadata.PackageNotFoundError, json.JSONDecodeError):
                self._available = False
        return self._available

    def metadata(self) -> dict:
        result = super().metadata()
        if self.is_available():
            result["commit_hash"] = self.source_commit
        return result
