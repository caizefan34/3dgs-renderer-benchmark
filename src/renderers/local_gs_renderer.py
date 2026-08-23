"""Adapter for the pinned Local-GS/TiCoGS rasterizer environment."""

import inspect

from .speedy_splat_renderer import SpeedySplatRenderer


class LocalGSRenderer(SpeedySplatRenderer):
    name = "local_gs"
    package_name = "diff-gaussian-rasterization"
    module_name = "diff_gaussian_rasterization"
    implementation = "tilaba/Local-GS tile-local coherent rasterizer"
    source_url = "https://github.com/tilaba/Local-GS"
    source_commit = "0c6d9e4a2cc458de90d3dc40753187d6d03ea514"

    @staticmethod
    def _backend():
        import diff_gaussian_rasterization
        return diff_gaussian_rasterization

    def is_available(self) -> bool:
        if self._available is None:
            try:
                backend = self._backend()
                parameters = inspect.signature(backend.GaussianRasterizer.forward).parameters
                self._available = "scores" in parameters
            except (ImportError, OSError, ValueError):
                self._available = False
        return self._available

    def metadata(self) -> dict:
        result = super().metadata()
        if self.is_available():
            result["commit_hash"] = self.source_commit
        return result
