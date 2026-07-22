"""Interfaces for tracked renderer candidates without local adapters."""
from __future__ import annotations

from typing import Any

import torch

from .base import RendererAdapter


class MissingRendererAdapter(RendererAdapter):
    """Registered placeholder for a renderer whose backend is not installed."""

    name = "missing_renderer"
    implementation = "missing_renderer"
    source_url = ""
    install_hint = ""

    def is_available(self) -> bool:
        return False

    def prepare_scene(self, scene_data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.name} is tracked but no local adapter/backend is installed. "
            f"{self.install_hint}"
        )

    def render(self, scene_data: dict[str, Any], camera: Any) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.name} is tracked but no local adapter/backend is installed. "
            f"{self.install_hint}"
        )


class LocalGSRenderer(MissingRendererAdapter):
    name = "local_gs"
    implementation = "Local-GS / TiCoGS"
    source_url = "https://github.com/tilaba/Local-GS"
    install_hint = "Install Local-GS/TiCoGS and replace this stub with a real adapter."


class GemmGSRenderer(MissingRendererAdapter):
    name = "gemm_gs"
    implementation = "GEMM-GS"
    source_url = "https://github.com/shieldforever/GEMM-GS"
    install_hint = "Install GEMM-GS and replace this stub with a real adapter."


class StopThePopRenderer(MissingRendererAdapter):
    name = "stopthepop"
    implementation = "StopThePop"
    source_url = "https://github.com/r4dl/StopThePop"
    install_hint = "Install StopThePop and implement a view-consistency adapter."
