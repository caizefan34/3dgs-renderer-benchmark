"""Checkpoint save/load for 3DGS training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch


class TrainingCheckpoint:
    """Manages training state checkpointing.

    Saves/loads model state, optimizer state, and training metadata.
    """

    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        iteration: int,
        model_state: Dict,
        optimizer_state: Dict,
        metrics: Optional[Dict] = None,
        label: str = "checkpoint",
    ):
        """Save a training checkpoint.

        Args:
            iteration: Current training iteration
            model_state: From model.get_checkpoint_state()
            optimizer_state: optimizer.state_dict()
            metrics: Optional dict of current metrics
            label: Checkpoint label (e.g., "latest", "best_psnr")
        """
        filename = f"{label}_iter{iteration}.pt"
        path = self.save_dir / filename

        data = {
            "iteration": iteration,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metrics": metrics or {},
            "format_version": 1,
        }
        torch.save(data, path)

        # Also save a plain "latest.pt" that is overwritten each time
        latest_path = self.save_dir / f"{label}_latest.pt"
        torch.save(data, latest_path)

        return path

    def load(
        self,
        iteration: Optional[int] = None,
        label: str = "checkpoint",
    ) -> Optional[Dict]:
        """Load the most recent or specific checkpoint.

        Args:
            iteration: Specific iteration to load
            label: Checkpoint label prefix

        Returns:
            Checkpoint data dict, or None if not found
        """
        if iteration is not None:
            path = self.save_dir / f"{label}_iter{iteration}.pt"
        else:
            path = self.save_dir / f"{label}_latest.pt"

        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)
