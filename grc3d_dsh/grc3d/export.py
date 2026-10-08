"""Call from an existing trusted trainer after restoring its native model/camera."""
from pathlib import Path
import torch


@torch.no_grad()
def snapshot(model):
    """Graphdeco-style GaussianModel raw optimization variables, not activated values.

    DSH must verify these fields against the actual model implementation. Restore
    checkpoints through the existing trusted trainer; this module does not guess
    arbitrary checkpoint schemas or unpickle remote objects.
    """
    fields = {"means": "_xyz", "quats": "_rotation", "log_scales": "_scaling",
              "opacity_logits": "_opacity", "sh0": "_features_dc", "shN": "_features_rest"}
    out = {k: getattr(model, attr).detach().cpu().clone() for k, attr in fields.items()}
    out["opacity_logits"] = out["opacity_logits"].reshape(-1)
    if out["sh0"].shape[1:] != (1, 3) or out["shN"].ndim != 3 or out["shN"].shape[-1] != 3:
        raise ValueError("SH must use [N, K, 3]; audit source and adapt export explicitly")
    return out


@torch.no_grad()
def save_bundle(path, current_params, historical_params, viewmat, K, target_hwc,
                sh_degree, metadata):
    """Historical state and current state must render the SAME camera and SH degree.

    Both states may have different topology; image anchors do not require an
    index correspondence. Ground truth must already match the trainer's exact
    resolution, color space and background policy. Only black background supported.
    """
    h, w, c = target_hwc.shape
    if c != 3:
        raise ValueError("Expected HWC RGB ground truth")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"params": current_params, "anchor_params": historical_params,
                "camera": {"viewmat": viewmat.detach().cpu(), "K": K.detach().cpu(),
                           "height": h, "width": w},
                "target": target_hwc.detach().cpu(), "sh_degree": int(sh_degree),
                "metadata": metadata}, path)
