"""Independent scalar RGB/VJP check for a fixed-support split shadow pass.

This is an optical compositor oracle, not FastGS CUDA parity or a cost result.
"""

import json

import torch


torch.set_default_dtype(torch.float64)
CUTOFF = 1 / 255
EARLY_STOP = 1e-4


def alpha_at(raw: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(exponent * torch.log1p(-raw))


def render(layers: list[tuple[float, str, torch.Tensor, torch.Tensor]],
           gate: torch.Tensor, bg: torch.Tensor) -> tuple[torch.Tensor, list]:
    rgb = torch.zeros(3)
    trans = torch.ones(())
    accepted = []
    for _, role, raw, color in sorted(layers, key=lambda layer: layer[0]):
        if raw.item() < CUTOFF:
            continue  # Ungated, fixed candidate support.
        exponent = 1 - gate if role == "parent" else gate if role == "child" else torch.ones(())
        alpha = alpha_at(raw, exponent)
        next_trans = trans * (1 - alpha)
        if next_trans.item() < EARLY_STOP:
            break  # Native exclusive early-stop rule for this fixed traversal.
        accepted.append((role, raw, color, alpha, trans))
        rgb = rgb + trans * alpha * color
        trans = next_trans
    return rgb + trans * bg, accepted


def shadow_vjp(accepted: list, bg: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    suffix = bg
    derivatives = []
    for role, raw, color, alpha, trans in reversed(accepted):
        suffix_after = suffix
        if role == "parent":
            d_alpha = (1 - raw) * torch.log1p(-raw)
        elif role == "child":
            d_alpha = -torch.log1p(-raw)
        else:
            d_alpha = torch.zeros(())
        derivatives.append(trans * (color - suffix_after) * d_alpha)
        suffix = alpha * color + (1 - alpha) * suffix
    return (torch.stack(derivatives).sum(0) * vector).sum() if derivatives else torch.zeros(())


def main() -> None:
    torch.manual_seed(20260927)
    max_birth_error = max_release_error = max_vjp_error = 0.0
    early_exit_cases = 0
    child_after_parent_cases = 0
    for trial in range(2500):
        depth = torch.rand(5).tolist()
        raw = torch.rand(5) * 0.989 + 0.001
        if trial >= 2000:
            raw = torch.rand(5) * 0.08 + 0.91
        if trial == 2499:
            raw = torch.tensor([0.99, 0.95, 0.97, 0.98, 0.98])
            depth = [0.3, 0.1, 0.9, 0.0, 0.2]
        color = torch.rand(5, 3)
        bg = torch.rand(3)
        vector = torch.randn(3)
        roles = ["parent", "child", "child", "other", "other"]
        layers = [(depth[i], roles[i], raw[i], color[i]) for i in range(5)]
        gate = torch.tensor(0.0, requires_grad=True)
        birth, accepted = render(layers, gate, bg)
        release, _ = render(layers, torch.ones(()), bg)
        native_before, _ = render([row for row in layers if row[1] != "child"], torch.zeros(()), bg)
        native_after, _ = render([row for row in layers if row[1] != "parent"], torch.ones(()), bg)
        if birth.requires_grad:
            vjp_autograd = torch.autograd.grad((birth * vector).sum(), gate, allow_unused=True)[0]
            if vjp_autograd is None:
                vjp_autograd = torch.zeros(())
        else:
            vjp_autograd = torch.zeros(())
        vjp_shadow = shadow_vjp(accepted, bg, vector)
        max_birth_error = max(max_birth_error, (birth - native_before).abs().max().item())
        max_release_error = max(max_release_error, (release - native_after).abs().max().item())
        max_vjp_error = max(max_vjp_error, abs((vjp_autograd - vjp_shadow).item()))
        if len(accepted) < sum(row[2].item() >= CUTOFF for row in layers):
            early_exit_cases += 1
        child_after_parent_cases += int(any(row[0] > depth[0] and row[1] == "child" for row in layers))

    result = {
        "trials": 2500,
        "fixed_support_cutoff": CUTOFF,
        "exclusive_early_stop": EARLY_STOP,
        "early_exit_cases": early_exit_cases,
        "child_after_parent_cases": child_after_parent_cases,
        "max_birth_rgb_error": max_birth_error,
        "max_release_rgb_error": max_release_error,
        "max_shadow_vjp_error": max_vjp_error,
    }
    if max(max_birth_error, max_release_error, max_vjp_error) > 1e-12:
        raise AssertionError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
