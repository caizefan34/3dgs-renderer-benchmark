"""CPU-only preflight: no MX GPU, CUDA build, checkpoints, or gsplat required."""
from __future__ import annotations

import math
import torch
from hwcs_mask_oracle import generate_common_uniforms, make_mask, custom_hwcs_mask


def run() -> None:
    torch.set_num_threads(1)
    h, w, p = 65, 77, 0.25  # deliberately non-multiples of 4/16
    u = generate_common_uniforms(h, w, seed=24601)
    output = {method: make_mask(h, w, u, method, p=p) for method in ("G16", "G4", "HWCS1", "HWCS2")}
    for method, (m, meta) in output.items():
        assert m.shape == (h, w) and m.dtype == torch.float32
        assert torch.all((m == 0) | (m == 1/p))
        assert meta["selected_pixel_count"] == int((m > 0).sum())
        assert meta["selected_child_count"] <= math.ceil(h/4)*math.ceil(w/4)
        assert meta["evidence_class"] == "sampling_mask_only_NOT_GPU_WORK"
        print(f"SHAPE/DEVICE PASS {method} {h}x{w} selected_pixels={meta['selected_pixel_count']}")

    assert torch.equal(output["G16"][0], custom_hwcs_mask(h, w, u, q=p, r=1.0))
    assert torch.equal(output["G4"][0], custom_hwcs_mask(h, w, u, q=1.0, r=p))
    print("ENDPOINT PASS G16=(q=p,r=1) and G4=(q=1,r=p)")

    # Analytic variance identity for scalar gradient contributions; independent
    # Bernoulli over parents and children, with signed contributions allowed.
    # Multiple parents, 16 children/parent, deterministic reproducible array.
    a = torch.tensor([[((i * 17 + j * 7) % 23 - 11) / 13.0
                       for j in range(16)] for i in range(6)], dtype=torch.float64)
    s1 = a.square().sum().item()
    s2 = a.sum(dim=1).square().sum().item()
    v_g4 = (1-p)/p*s1
    v_g16 = (1-p)/p*s2
    gen = torch.Generator(device="cpu"); gen.manual_seed(1234)
    n = 160000
    # Vectorized MC on a tiny synthetic scalar gradient model.
    up = torch.rand((n, 6), generator=gen)
    uc = torch.rand((n, 6, 16), generator=gen)
    true_grad = a.sum().item()
    for method, q, r in (("G16",p,1.0),("G4",1.0,p),("HWCS1",.5,.5),("HWCS2",.75,1/3)):
        parent = up < q
        child = uc < r
        xi = parent.unsqueeze(-1) & child
        est = (xi.to(a.dtype)*a).sum((1,2))/p
        empirical_mean = est.mean().item()
        empirical_variance = ((est - true_grad)**2).mean().item()
        alpha = r*(1-q)/(1-p)
        theoretical_variance = (1-alpha)*v_g4 + alpha*v_g16
        rel_err = abs(empirical_variance-theoretical_variance)/max(theoretical_variance,1e-12)
        assert abs(empirical_mean-true_grad) < 0.16, (method, empirical_mean, true_grad)
        assert rel_err < 0.045, (method, rel_err)
        print(f"MC PASS {method} alpha={alpha:.6f} rel_var_err={rel_err:.3%} abs_mean_err={abs(empirical_mean-true_grad):.4f}")
    print("ALL PREFLIGHT TESTS PASSED")


if __name__ == "__main__":
    run()
