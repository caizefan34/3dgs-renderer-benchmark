"""Check zero-gradient Adam catch-up against dense PyTorch Adam.

This checks one mathematical prerequisite for a visibility-lazy 3DGS optimizer.
It says nothing about conservative visibility, GPU cost, or quality.
"""

import json
import math
import random

import torch


def scalar_adam_step(theta, m, v, gradient, step, lr, beta1, beta2, eps):
    m = beta1 * m + (1.0 - beta1) * gradient
    v = beta2 * v + (1.0 - beta2) * gradient * gradient
    theta -= lr * (m / (1.0 - beta1**step)) / (
        math.sqrt(v / (1.0 - beta2**step)) + eps
    )
    return theta, m, v


def zero_gradient_displacement_bound(m, v, start_step, gap, max_lr, beta1, beta2, eps):
    geometric_sum = beta1 * (1.0 - beta1**gap) / (1.0 - beta1)
    smallest_rms = math.sqrt(beta2**gap * v / (1.0 - beta2 ** (start_step + gap)))
    return (
        max_lr * abs(m) * geometric_sum
        / ((1.0 - beta1 ** (start_step + 1)) * (smallest_rms + eps))
    )


def main():
    torch.set_num_threads(1)
    rng = random.Random(20260927)
    size, steps = 64, 500
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    initial = [rng.uniform(-2.0, 2.0) for _ in range(size)]
    parameter = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.01, betas=(beta1, beta2), eps=eps,
        foreach=False, fused=False,
    )
    lazy_theta = initial[:]
    lazy_m = [0.0] * size
    lazy_v = [0.0] * size
    last_step = [0] * size
    learning_rates = [0.0] + [0.02 * (0.04 ** (t / steps)) for t in range(1, steps + 1)]
    observed_updates = 0

    for step in range(1, steps + 1):
        gradients = [rng.gauss(0.0, 0.3) if rng.random() < 0.25 else 0.0 for _ in range(size)]
        optimizer.param_groups[0]["lr"] = learning_rates[step]
        parameter.grad = torch.tensor(gradients, dtype=torch.float64)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for i, gradient in enumerate(gradients):
            if gradient == 0.0:
                continue
            for skipped in range(last_step[i] + 1, step):
                lazy_theta[i], lazy_m[i], lazy_v[i] = scalar_adam_step(
                    lazy_theta[i], lazy_m[i], lazy_v[i], 0.0,
                    skipped, learning_rates[skipped], beta1, beta2, eps,
                )
            lazy_theta[i], lazy_m[i], lazy_v[i] = scalar_adam_step(
                lazy_theta[i], lazy_m[i], lazy_v[i], gradient,
                step, learning_rates[step], beta1, beta2, eps,
            )
            last_step[i] = step
            observed_updates += 1

    for i in range(size):
        for skipped in range(last_step[i] + 1, steps + 1):
            lazy_theta[i], lazy_m[i], lazy_v[i] = scalar_adam_step(
                lazy_theta[i], lazy_m[i], lazy_v[i], 0.0,
                skipped, learning_rates[skipped], beta1, beta2, eps,
            )
    state = optimizer.state[parameter]
    errors = {
        "theta": float((parameter.detach() - torch.tensor(lazy_theta, dtype=torch.float64)).abs().max()),
        "first_moment": float((state["exp_avg"] - torch.tensor(lazy_m, dtype=torch.float64)).abs().max()),
        "second_moment": float((state["exp_avg_sq"] - torch.tensor(lazy_v, dtype=torch.float64)).abs().max()),
    }
    max_bound_ratio = 0.0
    for _ in range(10000):
        start = rng.randrange(0, 20000)
        gap = rng.randrange(1, 101)
        m = rng.uniform(-1.0, 1.0)
        v = 10.0 ** rng.uniform(-20.0, 1.0)
        lrs = [10.0 ** rng.uniform(-5.0, -2.0) for _ in range(gap)]
        theta = 0.0
        for j, lr in enumerate(lrs, start=1):
            theta, m_after, v_after = scalar_adam_step(
                theta, m if j == 1 else m_after,
                v if j == 1 else v_after,
                0.0, start + j, lr, beta1, beta2, eps,
            )
        bound = zero_gradient_displacement_bound(
            m, v, start, gap, max(lrs), beta1, beta2, eps
        )
        ratio = abs(theta) / bound if bound else 0.0
        max_bound_ratio = max(max_bound_ratio, ratio)
        assert abs(theta) <= bound * (1.0 + 1e-12) + 1e-15

    result = {
        "coordinates": size,
        "optimizer_steps": steps,
        "observed_coordinate_updates": observed_updates,
        "observed_fraction": observed_updates / (size * steps),
        "max_absolute_errors": errors,
        "zero_gradient_bound_trials": 10000,
        "max_actual_to_bound_ratio": max_bound_ratio,
        "meaning": "CPU FP64 scalar Adam and displacement-bound checks only",
    }
    print(json.dumps(result, sort_keys=True))
    assert max(errors.values()) < 1e-12, result


if __name__ == "__main__":
    main()
