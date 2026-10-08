"""Algebra/representation diagnostic for finite-edit capacity allocation.

CPU FP64, not a trainer or a benchmark. The 1D moment calculation is an
additive normalized-density example, deliberately distinct from alpha blending.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def residualize(j, x, damping=0.0):
    if damping == 0:
        return x - j @ np.linalg.lstsq(j, x, rcond=None)[0]
    return x - j @ np.linalg.solve(j.T @ j + damping * np.eye(j.shape[1]), j.T @ x)


def finite_edit_gain(r, j, b, damping=0.0):
    """Signed gain after the SAME unconstrained linear repair in both arms.

    b is a complete finite image edit, not a derivative or an optimizable
    mixture coefficient. This proxy cannot implement trust-region bounds.
    """
    return float(-b @ residualize(j, r, damping) - 0.5 * b @ residualize(j, b, damping))


def direct_minimum(r, j, damping):
    augmented_j = np.vstack((j, np.sqrt(damping) * np.eye(j.shape[1])))
    augmented_r = np.concatenate((r, np.zeros(j.shape[1])))
    d = np.linalg.lstsq(augmented_j, -augmented_r, rcond=None)[0]
    return float(0.5 * np.sum((r + j @ d) ** 2) + 0.5 * damping * (d @ d))


def normal_pdf(x, mean, variance):
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def moment_probe():
    x = np.linspace(-8, 8, 16001)
    parent = normal_pdf(x, 0, 1)
    steps = np.array([0.4, 0.2, 0.1, 0.05])
    errors, raw_errors = [], []
    fourth = (x ** 4 - 6 * x ** 2 + 3) * parent
    limits = []
    for h in steps:
        children = 0.5 * (normal_pdf(x, h, 1 - h * h) + normal_pdf(x, -h, 1 - h * h))
        raw = 0.5 * (normal_pdf(x, h, 1) + normal_pdf(x, -h, 1))
        delta = children - parent
        errors.append(float(np.linalg.norm(delta)))
        raw_errors.append(float(np.linalg.norm(raw - parent)))
        limits.append(float(np.linalg.norm(delta / h ** 4 + fourth / 12) / np.linalg.norm(fourth / 12)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    raw_orders = np.log2(np.array(raw_errors[:-1]) / np.array(raw_errors[1:]))
    return {'steps': steps.tolist(), 'moment_preserving_orders': orders.tolist(),
            'fixed_covariance_orders': raw_orders.tolist(), 'fourth_order_limit_relative_error': limits,
            'scope': 'Normalized additive 1D Gaussian density; not 3DGS image compositing.'}


def run():
    rng = np.random.default_rng(20260926)
    max_error = 0.0
    for damping in (0.0, 0.3):
        for _ in range(50):
            j, r, b = rng.normal(size=(32, 7)), rng.normal(size=32), rng.normal(size=32)
            direct = direct_minimum(r, j, damping) - direct_minimum(r + b, j, damping)
            max_error = max(max_error, abs(direct - finite_edit_gain(r, j, b, damping)))
    assert max_error < 1e-10

    # Same edit offers no new direction if existing parameters can make it.
    j = np.array([[1.0], [0.0], [0.0]])
    r = np.array([-1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    raw_gain = float(-b @ r - 0.5 * b @ b)
    conditional_gain = finite_edit_gain(r, j, b)
    assert raw_gain == 0.5 and abs(conditional_gain) < 1e-12

    # Positive correlation alone does not accept an edit that overshoots.
    r = np.array([0.0, -1.0, 0.0])
    overshoot = np.array([0.0, 3.0, 0.0])
    assert -overshoot @ r > 0
    overshoot_gain = finite_edit_gain(r, j, overshoot)
    assert overshoot_gain < 0

    # Omitting a repair direction creates a false capacity deficit.
    full_j = np.eye(3)[:, :2]
    useful_edit = np.array([0.0, 1.0, 0.0])
    missing_mode_gain = finite_edit_gain(r, j, useful_edit)
    complete_mode_gain = finite_edit_gain(r, full_j, useful_edit)
    assert missing_mode_gain == 0.5 and abs(complete_mode_gain) < 1e-12

    moments = moment_probe()
    assert 3.98 < moments['moment_preserving_orders'][-1] < 4.05
    assert 1.98 < moments['fixed_covariance_orders'][-1] < 2.02
    assert moments['fourth_order_limit_relative_error'][-1] < 0.01

    # At zero displacement, halving opacity is NOT render preserving.
    profile = np.array([1.0, 0.5, 0.1])
    opacity = 0.6
    original_alpha = opacity * profile
    half_blended = 1 - (1 - opacity * profile / 2) ** 2
    assert np.allclose(original_alpha - half_blended, opacity ** 2 * profile ** 2 / 4)
    peak_child_opacity = 1 - np.sqrt(1 - opacity)
    peak_matched = 1 - (1 - peak_child_opacity * profile) ** 2
    assert abs(peak_matched[0] - original_alpha[0]) < 1e-12
    assert np.max(np.abs(peak_matched[1:] - original_alpha[1:])) > 0.01

    result = {
        'scope': 'Algebra and controlled representation counterexamples only; no 3DGS training.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks_passed': ['100 finite-edit identities', 'tangent redundancy', 'signed overshoot rejection',
                          'incomplete-basis false positive', 'fourth-order density response', 'alpha non-preservation'],
        'max_finite_edit_identity_error': max_error,
        'raw_vs_conditional_tangent_gain': [raw_gain, conditional_gain],
        'overshoot_gain': overshoot_gain,
        'incomplete_vs_complete_basis_gain': [missing_mode_gain, complete_mode_gain],
        'moment_probe': moments,
        'alpha_probe': {'opacity': opacity, 'profile': profile.tolist(),
                        'halved_opacity_alpha_error': (half_blended - original_alpha).tolist(),
                        'peak_matched_alpha_error': (peak_matched - original_alpha).tolist()},
        'end_to_end_speedup_verified': False, 'novelty_established': False,
    }
    destination = Path(__file__).with_name('topology_evidence.json')
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
