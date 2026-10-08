"""Check the measured checkpoint's newborn-row Adam bias-correction effect.

This establishes an update-size effect, not a bug, novelty or training benefit.
The step/betas below were read from all five parameter groups in the actual
room@2000 checkpoint. Its SHA is taken from the matching real probe artifact.
"""

import hashlib
import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).parent
    probe = json.loads((root / 'room_2000_topology.json').read_text())
    step = 1985
    beta1, beta2 = 0.9, 0.999
    lr = 0.001
    updates = []
    for age in (0, step):
        p = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        optimizer = torch.optim.Adam([p], lr=lr, betas=(beta1, beta2), eps=1e-15)
        optimizer.state[p] = {'step': torch.tensor(float(age)),
                              'exp_avg': torch.zeros_like(p),
                              'exp_avg_sq': torch.zeros_like(p)}
        p.grad = torch.tensor([0.1, -0.2, 0.3], dtype=p.dtype)
        optimizer.step()
        updates.append(p.detach().abs() / lr)
    predicted = (1-beta1)/(1-beta1**(step+1)) * ((1-beta2**(step+1))/(1-beta2))**0.5
    measured = updates[1] / updates[0]
    assert torch.max(torch.abs(measured-predicted)) < 1e-10
    result = {
        'checkpoint_sha256': probe['checkpoint_sha256'],
        'checkpoint_iteration': 2000, 'observed_group_step': step,
        'observed_groups': ['xyz', 'shs', 'opacity', 'scaling', 'rotation'],
        'betas': [beta1, beta2], 'eps': 1e-15,
        'fresh_row_abs_update_over_lr': updates[0].tolist(),
        'zero_moments_old_step_abs_update_over_lr': updates[1].tolist(),
        'predicted_first_update_ratio_ignoring_eps': predicted,
        'measured_ratio': measured.tolist(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'CPU Adam arithmetic plus actual checkpoint step metadata; no claim that age correction improves 3DGS',
        'training_effect_verified': False, 'novelty_established': False,
    }
    (root / 'newborn_adam_evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
