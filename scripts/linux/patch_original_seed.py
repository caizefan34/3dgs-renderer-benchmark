#!/usr/bin/env python3
from pathlib import Path
repo = Path('/root/renderer-candidates/original_3dgs_train')
train = repo / 'train.py'
gu = repo / 'utils' / 'general_utils.py'
t = train.read_text(encoding='utf-8')
g = gu.read_text(encoding='utf-8')
old_arg = 'parser.add_argument("--start_checkpoint", type=str, default = None)'
new_arg = old_arg + '\n    parser.add_argument(\'--seed\', type=int, default=0)'
assert t.count(old_arg) == 1, ('arg anchor', t.count(old_arg))
t = t.replace(old_arg, new_arg)
assert t.count('safe_state(args.quiet)') == 1
t = t.replace('safe_state(args.quiet)', 'safe_state(args.quiet, args.seed)')
assert g.count('def safe_state(silent):') == 1
g = g.replace('def safe_state(silent):', 'def safe_state(silent, seed=0):')
# replace np first since it contains the plain random.seed substring
for old, new in [('np.random.seed(0)', 'np.random.seed(seed)'),
                 ('torch.manual_seed(0)', 'torch.manual_seed(seed)'),
                 ('random.seed(0)', 'random.seed(seed)')]:
    assert g.count(old) == 1, (old, g.count(old))
    g = g.replace(old, new)
train.write_text(t, encoding='utf-8')
gu.write_text(g, encoding='utf-8')
print('patched train.py and general_utils.py')
