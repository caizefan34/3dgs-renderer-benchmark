#!/usr/bin/env python3
"""P16-RB: apply the curve-schedule patch to the copied nerficg tree ($D/curve_src/nerficg).

The patch makes the intermediate-checkpoint schedule driven by an explicit iteration list
(env P16_BACKUP_ITERS, comma-separated) instead of BACKUP.INTERVAL multiples, so curve
snapshots can be taken just before the periodic opacity resets (a save in the same iteration
as reset_opacities runs after it at priority 1 < 90 and captures the just-reset transparent
state; observed PSNR 9.8/10.5 at 6000/12000 vs 31.2 at 18000).

Inert when the env is unset (start/stride fall back to BACKUP.INTERVAL, no membership guard
fires) - i.e. the patched tree behaves upstream-identical without the env.
"""
import argparse
import difflib
from pathlib import Path

BLOCK_IMPORTS_OLD = """from Methods.Base.utils import CallbackTimer, pre_training_callback, training_callback, post_training_callback
from Optim.Losses.Base import BaseLoss
"""

BLOCK_IMPORTS_NEW = """from Methods.Base.utils import CallbackTimer, pre_training_callback, training_callback, post_training_callback
from Optim.Losses.Base import BaseLoss

# P16-RB curve-replay patch: optional explicit intermediate-checkpoint iteration list
# (env P16_BACKUP_ITERS, comma-separated). When set, intermediate saves fire exactly at
# those loop iterations (stride 1 + membership guard) instead of every BACKUP.INTERVAL
# steps, so curve points can sit just before the periodic opacity resets. Inert when unset.
_P16_BACKUP_ITERS = [int(x) for x in os.environ.get('P16_BACKUP_ITERS', '').split(',') if x.strip()]
"""

BLOCK_SAVE_OLD = """    @training_callback(priority=1, start_iteration='BACKUP.INTERVAL', iteration_stride='BACKUP.INTERVAL')
    def _save_intermediate_checkpoint(self, iteration: int, dataset: BaseDataset) -> None:
        \"\"\"Creates an intermediate checkpoint at the current iteration.\"\"\"
        Logger.log_info(f'creating intermediate model and training checkpoint at iteration {iteration}')
"""

BLOCK_SAVE_NEW = """    @training_callback(priority=1,
                       start_iteration=min(_P16_BACKUP_ITERS) if _P16_BACKUP_ITERS else 'BACKUP.INTERVAL',
                       iteration_stride=1 if _P16_BACKUP_ITERS else 'BACKUP.INTERVAL')
    def _save_intermediate_checkpoint(self, iteration: int, dataset: BaseDataset) -> None:
        \"\"\"Creates an intermediate checkpoint at the current iteration.\"\"\"
        if _P16_BACKUP_ITERS and iteration not in _P16_BACKUP_ITERS:
            return
        Logger.log_info(f'creating intermediate model and training checkpoint at iteration {iteration}')
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="copied nerficg tree root")
    ap.add_argument("--diff-out", required=True)
    ap.add_argument("--orig-out", required=True)
    args = ap.parse_args()

    target = Path(args.tree) / "src" / "Methods" / "Base" / "Trainer.py"
    orig_text = target.read_text(encoding="utf-8")

    if "_P16_BACKUP_ITERS" in orig_text:
        raise SystemExit("tree already patched; refusing (idempotence guard)")
    for name, old in (("imports", BLOCK_IMPORTS_OLD), ("save", BLOCK_SAVE_OLD)):
        if orig_text.count(old) != 1:
            raise SystemExit(f"anchor mismatch ({name}): count={orig_text.count(old)}")

    new_text = orig_text.replace(BLOCK_IMPORTS_OLD, BLOCK_IMPORTS_NEW).replace(BLOCK_SAVE_OLD, BLOCK_SAVE_NEW)

    Path(args.orig_out).write_text(orig_text, encoding="utf-8")
    target.write_text(new_text, encoding="utf-8")
    diff = difflib.unified_diff(
        orig_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
        fromfile="a/src/Methods/Base/Trainer.py", tofile="b/src/Methods/Base/Trainer.py")
    Path(args.diff_out).write_text("".join(diff), encoding="utf-8")
    print(f"patched {target}")
    print(f"diff -> {args.diff_out}; pristine copy -> {args.orig_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
