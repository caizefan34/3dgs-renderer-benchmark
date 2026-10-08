#!/usr/bin/env python3
"""Build an isolated author-trainer mask correction and 3K trial wrapper."""
from pathlib import Path


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one patch site, found {count}: {old!r}")
    return source.replace(old, new)


def main():
    root = Path(__file__).resolve().parent
    author = Path("/root/codex_sadgs_baseline_20260927/train.py")
    trainer = author.read_text()
    trainer = replace_once(trainer, "        last_radii = None\n        last_visibility_filter = None", "        last_radii = None\n        batch_visible = None\n        last_visibility_filter = None")
    trainer = replace_once(trainer, "            last_radii = radii\n            last_visibility_filter = visibility_filter", "            last_radii = radii\n            current_visible = radii > 0\n            batch_visible = current_visible if batch_visible is None else (batch_visible | current_visible)\n            last_visibility_filter = visibility_filter")
    trainer = replace_once(trainer, "                elif opt.optimizer_type == \"hybrid\":\n                    visible = radii > 0", "                elif opt.optimizer_type == \"hybrid\":\n                    visible = batch_visible")
    (root / "train_union_mask.py").write_text(trainer)

    wrapper_source = (root / "run_all_original.py").read_text()
    wrapper_source = replace_once(wrapper_source, "RUN=pathlib.Path('/root/sadgs_room_author_3k_l2zero_20260927')", "RUN=pathlib.Path('/root/sadgs_room_unionmask_3k_l2zero_20260928')")
    wrapper_source = replace_once(wrapper_source, "train=[PY,'train.py',", "train=[PY,'/root/sad_hybrid_unionmask_trial_20260928/train_union_mask.py',")
    if wrapper_source.count("'PYTHONPATH':f'{TARGET}:{SOURCE}'") != 2:
        raise RuntimeError("unexpected PYTHONPATH patch sites")
    wrapper_source = wrapper_source.replace("'PYTHONPATH':f'{TARGET}:{SOURCE}'", "'PYTHONPATH':f'{PROJECT}:{TARGET}:{SOURCE}'")
    wrapper_source = wrapper_source.replace("/root/sadgs_room_author_3k_l2zero_20260927/run.meta.json", "/root/sadgs_room_unionmask_3k_l2zero_20260928/run.meta.json")
    (root / "run_union_3k.py").write_text(wrapper_source)


if __name__ == "__main__":
    main()
