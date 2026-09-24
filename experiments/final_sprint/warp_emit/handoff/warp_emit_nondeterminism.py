"""Backward nondeterminism envelope control for frozen WARP_ALL.

Run in the baseline package twice (tags baseline_a/b), then in the candidate
package twice (candidate_a/b), then candidate with --reference baseline_a.
Each invocation repeats the exact same scene/camera three times.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from warp_emit_correctness import SCENES, metrics, run

ROOT = Path(os.environ.get("WARP_EMIT_REMOTE_ROOT", Path(__file__).resolve().parent))
OUT = ROOT / "raw" / "nondeterminism"; OUT.mkdir(parents=True, exist_ok=True)
GRADS = ("xyz_grad", "quats_grad", "scales_grad", "opacity_grad", "shs_grad")


def one_rep(scene):
    result = run(scene, 0)
    return {name: result["tensors"][name] for name in GRADS}, result["hashes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reference", type=Path)
    args = ap.parse_args()
    reference = torch.load(args.reference, weights_only=True) if args.reference else None
    output, saved_reference = {"tag": args.tag, "reps": args.reps, "scenes": {}}, {}
    for scene in SCENES:
        anchor = reference[scene]["grads"] if reference else None
        samples, structure_identity = [], []
        for rep in range(args.reps):
            grads, hashes = one_rep(scene)
            if anchor is None:
                anchor = grads
            samples.append({name: metrics(grads[name], anchor[name]) for name in GRADS})
            structure_identity.append(hashes)
        saved_reference[scene] = {"grads": anchor, "hashes": structure_identity[0]}
        output["scenes"][scene] = {"reference": str(args.reference) if args.reference else args.tag,
                                    "gradient_deltas": samples,
                                    "structure_hashes": structure_identity}
        print(scene, output["scenes"][scene], flush=True)
    torch.save(saved_reference, OUT / f"{args.tag}_reference.pt")
    (OUT / f"{args.tag}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
