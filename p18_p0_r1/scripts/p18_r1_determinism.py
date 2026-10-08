#!/usr/bin/env python3
"""P18-P0-R1 determinism probe: render the same cam repeatedly (intra-process)
and save tensors for cross-process comparison."""
import argparse
import sys

sys.path.insert(0, "/root/epic05_p18_r1/scripts")
import numpy as np
import torch

from p18_r1_capture import (FROZEN_EXT, TRACE_EXT, bootstrap, hash_tensor,
                            load_model, pack_scene, render_cam, sha256_file, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="frozen", choices=["frozen", "off", "on"])
    ap.add_argument("--scene", default="room")
    ap.add_argument("--cam", type=int, default=19)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="a")
    args = ap.parse_args()

    import json
    exp = bootstrap(FROZEN_EXT if args.mode == "frozen" else TRACE_EXT)
    if args.mode == "off":
        exp.p18_trace_set(0, 0, 0, 0, 255)
    # on/unused here: determinism probe runs without trace (keeps memory small)

    camset = json.load(open("/root/epic05_p18/P18_CAMERA_SET.json"))
    crec = camset["scenes"][args.scene]["camera_records"][str(args.cam)]

    model = load_model(args.scene)
    obj = pack_scene(model, args.scene)
    prev = None
    for i in range(args.reps):
        renders, alphas, W, H = render_cam(obj, crec)
        torch.cuda.synchronize()
        h, shape, dtype = hash_tensor(renders)
        arr = renders.detach().to("cpu").contiguous().numpy()
        line = f"[det {args.mode}.{args.tag}] rep{i} sha={h[:20]}"
        if prev is not None:
            d = np.abs(arr - prev)
            line += (f" max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
                     f"n_diff={int((d > 0).sum())}/{d.size}")
        print(line, flush=True)
        prev = arr
    np.save(OUT / f"det_{args.mode}_{args.tag}.npy", prev)
    print(f"[det] saved {OUT / f'det_{args.mode}_{args.tag}.npy'} sha={hash_tensor(renders)[0][:20]}",
          flush=True)


if __name__ == "__main__":
    main()
