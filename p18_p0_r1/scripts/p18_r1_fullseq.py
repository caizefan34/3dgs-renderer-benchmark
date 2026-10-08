#!/usr/bin/env python3
"""P18-P0-R1 full-sequence reproducibility probe: render all probe cams per scene
(same order as capture), dump hashes for all + tensors for selected cams."""
import argparse
import json
import sys
import time

sys.path.insert(0, "/root/epic05_p18_r1/scripts")
import numpy as np
import torch

from p18_r1_capture import (FROZEN_EXT, TRACE_EXT, bootstrap, hash_tensor,
                            load_model, pack_scene, render_cam, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="frozen", choices=["frozen", "off", "on"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenes", default="room,bicycle")
    ap.add_argument("--dump", default="room:19,room:291,bicycle:36")
    args = ap.parse_args()

    if args.dump == "all":
        dumps = None  # dump every cam
    else:
        dumps = set()
        for item in args.dump.split(","):
            if item:
                s, c = item.split(":")
                dumps.add((s, int(c)))

    exp = bootstrap(FROZEN_EXT if args.mode == "frozen" else TRACE_EXT)
    if args.mode == "off":
        exp.p18_trace_set(0, 0, 0, 0, 255)
    if args.mode == "on":
        from p18_r1_capture import CTA_WARPS, IDX_GATE, LOCAL_CAP, MAX_CTAS
        rec = torch.full((MAX_CTAS * CTA_WARPS * LOCAL_CAP * 16, 8), -1,
                         dtype=torch.int32, device="cuda")
        hist = torch.zeros(17, dtype=torch.int32, device="cuda")
        exp.p18_trace_set(rec.data_ptr(), hist.data_ptr(), MAX_CTAS, LOCAL_CAP, IDX_GATE)
        torch.cuda.synchronize()
        print("[seq] trace armed for on mode", flush=True)

    camset = json.load(open("/root/epic05_p18/P18_CAMERA_SET.json"))
    out = {}
    for scene in [s for s in args.scenes.split(",") if s]:
        model = load_model(scene)
        obj = pack_scene(model, scene)
        scores = camset["scenes"][scene]
        for ci in scores["probe_ids"]:
            crec = scores["camera_records"][str(ci)]
            renders, alphas, W, H = render_cam(obj, crec)
            torch.cuda.synchronize()
            h, shape, dtype = hash_tensor(renders)
            out[f"{scene}:{ci}"] = h
            if dumps is None or (scene, ci) in dumps:
                np.save(OUT / f"seq_{args.mode}_{args.tag}_{scene}_{ci}.npy",
                        renders.detach().to("cpu").contiguous().numpy())
            print(f"[seq {args.mode}.{args.tag}] {scene}:{ci} sha={h[:20]}", flush=True)
        del model, obj
        torch.cuda.empty_cache()
    (OUT / f"seq_hashes_{args.mode}_{args.tag}.json").write_text(
        json.dumps(out, indent=2))
    print(f"[seq] wrote seq_hashes_{args.mode}_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
