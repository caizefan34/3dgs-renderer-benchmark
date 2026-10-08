"""C0 disambiguation: re-run room+garden at bicycle's exact 4K->8K schedule
(densify_until=4000, end_iter=8000) to test whether the geometry-clock quality
hit is a training-STAGE effect (early clocking) or scene-INTRINSIC.

Uses a separate append-only log (c0_disamb_log.jsonl) and model dirs so the
authoritative 8K->12K c0_log.jsonl is never clobbered."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import train_clock

SCENES = {
    "room":   "data/datasets/mipnerf360/room",
    "garden": "data/datasets/mipnerf360/garden",
}
OUT = "clockgs/runs/c0_disamb"
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["room", "garden"])
    ap.add_argument("--kgeos", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--start_iter", type=int, default=4000)   # densify_until
    ap.add_argument("--end_iter", type=int, default=8000)
    ap.add_argument("--images", default="images_8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=1000)
    args = ap.parse_args()

    log_path = os.path.join(OUT, "c0_disamb_log.jsonl")
    for scene in args.scenes:
        source = SCENES[scene]
        control = None
        for kg in args.kgeos:
            label = str(kg)
            mp = os.path.join(OUT, f"{scene}_kgeo{label}")
            print(f"\n===== DISAMB {scene} k_geo={label} "
                  f"(densify->{args.start_iter}, clock {args.start_iter}->{args.end_iter}) =====",
                  flush=True)
            t0 = time.time()
            try:
                metrics, drift = train_clock(
                    scene_name=scene, source_path=source, model_path=mp,
                    k_geo=kg, k_opacity=1, k_sh=1,
                    start_iter=args.start_iter, end_iter=args.end_iter,
                    images=args.images, eval_split=True, seed=args.seed,
                    log_every=args.log_every, densify_until=args.start_iter,
                    out_log=log_path, record_drift=True)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                metrics = {"scene": scene, "k_geo_label": label, "error": "OOM",
                           "psnr": float("nan"), "ssim": float("nan")}
                drift = []
                print(f"[DISAMB] {scene} k_geo={label}: OOM", flush=True)
            metrics["wall_s"] = time.time() - t0
            metrics["k_geo_label"] = label
            metrics["schedule"] = f"densify->{args.start_iter}_clock_{args.start_iter}->{args.end_iter}"
            with open(os.path.join(mp, "drift.json"), "w", encoding="utf-8") as f:
                json.dump(drift, f)
            if kg == 1:
                control = metrics
            print(f"[DISAMB] {scene} k_geo={label}: PSNR={metrics['psnr']:.4f} "
                  f"SSIM={metrics['ssim']:.4f} N={metrics['N_final']} "
                  f"({metrics['wall_s']:.0f}s)", flush=True)
            with open(os.path.join(OUT, "c0_disamb_results.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f)
    print("\n[DISAMB] done. Log:", log_path, flush=True)


if __name__ == "__main__":
    main()
