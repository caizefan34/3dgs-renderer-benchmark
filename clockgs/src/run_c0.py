"""C0 runner: geometry-clock convergence oracle matrix."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import train_clock

SCENES = {
    "room":   "data/datasets/mipnerf360/room",
    "garden": "data/datasets/mipnerf360/garden",
    "bicycle":"data/datasets/mipnerf360/bicycle",
}
K_GEOS = [1, 2, 4, 8, 16, 10**9]
K_LABEL = {10**9: "inf"}
OUT = "clockgs/runs/c0"
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["room", "garden", "bicycle"])
    ap.add_argument("--kgeos", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--start_iter", type=int, default=8000)
    ap.add_argument("--end_iter", type=int, default=12000)
    ap.add_argument("--images", default="images_8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=1000)
    args = ap.parse_args()

    results = {}
    log_path = os.path.join(OUT, "c0_log.jsonl")
    for scene in args.scenes:
        source = SCENES[scene]
        control = None
        for kg in args.kgeos:
            label = K_LABEL.get(kg, str(kg))
            mp = os.path.join(OUT, f"{scene}_kgeo{label}")
            print(f"\n===== C0 {scene} k_geo={label} =====", flush=True)
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
                print(f"[C0] {scene} k_geo={label}: OOM", flush=True)
            metrics["wall_s"] = time.time() - t0
            metrics["k_geo_label"] = label
            with open(os.path.join(mp, "drift.json"), "w", encoding="utf-8") as f:
                json.dump(drift, f)
            if kg == 1:
                control = metrics
            results[f"{scene}_kgeo{label}"] = metrics
            print(f"[C0] {scene} k_geo={label}: PSNR={metrics['psnr']:.4f} "
                  f"SSIM={metrics['ssim']:.4f} N={metrics['N_final']} "
                  f"({metrics['wall_s']:.0f}s)", flush=True)
            with open(os.path.join(OUT, "c0_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    print("\n[C0] matrix done. Results at", os.path.join(OUT, "c0_results.json"))


if __name__ == "__main__":
    main()
