p = 'clockgs/src/run_c0.py'
s = open(p, encoding='utf-8').read()

# bounded schedule defaults: densify to 8K, clock 8K->12K
s = s.replace('ap.add_argument("--start_iter", type=int, default=15000)',
              'ap.add_argument("--start_iter", type=int, default=8000)')
s = s.replace('ap.add_argument("--end_iter", type=int, default=30000)',
              'ap.add_argument("--end_iter", type=int, default=12000)')
# decisive configs first
s = s.replace('ap.add_argument("--kgeos", nargs="+", type=int, default=K_GEOS)',
              'ap.add_argument("--kgeos", nargs="+", type=int, default=[1, 2, 4])')
# OOM-safe: wrap train_clock in try/except, record failure
old = ('            metrics, drift = train_clock(\n'
       '                scene_name=scene, source_path=source, model_path=mp,\n'
       '                k_geo=kg, k_opacity=1, k_sh=1,\n'
       '                start_iter=args.start_iter, end_iter=args.end_iter,\n'
       '                images=args.images, eval_split=True, seed=args.seed,\n'
       '                log_every=args.log_every, densify_until=args.start_iter,\n'
       '                out_log=log_path, record_drift=True)\n')
new = ('            try:\n'
       '                metrics, drift = train_clock(\n'
       '                    scene_name=scene, source_path=source, model_path=mp,\n'
       '                    k_geo=kg, k_opacity=1, k_sh=1,\n'
       '                    start_iter=args.start_iter, end_iter=args.end_iter,\n'
       '                    images=args.images, eval_split=True, seed=args.seed,\n'
       '                    log_every=args.log_every, densify_until=args.start_iter,\n'
       '                    out_log=log_path, record_drift=True)\n'
       '            except torch.cuda.OutOfMemoryError as e:\n'
       '                torch.cuda.empty_cache()\n'
       '                metrics = {"scene": scene, "k_geo_label": label, "error": "OOM",\n'
       '                           "psnr": float("nan"), "ssim": float("nan")}\n'
       '                drift = []\n'
       '                print(f"[C0] {scene} k_geo={label}: OOM", flush=True)\n')
assert old in s, 'train_clock call block not found'
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(s)
import ast; ast.parse(s); print('run_c0.py updated: bounded schedule + OOM-safe; syntax OK')
