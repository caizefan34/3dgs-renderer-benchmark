import json, os

RUNS = "/mnt/storage_pool/liaoyuanjun/pub_runs"
scenes = ["room", "bicycle", "garden"]
arms = ["a0", "a1", "a2"]

print(f"{'run':14s} {'wall_s':>7s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'N_final':>8s} {'clones':>8s} {'splits':>7s} {'VRAM':>5s} {'env'}")
data = {}
for s in scenes:
    for a in arms:
        rj = os.path.join(RUNS, f"{a}_{s}", "results.json")
        if not os.path.exists(rj):
            print(f"{a+'_'+s:14s} MISSING")
            continue
        d = json.load(open(rj))
        fe = d["final_eval"]
        t = d["timing"]
        env = d["renderer"]["env"]
        env_str = (f"F9={'0' if env['HIGS_DISABLE_F9'] else '1'}"
                   f" SA={'1' if env['HIGS_BWD_SCALAR_ADJOINT'] else '0'}"
                   f" H8={'1' if env['HIGS_BWD_H8_MR'] else '0'}"
                   f" PX={env['HIGS_PX_RUNTIME']} AB={env['HIGS_BWD_ABSGRAD']}")
        data[(a, s)] = d
        print(f"{a+'_'+s:14s} {t['total_wall_s']:7.0f} {fe['psnr']:7.3f} {fe['ssim']:7.4f} "
              f"{fe['lpips']:7.4f} {fe['n_gaussians']:8d} {d['total_clones']:8d} "
              f"{d['total_splits']:7d} {d['peak_vram_gb']:5.1f} {env_str}")

print("\n== cross-arm consistency per scene (PSNR spread, wall spread) ==")
for s in scenes:
    ps = [data[(a, s)]["final_eval"]["psnr"] for a in arms if (a, s) in data]
    ws = [data[(a, s)]["timing"]["total_wall_s"] for a in arms if (a, s) in data]
    if len(ps) == 3:
        print(f"  {s:8s} PSNR a0/a1/a2 = {ps[0]:.3f}/{ps[1]:.3f}/{ps[2]:.3f}  "
              f"spread={max(ps)-min(ps):.3f} dB | wall = {ws[0]:.0f}/{ws[1]:.0f}/{ws[2]:.0f}s "
              f"spread={(max(ws)-min(ws))/min(ws)*100:.1f}%")

print("\n== phase means (room) ==")
for a in arms:
    ph = data[(a, "room")]["timing"]["phase_ms"]
    def m(k):
        v = ph[k]
        return v.get("mean") if isinstance(v, dict) else v
    print(f"  a{(arms.index(a)):d} fwd={m('forward'):.2f} loss={m('loss'):.2f} "
          f"bwd={m('backward'):.2f} dens={m('densify'):.2f} opt={m('optimizer'):.2f}")
