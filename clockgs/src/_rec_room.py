import json
r = json.load(open(r"clockgs/runs/c0/c0_results.json"))
c = r["room_kgeo1"]
lines = ["", "## 2026-10-03 — ClockGS C0 room result (images_8, densify->8K, clock 8K->12K, seed 0)", ""]
lines.append("control k_geo=1: PSNR=%.4f SSIM=%.4f N=%d" % (c["psnr"], c["ssim"], c["N_final"]))
for k in ["2","4"]:
    m = r["room_kgeo"+k]
    dp = m["psnr"]-c["psnr"]; ds = m["ssim"]-c["ssim"]
    gate = "PASS" if (abs(dp)<=0.15 and abs(ds)<=0.005) else "FAIL"
    lines.append("k_geo=%s: PSNR=%.4f (d=%+.4f dB) SSIM=%.4f (d=%+.4f) -> %s" % (k, m["psnr"], dp, m["ssim"], ds, gate))
lines.append("Note: room is small/early; deltas within noise. garden+bicycle pending (larger, geometry-sensitive).")
txt = "\n".join(lines)
with open(r"D:\学习\3dgs-renderer-benchmark\CLOCKGS_INFRA.md","a",encoding="utf-8") as f:
    f.write(txt+"\n")
print(txt)
