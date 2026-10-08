import io
p="cg0_run.py"
s=open(p,encoding="utf-8").read()
anchor='    s_best, b_best = best[1], best[2]\n    t_cal = time.perf_counter() - t_cal0\n'
assert anchor in s, "anchor not found"
add=anchor+'''    # ---- calibration artifact (prereg section 8 / 16) ----\n    jdump({"cell": cell, "grid_s": list(APP.CALIB_GRID_S),\n           "grid_b": list(APP.CALIB_GRID_B),\n           "loss_type": "0.8*L1+0.2*(1-SSIM) on CALIB8",\n           "forward_only": True, "n_combinations": len(calib_results),\n           "results": calib_results,\n           "best_s": s_best, "best_b": b_best,\n           "best_calib8_c0_loss": best[3]},\n          os.path.join(out, "CG0_CALIBRATION.json"))\n'''
s=s.replace(anchor,add,1)
open(p,"w",encoding="utf-8").write(s)
print("patched" if "CG0_CALIBRATION.json" in s else "FAILED")
