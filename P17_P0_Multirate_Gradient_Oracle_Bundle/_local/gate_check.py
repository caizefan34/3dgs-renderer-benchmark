import json
import statistics

for scene in ("room", "bicycle"):
    d = f"/root/epic05_p17/P17_UNITS/{scene}"
    import os
    if not os.path.isdir(d):
        print(scene, ": no units dir")
        continue
    units = [json.load(open(os.path.join(d, f)))
             for f in sorted(os.listdir(d)) if f.endswith(".json")]
    print(f"== {scene}: {len(units)} units ==")
    print("parity all pass:", all(u["d1_parity_pass"] for u in units))
    co = [u["oracle_interaction_ratio"] for u in units]
    med = statistics.median(co)
    p75 = sorted(co)[int(0.75 * (len(co) - 1))]
    print(f"C_oracle median={med:.4f} p75={p75:.4f}")
    for key in ("means", "scales", "quats", "opacity", "sh0", "shN"):
        cs = [u["metrics"][key]["cosine"] for u in units
              if u["metrics"][key]["cosine"] is not None]
        if cs:
            print(f"{key:8s} cosine med={statistics.median(cs):.5f} "
                  f"p10={sorted(cs)[int(0.1 * (len(cs) - 1))]:.5f} "
                  f"n={len(cs)}")
        else:
            print(f"{key:8s} cosine all None")
    imp = [u for u in units if u["step"]["reference_improving"]]
    rs = [u["step"]["R_step"] for u in imp
          if u["step"]["R_step"] is not None]
    n_nonimproving = len(units) - len(imp)
    if rs:
        rmed = statistics.median(rs)
        n90110 = sum(1 for r in rs if 0.90 <= r <= 1.10)
        print(f"R_step: n_improving={len(imp)} n_nonimproving={n_nonimproving} "
              f"n_R={len(rs)} med={rmed:.4f} in[.9,1.1]={n90110}")
    else:
        print(f"R_step: none improving (n_nonimproving={n_nonimproving})")
    pa = [u["step"]["P_audit"] for u in units]
    pmed = statistics.median(pa)
    nbig = sum(1 for p in pa if p > 0.01)
    print(f"P_audit med={pmed:.5f} n>0.01={nbig}")
    drop = []
    if med > 0.50:
        drop.append("C_oracle>0.5")
    mcos = [u["metrics"]["means"]["cosine"] for u in units
            if u["metrics"]["means"]["cosine"] is not None]
    if mcos and statistics.median(mcos) < 0.980:
        drop.append("means_cos<0.98")
    ocos = [u["metrics"]["opacity"]["cosine"] for u in units
            if u["metrics"]["opacity"]["cosine"] is not None]
    if ocos and statistics.median(ocos) < 0.980:
        drop.append("opacity_cos<0.98")
    if rs and statistics.median(rs) < 0.80:
        drop.append("R_step<0.8")
    if nbig >= 4:
        drop.append("P_audit>0.01 x4")
    print("HARD_DROP triggers:", drop if drop else "none")
    cand = []
    if med <= 0.30 and p75 <= 0.35:
        cand.append("work OK")
    else:
        cand.append("work FAIL")
    if mcos and statistics.median(mcos) >= 0.995 and \
            sorted(mcos)[int(0.1 * (len(mcos) - 1))] >= 0.990:
        cand.append("means OK")
    else:
        cand.append("means FAIL")
    if ocos and statistics.median(ocos) >= 0.995:
        cand.append("opacity OK")
    else:
        cand.append("opacity FAIL")
    if rs and statistics.median(rs) >= 0.95 and \
            statistics.median(rs) <= 1.05 and \
            sum(1 for r in rs if 0.90 <= r <= 1.10) >= 10:
        cand.append("R_step OK")
    else:
        cand.append("R_step FAIL")
    if statistics.median(pa) <= 0.002 and nbig <= 1:
        cand.append("audit OK")
    else:
        cand.append("audit FAIL")
    print("CANDIDATE:", " and ".join(cand))
