p = "h0_run/h0_summarize.py"
s = open(p, encoding="utf-8").read()
s = s.replace("       Ks = sorted({r[\"K\"] for r in rollouts})", "        Ks = sorted({r[\"K\"] for r in rollouts})")
open(p, "w", encoding="utf-8").write(s)
print("fixed")
