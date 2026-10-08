import json
import re
import statistics

# tau logs use a shorter format:
# [tauX.X] <scene> cam<N>: C=<val> n_d2+=<n> R=<val> P=<val>
log_re = re.compile(
    r"\[tau([\d.]+)\] (room|bicycle) cam(\d+): C=([\d.eE+-]+) "
    r"n_d2\+=\d+ R=([\d.eE+-]+) P=([\d.eE+-]+)")

def parse_log(path):
    out = []
    for m in log_re.finditer(open(path).read()):
        tau, scene, cam, C, R, P = m.groups()
        out.append({
            "tau": float(tau), "scene": scene, "camera_id": int(cam),
            "C_oracle": float(C), "R_step": float(R), "P_audit": float(P),
        })
    return out

logs = [
    "/root/epic05_p17/room_tau1.log",
    "/root/epic05_p17/bicycle_tau1.log",
    "/root/epic05_p17/room_tau4.log",
    "/root/epic05_p17/bicycle_tau4.log",
]

sens = {"taus": {}}
for path in logs:
    recs = parse_log(path)
    if not recs:
        continue
    tau = recs[0]["tau"]
    scene = recs[0]["scene"]
    co = [r["C_oracle"] for r in recs]
    rs = [r["R_step"] for r in recs]
    pa = [r["P_audit"] for r in recs]
    sens["taus"].setdefault(str(tau), {})[scene] = {
        "n_units": len(recs),
        "C_oracle_median": statistics.median(co),
        "R_step_median": statistics.median(rs),
        "P_audit_median": statistics.median(pa),
        "R_step_values": [r["R_step"] for r in recs],
        "C_oracle_values": [r["C_oracle"] for r in recs],
    }

with open("/root/epic05_p17/P17_SENSITIVITY_TAU.json", "w") as f:
    json.dump(sens, f, indent=2)
print(json.dumps(sens, indent=2))
print("Wrote P17_SENSITIVITY_TAU.json")
