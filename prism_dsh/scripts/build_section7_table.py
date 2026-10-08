"""Build the Section-7 delivery report from raw probe JSONs.

Reads:  prism_dsh/remote_results/probe_*.json   (12 Section-4 units)
        prism_dsh/remote_outputs/parity_*.json  (Section 2)
        prism_dsh/remote_results/raw_m0_v0*.json (M0 raw, both versions)
Writes: prism_dsh/SECTION7_REPORT.md and SECTION4_TABLE.json
No verdict beyond the pre-registered screening labels; no speedup claim.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "remote_results", "v012")
OUT_MD = os.path.join(BASE, "SECTION4_TABLE.md")
OUT_TABLE = os.path.join(BASE, "SECTION4_TABLE.json")
OUT_AGG = os.path.join(BASE, "SECTION4_AGG.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def unit_key(path):
    name = os.path.basename(path)
    parts = name.split("_")
    return parts[1], parts[2], parts[3].split(".")[0]


def fmt(x, nd=6):
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def main():
    units = []
    for path in sorted(glob.glob(os.path.join(RES, "probe_*.json"))):
        d = load(path)
        scene, itag, stag = unit_key(path)
        arms = d.get("arms", {})
        obs = d.get("position_observability") or {}
        min_sing = obs.get("min_relative_singular")
        rank_deficient = bool(obs) and (min_sing is not None and min_sing < 1e-6)
        rec = {
            "file": os.path.basename(path), "scene": scene, "iteration": itag,
            "seed": stag, "status": d.get("status"),
            "N": (d.get("selection") or {}).get("N"),
            "common_visible": (d.get("selection") or {}).get("common_visible"),
            "active": (d.get("selection") or {}).get("active"),
            "camera_ids": d.get("camera_ids"),
            "shared_setup_seconds": d.get("shared_setup_seconds"),
            "geometry_pack": d.get("geometry_pack"),
            "observability": obs, "rank_deficient": rank_deficient,
            "before_per_view_loss": d.get("before_per_view_loss"),
            "arms": {},
        }
        for arm, v in arms.items():
            rec["arms"][arm] = {
                "train_delta": v.get("train_delta"), "audit_delta": v.get("audit_delta"),
                "losses": v.get("losses"), "optimization_seconds": v.get("optimization_seconds"),
                "with_evaluation_seconds": v.get("with_evaluation_seconds"),
                "full_resolution_backward_views": v.get("full_resolution_backward_views"),
                "peak_allocated_bytes": v.get("peak_allocated_bytes"),
                "trace": v.get("trace"), "fit_history": v.get("fit_history"),
            }
        units.append(rec)

    table = {"units": units, "count": len(units)}
    with open(OUT_TABLE, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    complete = [u for u in units if u["status"] == "PROBE_COMPLETE_NO_GO_VERDICT"]

    # Aggregate across paired seeds per (scene, iteration): mean train/audit delta per arm.
    agg = []
    groups = sorted({(u["scene"], u["iteration"]) for u in complete})
    for scene, itag in groups:
        g = [u for u in complete if u["scene"] == scene and u["iteration"] == itag]
        rec = {"scene": scene, "iteration": itag, "seeds": [u["seed"] for u in g], "arms": {}}
        for arm in sorted({a for u in g for a in u["arms"]}):
            tr = [u["arms"][arm]["train_delta"] for u in g if arm in u["arms"]]
            au = [u["arms"][arm]["audit_delta"] for u in g if arm in u["arms"]]
            rec["arms"][arm] = {
                "train_delta_mean": sum(tr) / len(tr), "audit_delta_mean": sum(au) / len(au),
                "audit_delta_per_seed": au,
                "with_evaluation_seconds_mean": sum(
                    u["arms"][arm]["with_evaluation_seconds"] for u in g if arm in u["arms"]) / len(au),
                "peak_MB_mean": sum(u["arms"][arm]["peak_allocated_bytes"] for u in g
                                    if arm in u["arms"]) / len(au) / 2 ** 20,
            }
        rec["audit_winner_per_seed"] = [
            min((u["arms"][a]["audit_delta"], a) for a in u["arms"])[1] for u in g]
        agg.append(rec)
    with open(OUT_AGG, "w", encoding="utf-8") as f:
        json.dump({"groups": agg}, f, indent=2)

    lines = []
    lines.append("# Section 4/7 交付表（自动生成）\n")
    lines.append(f"单元数：{len(units)}（其中完整 {len(complete)}，期望 12）\n")
    lines.append("## 跨 seed 汇总（每场景/迭代）\n")
    for rec in agg:
        lines.append(f"### {rec['scene']} {rec['iteration']} seeds={rec['seeds']}\n")
        for arm, v in rec["arms"].items():
            lines.append(f"- {arm}: train_delta_mean={fmt(v['train_delta_mean'])} "
                         f"audit_delta_mean={fmt(v['audit_delta_mean'])} "
                         f"per_seed_audit={[fmt(x) for x in v['audit_delta_per_seed']]} "
                         f"with_eval_s_mean={fmt(v['with_evaluation_seconds_mean'])} "
                         f"peak_MB_mean={v['peak_MB_mean']:.1f}")
        lines.append(f"- audit 最优臂（逐 seed，越小越好）: {rec['audit_winner_per_seed']}\n")
    lines.append("## 逐单元明细\n")
    for u in units:
        lines.append(f"## {u['scene']} {u['iteration']} seed {u['seed']}\n")
        lines.append(f"- status: `{u['status']}`; N={u['N']}; 共同可见={u['common_visible']}; active={u['active']}")
        lines.append(f"- cameras: {u['camera_ids']}")
        obs = u.get("observability") or {}
        lines.append(f"- observability: min_rel_sing={fmt(obs.get('min_relative_singular'))}, "
                     f"median={fmt(obs.get('median_relative_singular'))}, "
                     f"frac<1e-6={fmt(obs.get('fraction_below_1e_minus_6'))}"
                     + ("  **[INCONCLUSIVE_OBSERVABILITY]**" if u["rank_deficient"] else ""))
        lines.append(f"- shared_setup_seconds: {fmt(u['shared_setup_seconds'])}")
        lines.append(f"- before_per_view_loss: {u['before_per_view_loss']}")
        for arm, v in u["arms"].items():
            lines.append(f"- {arm}: train_delta={fmt(v['train_delta'])} audit_delta={fmt(v['audit_delta'])} "
                         f"opt_s={fmt(v['optimization_seconds'])} with_eval_s={fmt(v['with_evaluation_seconds'])} "
                         f"bwd_views={v['full_resolution_backward_views']} "
                         f"peak_MB={None if v['peak_allocated_bytes'] is None else round(v['peak_allocated_bytes']/2**20,1)}")
        lines.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\nWROTE {OUT_MD} and {OUT_TABLE}")


if __name__ == "__main__":
    main()
