#!/usr/bin/env python3
"""Generate an honest BES0_REPORT.md from VERDICT.json + stage JSONs.

Organizer script (not an experiment). Reads the machine verdict and the
per-stage evidence already written by the drivers, and fills the frozen
report template. It never invents numbers: missing stages/fields are
reported as NOT_RUN / NOT_COMPUTABLE. Fixed-N, MX-only, source precedents
and limitations are stated explicitly.
"""
import argparse
import json
from pathlib import Path


def load(review, name):
    p = review / (name + ".json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fmt(x, nd=4):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}g}" if isinstance(x, float) else str(x)
    return str(x) if x is not None else "NOT_COMPUTABLE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("review_dir", type=Path)
    a = ap.parse_args()
    review = a.review_dir.resolve()
    verdict = load(review, "VERDICT") or {}
    stages = {n: load(review, n) for n in ("E0", "E1", "E2", "E3")}
    v = verdict.get("verdict", "NOT_RUN")
    reasons = verdict.get("reasons", [])
    sel = load(review, "selected_config") or {}

    L = []
    L.append("# BES0 MX 结果（DSH 自动填写）\n")
    L.append(f"裁决：**{v}**。这不是已验证创新成果。\n")

    L.append("## 是否值得继续\n")
    if v == "INCONCLUSIVE_CUDA_NOT_IMPLEMENTED":
        L.append(
            "3× 训练加速目标**未达成且当前无法判定**：生产 CUDA BES 渲染器未实现，"
            "E2/E3 仅有 Torch-reference 步时/训练时（按 NUMERICS.md 与 ADAPTER_CONTRACT.md，"
            "Torch 参考时不得当 CUDA 性能）。因此本裁决为 `INCONCLUSIVE_CUDA_NOT_IMPLEMENTED`，"
            "而非对 3× 的 GO 或 NO_GO。是否值得继续取决于是否投入实现并验证生产 CUDA 渲染器"
            "（R0 认证 kernel、R/D 路由 CUDA、FP64 prefix CUDA、Philox RNG CUDA、native alpha wrapper parity、backward VJP）。\n")
    elif v == "GO_FIXED_N_PILOT":
        L.append(
            "固定 N pilot 的预注册质量门槛与时间门槛在 reference-level 下通过；"
            "但这**不是**完整 densification 系统或全场景通用方法的验证。\n")
    else:
        L.append(
            f"裁决 {v}。主要结论见下「运行身份」「表格」与限制。\n")
    if reasons:
        L.append("裁决理由：\n")
        for r in reasons:
            L.append(f"- {r}")
        L.append("")

    L.append("## 运行身份与输入\n")
    L.append(f"- 协议：{sel.get('protocol','BES0-v1')}；gamma={sel.get('gamma')}；tile={sel.get('tile')}；"
             f"bound_mode={sel.get('bound_mode')}；seeds={sel.get('seeds')}；scenes={sel.get('scene_count')}；fixed_N={sel.get('fixed_n')}。\n")
    L.append(f"- protocol_sha256：{sel.get('protocol_sha256','NOT_COMPUTABLE')}\n")
    L.append(f"- implementation_sha256：{sel.get('implementation_sha256','NOT_COMPUTABLE')}\n")
    L.append(f"- selected_config 状态：{sel.get('status','NOT_COMPUTABLE')}（冻结）。\n")
    L.append(f"- 科研计算全部在 MX（host bms-39468022-001，8×A100-PCIE-40GB，SM80）；"
             f"允许卡 1/2/3/4，保护卡 0/5/6/7 未用；单卡、不驱逐他人作业。\n")
    L.append(f"- repo commit（checkpoint provenance）：32ab80e773f74f4d8e40ff4e338c86b29c7957f5。\n")
    L.append("- 三场景 room/bicycle/garden 的 COLMAP sparse、GT 图像、camera_sequence、checkpoint SHA256 见各 stage raw 与 config/scene_manifest.json。\n")

    # E0
    L.append("## E0 数学参考（期望/梯度恒等式）\n")
    e0 = stages.get("E0")
    if e0:
        L.append(f"- status：{e0.get('status','NOT_COMPUTABLE')}；"
                 f"color_maxabs={fmt(e0.get('color_maxabs'),2)}（门 1e-10）；"
                 f"gradient_maxabs={fmt(e0.get('gradient_maxabs'),2)}（门 1e-9）。\n")
        L.append("- scope：MX 数学参考（FP64 小列表），非生产 native 证书。\n")
    else:
        L.append("- E0：NOT_RUN。\n")

    # E1
    L.append("## E1 生产正确性 + 机会筛查\n")
    e1 = stages.get("E1")
    if e1:
        L.append(f"- status：{e1.get('status','NOT_COMPUTABLE')}；cuda_implemented={e1.get('cuda_implemented','NOT_COMPUTABLE')}。\n")
        det = e1.get("details", {})
        ra = det.get("r0_audit", {})
        if ra:
            L.append(f"- R0 审计：occurrences={fmt(ra.get('occurrences'))}，violations={fmt(ra.get('bound_violations'))}，"
                     f"n_R={fmt(ra.get('n_R'))}，n_D={fmt(ra.get('n_D'))}，gamma={fmt(ra.get('gamma'))}。\n")
        wl = det.get("workload", {})
        if wl:
            L.append(f"- 工作负载（proxy）：R={fmt(wl.get('R_occurrences'))}，D={fmt(wl.get('D_occurrences'))}，"
                     f"candidate upper 和={fmt(wl.get('candidate_eval_upper_bound_sum'))}。\n")
        L.append("- scope：DSH E1 fixture-scale 正确性 + 机会筛查，非全场景 native 计时。\n")
    else:
        L.append("- E1：NOT_RUN。\n")

    # E2
    L.append("## E2 完整 step 收敛风险 + 投资门\n")
    e2 = stages.get("E2")
    if e2:
        L.append(f"- status：{e2.get('status','NOT_COMPUTABLE')}；cuda_implemented={e2.get('cuda_implemented','NOT_COMPUTABLE')}；"
                 f"timing_level={e2.get('timing_level','NOT_COMPUTABLE')}。\n")
        gi = e2.get("details", {}).get("gradient_identity_cos", {})
        if gi:
            L.append("- gradient 恒等式 cos（E[grad L_sur] vs grad full L2）："
                     + "，".join(f"{k}={fmt(v,3)}" for k, v in gi.items()) + "。\n")
        rt = e2.get("details", {}).get("reference_step_timings", {})
        if rt:
            L.append(f"- reference 步时（**非 CUDA**）：baseline={fmt(rt.get('baseline_step_s'),4)}s，"
                     f"candidate={fmt(rt.get('candidate_step_s'),4)}s。\n")
        L.append("- 注：生产 CUDA 完整 step 计时组未实现，故 3× 投资门不可判定。\n")
    else:
        L.append("- E2：NOT_RUN。\n")

    # E3
    L.append("## E3 固定 N pilot（time-to-quality）\n")
    e3 = stages.get("E3")
    if e3:
        L.append(f"- status：{e3.get('status','NOT_COMPUTABLE')}；cuda_implemented={e3.get('cuda_implemented','NOT_COMPUTABLE')}；"
                 f"timing_level={e3.get('timing_level','NOT_COMPUTABLE')}。\n")
        pairs = e3.get("details", {}).get("pairs", {})
        if pairs:
            L.append(f"- 记录 pair 数：{len(pairs)}（3 场景 × 3 种子）。\n")
        L.append("- 注：生产 CUDA 训练时与质量门未实现，故固定 N pilot 的 3× 判定不可判定。\n")
    else:
        L.append("- E3：NOT_RUN。\n")

    # render addendum
    L.append("## 渲染附项\n")
    rs = (e3 or {}).get("render_same_quality_success", "NOT_RUN")
    L.append(f"- render_same_quality_success：{rs}。未从训练收益推断推理收益。\n")

    # novelty & limitations
    L.append("## 新颖性与限制\n")
    L.append("- 候选差异（有界事件路由 + log-hazard 跳跃 + 条件积分 + 三查询训练路径）为候选组合，"
             "非本方案首创；score 项来自已有 dual-hit 估计器。\n")
    L.append("- 本版仅固定 N pilot；densification、opacity reset、完整训练系统、普适场景、"
             "机器随机误差与可证明快速边界（R1）均**未验证**。\n")
    L.append("- 生产 CUDA 渲染器未实现：R0 认证 kernel、R/D 路由 CUDA、FP64 prefix CUDA、"
             "Philox RNG CUDA、native alpha wrapper parity、backward VJP 均为未完成项。\n")
    L.append("- 减少 primitives / 降分辨率 / 低 SH / 阈值剪枝 / 省略 score 项均**未**作为 BES0。\n")

    # verdict & deliverables
    L.append("## 自动裁决与交付\n")
    L.append(f"- 自动 VERDICT.json：{v}。\n")
    L.append(f"- e3_geomean_speedup：{fmt(verdict.get('e3_geomean_speedup'),4)}；"
             f"render_same_quality_success：{rs}。\n")
    L.append("- 交付物：代码 patch、依赖版本、命令/guard/console/runstatus、原始计时与质量、"
             "checkpoint 引用 hash、自动 VERDICT、RESULT_MANIFEST。\n")
    L.append("- 下一步（未自动扩大预算）：实现并验证生产 CUDA BES 渲染器后，用新 run-id 重跑 E1/E2/E3；"
             "E0 数学参考无需重跑。\n")

    out = review / "BES0_REPORT.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "verdict": v, "sections": len(L)}))


if __name__ == "__main__":
    raise SystemExit(main())
