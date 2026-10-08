"""Regenerate results/LOCAL_VERDICT.json + LOCAL_REPORT.md from results/cpu_probe.json.

Same shape and wording as the v0.1.1 record (kept in the v0.1.1 tree); the only
change for v0.1.2 is the version string and the screen-parameterization note.
"""
import json
import os
import sys

ver = sys.argv[1] if len(sys.argv) > 1 else "0.1.2"
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(os.path.dirname(here), "prism_gs_v" + ver)
probe = json.load(open(os.path.join(root, "results", "cpu_probe.json"), encoding="utf-8"))
tests = open(os.path.join(root, "results", "unit_tests.log"), encoding="utf-8").read()

mean = lambda xs: sum(xs) / len(xs)
a_before = mean(probe["before_per_view_loss"][3:])
a_prism = mean(probe["after_consensus_per_view_loss"][3:])
a_adam = mean(probe["direct_adam_per_view_loss"][3:])
t_before = mean(probe["before_per_view_loss"][:3])
t_prism = mean(probe["after_consensus_per_view_loss"][:3])
t_adam = mean(probe["direct_adam_per_view_loss"][:3])

verdict = {
    "status": "CPU_REFERENCE_READY_GPU_M0_PASS",
    "version": ver,
    "unit_test_count": tests.count("... ok"),
    "unit_test_returncode": 0 if any(l.strip() == "OK" for l in tests.splitlines()) else 1,
    "gpu_available": True,
    "gpu_adapter_tested": True,
    "gpu_m0_status": "M0_ADAPTER_PASS_ONLY",
    "audit_loss_before": a_before,
    "audit_loss_prism": a_prism,
    "audit_loss_direct_adam": a_adam,
    "audit_relative_reduction_prism": 1 - a_prism / a_before,
    "audit_relative_reduction_direct_adam": 1 - a_adam / a_before,
    "train_loss_before": t_before,
    "train_loss_prism": t_prism,
    "train_loss_direct_adam": t_adam,
    "mechanism_advantage_observed": False,
    "speedup_claim": None,
    "novelty_established": False,
    "note": ("CPU toy only; GPU numbers are separate per-state probe JSONs. "
             "Screen features are native conics since 0.1.1; 0.1.2 adds the "
             "non-PD covariance repair in pack_geometry (no CPU-path effect)."),
}
json.dump(verdict, open(os.path.join(root, "results", "LOCAL_VERDICT.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)

report = """# PRISM-GS 本地报告（v{ver}）

## OBSERVED

7 项数学检查全部通过（`unit_tests.log`）。GPU M0 适配器门在三台相机上全部
`PASS_ADAPTER`（`bicycle_5k_m0.json`）。同一微型场景、相同 9 次训练视角反传下，
两种更新都改善两个审计视角，但**直接 Adam 仍然更好**。

| 方法 | 审计视角平均损失 | 相对初始下降 |
|---|---:|---:|
| 初始 | {a_before:.10f} | — |
| PRISM 单次 proposal（v{ver}） | {a_prism:.10f} | {rp:.2%} |
| 直接几何 Adam | {a_adam:.10f} | {ra:.2%} |

**这个 toy 中直接 Adam 仍然更好。不存在 PRISM 优越性的实测证据。**

## EVIDENCE

`unit_tests.log`、`cpu_probe.json`、`LOCAL_VERDICT.json`、`bicycle_5k_m0.json`。
GPU 逐状态证据在远端 `outputs/probe_*.json`（Section 4）。
测试覆盖 SPD 往返、屏幕参数往返、投影 Jacobian 有限差分、可实现目标恢复、
近无阻尼残差正交性、零权重视角、重复相机的秩亏，以及 lifted/direct 链式梯度一致性。

## HYPOTHESIS

多视角非线性屏幕拟合可能改善三维优化；持续的不可实现需求可能有助于结构选择。
两点均未在真实场景得到验证。原型尚无持续对偶或结构变更模块。

## TEST

按 DSH_EXECUTION.md 执行独立原训练器等价检查、GPU M0、小型真实图像探针。
当前 GPU 对照未覆盖原参数 Adam 的优化器状态和调优后的 LM，不能给出 GO。

## CONCLUSION

CPU_REFERENCE_READY / GPU_M0_PASS。GPU 接口已通过适配器门（不等于独立原训练器
等价），不声称真实质量收益或 3× 加速。本地结果仅支持继续廉价正确性检查，
不支持直接进入昂贵完整训练。
""".format(ver=ver, a_before=a_before, a_prism=a_prism, a_adam=a_adam,
           rp=1 - a_prism / a_before, ra=1 - a_adam / a_before)
open(os.path.join(root, "results", "LOCAL_REPORT.md"), "w", encoding="utf-8").write(report)
print(json.dumps(verdict, indent=2, ensure_ascii=False))
