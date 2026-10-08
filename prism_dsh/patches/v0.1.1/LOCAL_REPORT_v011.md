# PRISM-GS 本地报告（v0.1.1）

## OBSERVED

7 项数学检查全部通过（`unit_tests.log`，0.87 s）。GPU M0 适配器门在三台相机上全部
`PASS_ADAPTER`（`bicycle_5k_m0.json`）；此前 v0.1.0 同样输入、同样阈值下为
`BLOCKED_M0`。差值来源为接口/数值问题，已按 v0.1.1 patch 修复，阈值未改。

同一微型场景、相同 9 次训练视角反传下，两种更新都改善两个审计视角。

| 方法 | 审计视角平均损失 | 相对初始下降 |
|---|---:|---:|
| 初始 | 0.0032851224 | — |
| PRISM 单次 proposal（v0.1.1） | 0.0016579091 | 49.53% |
| PRISM 单次 proposal（v0.1.0，屏幕参数化不同） | 0.0015847744 | 51.76% |
| 直接几何 Adam | 0.0014601380 | 55.55% |

**这个 toy 中直接 Adam 仍然更好。不存在 PRISM 优越性的实测证据。**
屏幕特征表示改为渲染器原生 conic 后，PRISM 在此 toy 上的下降幅度略降
（51.76% -> 49.53%），直接 Adam 不受影响（其参数化未变）。

## EVIDENCE

`unit_tests.log`、`cpu_probe.json`、`LOCAL_VERDICT.json`、`bicycle_5k_m0.json`；
报告从原始 JSON 生成。GPU 逐状态证据在 `../../outputs/probe_*.json`（Section 4）。
测试覆盖 SPD 往返、屏幕参数往返（involution）、投影 Jacobian 有限差分、可实现目标
恢复、近无阻尼残差正交性、零权重视角、重复相机的秩亏，以及 lifted/direct 链式梯度一致性。

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
