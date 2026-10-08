# GRC3D 原型与 DSH 交接结果

日期：2026-09-27（Asia/Shanghai）。状态：本地原型可运行；真实 GPU 实验待 DSH 执行。

## OBSERVED

CPU PyTorch float64 的四组测试通过：非均匀 Bernoulli 全 outcome 枚举期望与解析方差、SSIM halo loss 与 image gradient、锚点 detach/概率校验、p=1 的 3D 梯度一致性与 directional finite difference。耗时见 tests.log。

本轮新增的 3D oracle 包含透视投影、3D anisotropic covariance、quaternion、opacity 与 SH3，共六个 Gaussian、24×32 RGB。它不同于先前 NumPy 的2D oracle，因此数值不可直接与旧结果对照。

| 构造样例 | 汇总 kappa | anchored p=1/8 / plain p=1/4 方差比 | 关键问题 |
|---|---:|---:|---|
| 平滑，scale=.18，参数扰动=.003 | 0.138455 | 0.323061 | 此样例各组 kappa 0.131–0.297 |
| 细小 Gaussian，scale=.04，扰动=.003 | 0.622857 | 1.453332 | opacity kappa≈11.688；SH≈19.481 |
| 陈旧代理模拟，scale=.18，扰动=.06 | 1.062880 | 2.480052 | 所有组均未通过3/7门槛 |

构造样例中的扰动是 raw参数随机扰动，不对应真实训练 lag；仅用于寻找机制失败，不能作真实 GPU gate 的替代。

## EVIDENCE

- `grc3d/results/tests.log`
- `grc3d/results/smooth.json`
- `grc3d/results/fine_detail.json`
- `grc3d/results/stale_anchor.json`
- `MANIFEST.json`：代码/交接文档/结果文件的 SHA256。

## HYPOTHESIS

历史细图可降低代理的图像误差，但无法保证粗分辨率 Jacobian 对细节、颜色、opacity 等参数的解释能力。当前样例支持继续做小规模证伪实验，不支持宣称通用的低方差或训练加速。

## TEST

DSH 按 `DSH_EXECUTION.md` 在 EPIC/epic 的真实 bicycle/garden 状态执行 M0/M1。包括原生 checkpoint 导出、loss与gradient parity、真实 lag、逐参数组方差、缓存与整图proxy loss成本。GPU wrapper 尚未在本轮 GPU 执行；需 DSH 验证实际 gsplat 版本与原生 trainer 一致性。

## CONCLUSION

本地结论：`READY_FOR_DSH_M0_M1`。

真实结论：`GPU_REAL_SCENE_GATE_PENDING`。尚无训练速度、质量或SOTA胜出结果。精细分支目前全图渲染后取块，未实现 sparse tile CUDA、共享投影或加权 fused SSIM。本轮已保留失败样例；若真实同类失败重复出现，应 DROP 候选，不进入完整训练。

当前连接设备离线；未向 DSH 发送消息、未启动远端 GPU job、未更新外部仓库。本包是可交付的代码与执行协议。
