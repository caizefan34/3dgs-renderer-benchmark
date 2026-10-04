# BES0 MX 结果（DSH 自动填写）

裁决：**INCONCLUSIVE_CUDA_NOT_IMPLEMENTED**。这不是已验证创新成果。

## 是否值得继续

3× 训练加速目标**未达成且当前无法判定**：生产 CUDA BES 渲染器未实现，E2/E3 仅有 Torch-reference 步时/训练时（按 NUMERICS.md 与 ADAPTER_CONTRACT.md，Torch 参考时不得当 CUDA 性能）。因此本裁决为 `INCONCLUSIVE_CUDA_NOT_IMPLEMENTED`，而非对 3× 的 GO 或 NO_GO。是否值得继续取决于是否投入实现并验证生产 CUDA 渲染器（R0 认证 kernel、R/D 路由 CUDA、FP64 prefix CUDA、Philox RNG CUDA、native alpha wrapper parity、backward VJP）。

裁决理由：

- production CUDA renderer not implemented (cuda_implemented=false); Torch-reference timings are NOT CUDA evidence and cannot support a GO/NO_GO
- selected config fingerprint differs
- selected config outside frozen search
- E1 implementation differs (rerun production stages after edits)
- E1 guard/status/measurements/source_patch evidence incomplete
- E2 missing/invalid
- E3 missing/invalid
- E1:native_semantics_frozen unknown
- E1:native_wrapper_correct unknown
- E1:bounds_certified unknown
- E1:routing_partition_correct unknown
- E1:prefix_numerics_correct unknown
- E1:rng_streams_independent unknown
- E1:native_small_fixture_gradient_correct unknown
- E1:environment_verified unknown
- E1:bound_violation_count missing
- E1:native_wrapper_color_maxabs missing
- must have 3 E1 scenes, 3 E2 scenes, 9 E3 pairs
- three unique scene ids required
- E3 pair identities incomplete/duplicated

## 运行身份与输入

- 协议：BES0-v1；gamma=2；tile=16；bound_mode=R0_ENUMERATED_NATIVE；seeds=[17, 29, 43]；scenes=3；fixed_N=True。

- protocol_sha256：b70c686330b31f494c6d1d8416e1db03a45f0a448fa5076b4ae71831f0bc0a83

- implementation_sha256：28c7894c4ec24656b1550f9fa98b214a0a85cf145ab067a386d710950b99c4c1

- selected_config 状态：FROZEN（冻结）。

- 科研计算全部在 MX（host bms-39468022-001，8×A100-PCIE-40GB，SM80）；允许卡 1/2/3/4，保护卡 0/5/6/7 未用；单卡、不驱逐他人作业。

- repo commit（checkpoint provenance）：32ab80e773f74f4d8e40ff4e338c86b29c7957f5。

- 三场景 room/bicycle/garden 的 COLMAP sparse、GT 图像、camera_sequence、checkpoint SHA256 见各 stage raw 与 config/scene_manifest.json。

## E0 数学参考（期望/梯度恒等式）

- status：PASS；color_maxabs=1.1e-16（门 1e-10）；gradient_maxabs=5.6e-17（门 1e-9）。

- scope：MX 数学参考（FP64 小列表），非生产 native 证书。

## E1 生产正确性 + 机会筛查

- status：FAIL；cuda_implemented=False。

- scope：DSH E1 fixture-scale 正确性 + 机会筛查，非全场景 native 计时。

## E2 完整 step 收敛风险 + 投资门

- E2：NOT_RUN。

## E3 固定 N pilot（time-to-quality）

- E3：NOT_RUN。

## 渲染附项

- render_same_quality_success：NOT_RUN。未从训练收益推断推理收益。

## 新颖性与限制

- 候选差异（有界事件路由 + log-hazard 跳跃 + 条件积分 + 三查询训练路径）为候选组合，非本方案首创；score 项来自已有 dual-hit 估计器。

- 本版仅固定 N pilot；densification、opacity reset、完整训练系统、普适场景、机器随机误差与可证明快速边界（R1）均**未验证**。

- 生产 CUDA 渲染器未实现：R0 认证 kernel、R/D 路由 CUDA、FP64 prefix CUDA、Philox RNG CUDA、native alpha wrapper parity、backward VJP 均为未完成项。

- 减少 primitives / 降分辨率 / 低 SH / 阈值剪枝 / 省略 score 项均**未**作为 BES0。

## 自动裁决与交付

- 自动 VERDICT.json：INCONCLUSIVE_CUDA_NOT_IMPLEMENTED。

- e3_geomean_speedup：NOT_COMPUTABLE；render_same_quality_success：NOT_RUN。

- 交付物：代码 patch、依赖版本、命令/guard/console/runstatus、原始计时与质量、checkpoint 引用 hash、自动 VERDICT、RESULT_MANIFEST。

- 下一步（未自动扩大预算）：实现并验证生产 CUDA BES 渲染器后，用新 run-id 重跑 E1/E2/E3；E0 数学参考无需重跑。


---

## 关键发现（DSH 2026-10-05 补充）

### GPU 授权修正（操作员）
操作员于 2026-10-05 明确授权 GPU 6/7（当时空闲）。据此把冻结的
`config/mx_target.json` 由 `allowed=[1,2,3,4]`/`protected=[0,5,6,7]`
修正为 `allowed=[1,2,3,4,6,7]`/`protected=[0,5]`，并重算 protocol/implementation
hash 后全新起跑（旧 run-id 因 hash 失配作废）。E0 用 GPU7，E1 用 GPU6；全程单卡、
不触碰受保护卡 0/5、不驱逐共驻作业。

### E1 适配器 conic bug（根因链式发现）
稳健投影门（对比适配器 `project_gaussians_2d` 与 gsplat 1.5.3 `_render` 的
per-Gaussian `means2d`/`conics`）在 67 个被光栅化的 Gaussian 上测得：
- `means2d_max_abs = 6.1e-5`（像素）——适配器投影中心与 gsplat **一致**。
- `conic_max_rel = 3.8e+25`——适配器 conic 与 gsplat **严重发散**。

根因链：**conic 发散 → 2D alpha 退化（`alpha=opacity*exp(-0.5*q)` 因 q 异常而落到
cutoff 以下 → alpha≈0）→ 每个 occurrence 的下界 l=0 → 全部路由到 D
（`l_zero_fraction=1.0`, n_R=0, n_D=860）→ 机会筛查（R/D 比例、candidate 上界和）
完全失效（恒为 0）**。

因此 fixture 的 “全 D / 零机会” **不是** “方法有界事件无加速空间” 的证据，而是
**适配器 conic 实现与 gsplat 不一致的假象**。任何真实机会测量都必须先修复 conic。

### 机器裁决解读
主裁决理由：`production CUDA renderer not implemented (cuda_implemented=false)`。
其余理由（selected config fingerprint differs / E1 implementation differs /
E2/E3 missing / E1 required-checks unknown / bound_violation_count missing 等）
均为**证据形态问题**：E1 修复（amax.values 一行 bug）发生在 E0 起跑之后，导致
implementation hash 漂移；且交付的 E1/E2/E3 driver 的 checks 键与 reduce_verdict
的 `required` 契约不一致。这些不是科学结论，但会使裁决停在 INCONCLUSIVE 而非
更干净的 INCONCLUSIVE_CUDA_NOT_IMPLEMENTED。

### 诚实结论
BES0（有界事件高斯泼溅，3× 训练加速）当前**无法判定**：
1. 生产 CUDA BES 渲染器未实现（R0 认证 kernel、R/D 路由 CUDA、FP64 prefix CUDA、
   Philox RNG CUDA、native alpha wrapper parity、backward VJP 均未完成）。
2. 即便在 Torch-reference 层，适配器 conic 与 gsplat 不一致，候选路径渲染正确性存疑。
E0 已证明**核心数学（期望/梯度恒等式、采样分布、候选计数界）在 FP64 下精确正确**。
要得到真正的 GO/NO_GO，需先修复适配器 conic，再实现并验证生产 CUDA 渲染器。
本轮**不**把 “查询数减少” 改名 “训练 3×”，不宣称首创或颠覆性成功。
