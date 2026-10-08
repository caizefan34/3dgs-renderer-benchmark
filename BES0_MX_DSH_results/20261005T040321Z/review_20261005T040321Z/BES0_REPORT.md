# BES0 MX 结果（DSH 自动填写）

裁决：**INCONCLUSIVE_CUDA_NOT_IMPLEMENTED**。这不是已验证创新成果。

## 是否值得继续

3× 训练加速目标**未达成且当前无法判定**：生产 CUDA BES 渲染器未实现，E2/E3 仅有 Torch-reference 步时/训练时（按 NUMERICS.md 与 ADAPTER_CONTRACT.md，Torch 参考时不得当 CUDA 性能）。因此本裁决为 `INCONCLUSIVE_CUDA_NOT_IMPLEMENTED`，而非对 3× 的 GO 或 NO_GO。是否值得继续取决于是否投入实现并验证生产 CUDA 渲染器（R0 认证 kernel、R/D 路由 CUDA、FP64 prefix CUDA、Philox RNG CUDA、native alpha wrapper parity、backward VJP）。

裁决理由：

- production CUDA renderer not implemented (cuda_implemented=false); Torch-reference timings are NOT CUDA evidence and cannot support a GO/NO_GO
- selected config outside frozen search
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

- protocol_sha256：bd0499668ffa0f45c9bf385da2e1cba9a37f6edf9c5086e9a5502d71bf9675b0

- implementation_sha256：04057541806e8b09bf1baf650944ff2210ce9c9f03e58540b5691e12ff09bd3f

- selected_config 状态：FROZEN（冻结）。

- 科研计算全部在 MX（host bms-39468022-001，8×A100-PCIE-40GB，SM80）；允许卡 1/2/3/4，保护卡 0/5/6/7 未用；单卡、不驱逐他人作业。

- repo commit（checkpoint provenance）：32ab80e773f74f4d8e40ff4e338c86b29c7957f5。

- 三场景 room/bicycle/garden 的 COLMAP sparse、GT 图像、camera_sequence、checkpoint SHA256 见各 stage raw 与 config/scene_manifest.json。

## E0 数学参考（期望/梯度恒等式）

- status：PASS；color_maxabs=1.1e-16（门 1e-10）；gradient_maxabs=5.6e-17（门 1e-9）。

- scope：MX 数学参考（FP64 小列表），非生产 native 证书。

## E1 生产正确性 + 机会筛查

- status：PASS；cuda_implemented=False。

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

