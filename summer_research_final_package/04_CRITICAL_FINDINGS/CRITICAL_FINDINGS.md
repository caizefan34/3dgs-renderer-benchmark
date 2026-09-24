# CRITICAL_FINDINGS.md — 关键发现（按重要性排序，每条含证据路径）

> 本文件是最终报告“结果与讨论”部分的**核心素材**。每条发现都附有：
> (1) 结论的一句话表述；(2) 支持该结论的关键数值/事实；
> (3) 对应的证据文件路径（相对 `summer_research_final_package/` 或仓库根目录）。
> 规则：只有能从证据文件直接回溯的数值才可写入报告；无法回溯的只能作为“推测”或“未验证”。

---

## 1. C42 不是 universal additive optimization（全基线普适叠加优化）【最高优先级】

- **结论**：C42（下采样/降采样 SSIM 相关加速方法）对不同的 baseline 效果不一致，
  其对 Reference V1 提升的数值不能推广到 Speed / FastGS / Faster-GS 等其他 baseline；
  “叠加后仍然成立”这一说法不成立。
- **关键数值**：13 场景 × 3 种子（132 个训练任务的汇总）中，C42 在某些 baseline 组合下
  有正收益、在另一些组合下收益消失甚至出现回退；最终判定为 **DROP as universal，
  KEEP as tuning tool**。
- **证据路径**：
  - `08_C42/` 目录下全部文件（尤其 `C42_final_decision.md`）
  - `17_Raw_Data/results/c42_adaptive/`、`results/a100/phase-c42/`
  - `17_Raw_Data/results/c42_scene_raw/*.json`
- **写报告注意**：不要出现“C42 带来 X% 加速”而无基线限定；
  必须写“在 AAA baseline 下 C42 使 B 阶段提速 X%（wall，IA）”。

---

## 2. 唯一经过确认的训练加速：`gsat_30k_fused_prune10_rclip05`（1.164x wall）

- **结论**：这是 CI 中唯一覆盖 11 scenes × 3 seeds × 132 jobs 全矩阵的确认（CONFIRMED）
  训练加速方案；对应 wall-clock 平均加速比 **1.164x**（IA 区间下界 1.034x）。
- **关键数值**：`speedup 1.164x（95% CI [1.034, 1.306]）`；该方案同时满足了
  质量门（PSNR/SSIM/LPIPS 增量边界）与正确性门（梯度拓扑/优化器状态检查）。
- **证据路径**：
  - `01_PROJECT_OVERVIEW/project_overview.md` 中“CONFIRMED 加速”小节
  - `17_Raw_Data/results/training/*30k*.json`（132-job 汇总）
  - 仓库 `configs/` 下 `gsat_30k_fused_prune10_rclip05` 配置
- **写报告注意**：这是报告 1-2 句“核心结果”最可靠的数字。

---

## 3. 反向传播的“1% 法则”：1% 高稀疏参数 ≈ 29% 反向成本；固定成本地板 ≈1.7ms

- **结论**：反向过程存在强稀疏性——仅约 1% 的 Gaussians 贡献约 29% 的后向计算成本；
  无论稀疏度过高，后向仍有约 1.7ms 的固定地板（非可压缩部分）。
- **关键数值**：`1% Gaussians -> 28.8% backward cost`；fixed floor ≈ 1.7ms；
  这是 C25/C51 系列选择性反向方案的事実依据。
- **证据路径**：
  - `10_BACKWARD_PROFILING_C25/` 目录下 `C25_profile_and_floor.md`
  - `17_Raw_Data/results/a100/phase-c25/`、`phase-c26/`
- **写报告注意**：1.7ms 是“固定地板”而非“总时间”；不要把它写进总加速结果。

---

## 4. C49 的收益被系统性高估：修正后上限 ≈ 1.13%

- **结论**：早期报告把 densification 梯度（G_dens）当作优化梯度（G_opt）使用，
  造成 82.0% 的浓度假象；修正后（两者分离、带符号校正）浓度降至 75.8%，
  且理论极限（E2E）只有 ≈ 1.13%。
- **关键数值**：82.0% (uncorrected) → 75.8% (corrected)；max E2E ≈ 1.13%。
- **证据路径**：
  - `12_ATTRIBUTE_DECOUPLED_C49/` 目录下全部文件
  - `reports/phase-r2-attribute-decoupled-backward-gate.md`
  - `reports/phase_c49_gaussian_lifecycle_research.md`
- **写报告注意**：引用时必须注明“修正后”；不能把 82% 当结果。

## 5. Candidate C 的证书审计：R3 7,494 违规 → R3.1 修复 → 0 违规（VEC pass）

- **结论**：候选 C 最初的实现存在系统性数值问题（把二维投影近似当作颜色代理，
  conic-as-color），导致 7,494 个检查点违规；R3.1 用 RGB 差分修正后，90 项测量
  违规数为 0，最坏比率 0.9998（容差 0.0002），VEC 门通过。R4 的 CUDA 内核已写但
  **未编译、未在 GPU 上验证**，因此不能声称端到端加速。
- **关键数值**：7,494 violations（R3）→ 0 violations（R3.1, n=90）；VEC gate PASS；
  最坏比率 0.9998。
- **证据路径**：
  - `13_CANDIDATE_C/` 目录下 `R3_audit.md`、`R3.1_*.md`、`R4_*.md`
  - 仓库 `candidate_c_source_audit/`（源码审计 JSON）
  - `reports/r4/r4-cuda-insertion-audit.md`
- **写报告注意**：R3 的 7,494 不是“候选 C 失败”，而是“审计发现缺陷并修复”；
  R4 是“实现完成，验证未完成”。

## 6. 瓦片几何：tile16 是 A100 与 RTX 5070 双平台的共同最优；tile 是运行时参数

- **结论**：tile size（16/20/24/32）扫描中，**tile16** 在 fwd 与 fwd+bwd 两个指标上
  都是稳定最优（RTX 5070 上 dtile16 微观时间最优；A100 上亦是）；tile 值是运行时参数，
  同一编译产物可以运行任何 tile，因此切换成本为零。
- **关键数值**：tile16 的 fwd 与 fwd+bwd 在全部场景中均优于 tile20/tile24/tile32；
  tile20 训练异常（561.8 min）已单独追踪为异常点。
- **证据路径**：
  - `07_TILE_GEOMETRY_HARDWARE/`
  - `reports/epic05/hardware-aware-tile-study-2026-08-19.md`
  - `reports/epic05/tile-mechanism-training-study-2026-08-19.md`
  - `reports/epic05/gradient-correctness-tile-size-2026-09-01.md`
- **写报告注意**：不要写“tile16 比 tile32 快 X%”——tile 是运行时参数，比较应视为
  同一个二进制不同配置，而不是代码版本之间的比较。

## 7. 占用率坍塌机制（C19-2 受控重放）

- **结论**：瓦片占用率（occupancy）在大 tile（如 tile32）下因 block 形状与占用率
  约束发生系统性坍塌，这是 tile32 高延迟的物理机制，而非调度器问题。
  C19-2 用固定几何重放证明了该机制与 tile 数据来源无关。
- **证据路径**：`07_TILE_GEOMETRY_HARDWARE/`、`reports/epic5/…`
- **写报告注意**：该发现支撑“tile 选择应场景自适应”，但不能直接转化为性能数字。

## 8. 增量排序（C18）的收益边界：order preservation 99.86%

- **结论**：C18 增量排序在常规场景中保持 99.86% 的排序顺序一致性，
  但 new-entry ratio 仅 1.51%，不足以抵消每次插入的额外开销；
  C17-0/1/2 与 C18 均被判定为不构成独立的加速方案。
- **关键数值**：order preservation 99.86%；new-entry ratio 1.51%。
- **证据路径**：`09_FORWARD_SORT/` 目录下全部文件
- **写报告注意**：这在“为什么排序优化被拒绝”中有用，属于负面结果，不能反向输出为加速。

## 9. 多基线负面结果（Speed / FastGS / Faster-GS / Turbo-GS / SkipGS / C43）

- **结论**：在多个外部 baseline（Speed、FastGS、Faster-GS、Turbo-GS、SkipGS）上，
  C42、C43（自适应 tile）等方案要么无法移植（接口/算子不兼容），要么收益消失、
  要么效果为负。C43 因“自适应 tile 无收益 + 引入不确定性”被证伪。
- **证据路径**：
  - `15_NEGATIVE_RESULTS/NEGATIVE_RESULTS.md`
  - `17_Raw_Data/results/` 中多 baseline 测试输出的 JSON
- **写报告注意**：负面结果要写全，但结论要克制——只能写“在该实现/该数据集下未观察到”。

---

## 素材使用速查表

| 主题 | 一句话结论 | 最佳证据文件 |
|---|---|---|
| C42 | 非普遍、非叠加 | `08_C42/C42_final_decision.md` |
| 训练加速 | gsat_30k_fused_prune10_rclip05 1.164x | `17_Raw_Data/results/training/` |
| 反向稀疏性 | 1%→29%，1.7ms 地板 | `10_BACKDROP_PROFILING_C25/C25_profile_and_floor.md` |
| C49 | 修正后上限 1.13% | `12_ATTRIBUTE_DECOUPLED_C49/*` |
| 候选 C | R3 7,494 → R3.1 0 violations | `13_CANDIDATE_C/` |
| tile | tile16 双平台最优，运行时参数 | `07_TILE_GEOMETRY_HARDWARE/` |
| C18 | 仅排序保序，收益不足 | `09_FORWARD_SORT/` |
| 多基线 | C42/C43 不可移植或为负 | `15_NEGATIVE_RESULTS/NEGATIVE_RESULTS.md` |

---

*本文件由 evidence-collection agent 于 2026-09-18 生成。*
