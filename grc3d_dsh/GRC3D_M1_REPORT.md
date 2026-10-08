# GRC3D M0/M1 执行报告（EPIC-05 / A100）

日期：2026-09-27。裁决：**DROP**（`GRC3D_M1_VERDICT.json`）。
范围：M0 正确性 + M1 真实状态梯度方差 + 组件成本诊断。未启动 30K campaign；未做训练加速或质量结论。

对照口径（全部 kappa 与能量）：**真实场景、GPU fp32、强基线 GMAIN_FUSED、完整 64×64 块但按块抽样（每相机 32 块）、固定真实状态（精确 replay 锚点）、oracle 无关**。它是抽样块的估计量，不是全图精确值，也不是训练结果。计时一律标注 FUNCTIONAL_ONLY（共享 A100，非独占）。

---

## OBSERVED

### 1. M0 正确性（通过）

| 检查 | 结果 | 关键数字 | 原始文件 |
|---|---|---|---|
| `grc3d.test_core` 单测（CPU float64） | PASS | 4 组检查 | `local_logs_m0_test_core.txt` |
| 导出映射审计（字段级） | PASS | 六字段逐一相等；激活语义逐一相等 | `remote_outputs/m0_bicycle_30k.json` (`mapping_audit`) |
| 30K 首轮 parity | FAIL | forward maxabs 2.97e-3（984/6.22M 像素超 1e-5）；quats 梯度 rel L2 1.3e-3 | 同上 (`renderer_parity`) |
| 30K 归因后重验（addendum） | PASS | **原始（未预归一化）quaternions 下 forward 按位相等 maxabs 0.0**；全组梯度差 ≤ 5 次重复的 backward 噪声上界 | `remote_outputs/m0_bicycle_30k_addendum.json` |
| 14K bundle parity 首轮 | FAIL | 与 30K 同型（backward 非确定性特征） | `remote_outputs/m0_bicycle_it14000_bundle_parity.json` |
| 14K 噪声地板复验 | PASS | forward 按位相等；六组 rel L2 全部 ≤ 重复地板（如 quats 8.65e-5 ≤ 1.41e-4） | `remote_outputs/m0_bundle_noisefloor_bicycle_14k.json` |
| SSIM scalar / image gradient | PASS | image gradient maxabs 1.6e-10（阈值 1e-5） | `m0_bicycle_30k.json` (`ssim_parity`) |
| p=1 校正器 vs dense | PASS（loss 精确） | loss_abs_diff 0.0；quats 梯度差 1.43e-4 ≤ 同型重复地板 1.38e-4 | `m0_bicycle_30k.json` + addendum `p1_repeat_floor` |
| halo（边缘块、非整除分辨率） | PASS | 全图/patch loss ≤1e-6；完整图像梯度 ≤1e-5 | `m0_bicycle_30k.json` (`halo`) |

30K 首轮失败的两个根因均被定量归因，不是算法差异：(a) 导出 bundle 携带了已被 trainer 预归一化的 quaternions，二次归一化导致 984 个像素偏移（0.011% 像素），改用原始 quats 后 forward **按位相等**（`renorm_attribution.quat_input_diff.relative_l2 = 0.181`）；(b) gsplat backward 的原子累加非确定性，其地板用 5 次同调用重复实测（如 quats 重复 spread max 2.23e-4）。

### 2. M1 主矩阵 q=1/4（32 块 × 2 相机 = 64 块行/单元；lag16）

预注册判据：anchored p=1/8 相对 plain p=1/4 的方差比 = 7κ/3；κ ≤ 3/7 = 0.4286 才「不更差」。

| 场景/it | 组 | κ [95% CI] | raw-coarse κ | 方差比 7κ/3 | CI 上界 ≤3/7？ | 近零块 |
|---|---|---:|---:|---:|---|---:|
| bicycle 5K | means | **0.8033 [0.7185, 0.8862]** | 0.9332 | 1.874 | ✗ | 1/64 |
| bicycle 5K | quats | **0.7984 [0.7073, 0.9386]** | 0.9095 | 1.863 | ✗ | 1/64 |
| bicycle 5K | log_scales | 0.3938 [0.2366, 0.6722] | 0.7919 | 0.919 | ✗ | 1/64 |
| bicycle 5K | opacity_logits | 0.2063 [0.1080, 0.4442] | 0.5974 | 0.481 | ✗ | 1/64 |
| bicycle 5K | sh0 / shN | 0.1412 [0.0835, 0.2178] | 0.4521 | 0.330 | ✓ | 1/64 |
| bicycle 14K | means | **0.9004 [0.8639, 0.9364]** | 1.0638 | 2.101 | ✗ | 1/64 |
| bicycle 14K | quats | **0.8592 [0.7902, 0.9340]** | 1.2291 | 2.005 | ✗ | 1/64 |
| bicycle 14K | log_scales | 0.6222 [0.5127, 0.7494] | 2.7230 | 1.452 | ✗ | 1/64 |
| bicycle 14K | opacity_logits | 0.5284 [0.4295, 0.6490] | 3.3330 | 1.233 | ✗ | 1/64 |
| bicycle 14K | sh0 / shN | 0.1515 [0.0999, 0.2337] | 2.3004 | 0.354 | ✓ | 1/64 |
| garden 5K | means | **0.7294 [0.6368, 0.8220]** | 0.8486 | 1.702 | ✗ | 0 |
| garden 5K | quats | **0.7359 [0.6306, 0.8427]** | 0.8674 | 1.717 | ✗ | 0 |
| garden 5K | log_scales | 0.0733 [0.0280, 0.2511] | 0.1445 | 0.171 | ✓ | 0 |
| garden 5K | opacity_logits | 0.0933 [0.0544, 0.1544] | 0.2969 | 0.218 | ✓ | 0 |
| garden 5K | sh0 / shN | 0.0908 [0.0533, 0.1375] | 0.2637 | 0.212 | ✓ | 0 |
| garden 14K | means | **0.8136 [0.7397, 0.8783]** | 1.0299 | 1.899 | ✗ | 0 |
| garden 14K | quats | **0.6919 [0.5604, 0.8380]** | 1.2311 | 1.615 | ✗ | 0 |
| garden 14K | log_scales | 0.1339 [0.0620, 0.4127] | 0.7511 | 0.313 | ✓ | 0 |
| garden 14K | opacity_logits | 0.2473 [0.1746, 0.3719] | 1.7958 | 0.577 | ✓ | 0 |
| garden 14K | sh0 / shN | 0.1779 [0.1388, 0.2223] | 1.4145 | 0.415 | ✓ | 0 |

（每单元另报告 `all_raw_units` 汇总：bicycle 0.8032/0.9003，garden 0.7283/0.8126 —— 与 means 组几乎相同，因其主导总能量。全表 12 单元 × 7 组见 `remote_outputs/m1_matrix_table.csv`；逐块能量与块坐标、seed 在各 `remote_outputs/probe_*_b32.json`。）

近零分母：bicycle 每个单元每组有 1/64 块 fine_energy < 1e-30（已按协议显式计数，不做 None→0）；garden 为 0。

### 3. lag 诊断（q=1/4，means/quats 摘要）

| 状态 | lag0 κ(means) | lag16 | lag64 |
|---|---:|---:|---:|
| bicycle 5K | 0.8440 [0.7754, 0.9087] | 0.8033 | 0.8022 |
| bicycle 14K | 0.8801 [0.8405, 0.9172] | 0.9004 | 0.8693 |
| garden 5K | 0.7303 [0.6380, 0.8084] | 0.7294 | 0.7224 |
| garden 14K | 0.8071 [0.7323, 0.8714] | 0.8136 | 0.8298 |

**lag0（缓存零陈旧）与 lag16/64 几乎相同** → 失败不是缓存寿命问题；「当前代理 Jacobian 本身不足」（§3 对 lag0 失败的定性）。quats 同型。

### 4. 预注册补救 q=1/2（b8 pilot，同相机/状态，每图 8 块）

| 场景 | means | quats | log_scales | opacity | sh0/shN | 结论 |
|---|---:|---:|---:|---:|---:|---|
| bicycle 5K | **0.4750** | 0.3085 | 0.2175 | 0.0673 | 0.0142 | means > 3/7 → FAIL |
| garden 5K | **0.5306** | 0.5848 | 0.0018 | 0.0122 | 0.0088 | means > 3/7 → FAIL |

原始文件：`remote_outputs/remedy_q2_{bicycle,garden}_it5000_cam163_lag16_b8.json`。按 §3「仍失败则结束此候选」。

### 5. 预条件 κ（辅助，非预注册判据）

用 CURRENT 快照 Adam `exp_avg_sq`（detach，eps=1e-15）实测：bicycle 与普通 κ 基本同水平或更高（means 0.687–1.049）；garden 显著更低（means 0.292–0.342，但在 5K 状态仍 0.684–0.694）。该度量受零 v 元素支配（shs 21.6%、xyz/opacity/scaling 各约 0.8% 元素 v=0），**不能**当作完整 Adam 收敛证明。12 单元见 `m1_aggregate.json` (`preconditioned`)。

### 6. 成本（§4 模型，eta=1.2 HYPOTHESIS）

基元 = 同机强基线完整 step：bicycle 28.45 ms / garden 28.33 ms（`replay_*_14000/timing.json`，**FUNCTIONAL_ONLY**，共享 A100）。

| 状态 | u | c | h/step | T_ratio [eta=1.2] | 加速区间 | 1.3× gate (T_ratio≤0.769) |
|---|---:|---:|---:|---|---|---|
| bicycle 5K | 0.268 | 0.206 | 0.0198 | [0.593, 0.673] | 1.49–1.69× | 乐观/保守均过 |
| bicycle 14K | 0.268 | 0.344 | 0.0361 | [0.778, 0.862] | 1.16–1.29× | **均未过** |
| garden 5K | 0.275 | 0.231 | 0.0254 | [0.637, 0.721] | 1.39–1.57× | 乐观/保守均过 |
| garden 14K | 0.275 | 0.336 | 0.0313 | [0.771, 0.856] | 1.17–1.30× | **均未过** |

粗分支份额 c 随场景增大（5K→14K: 0.206→0.344 / 0.231→0.336），把 14K 状态的乐观界也推到 gate 之上。选块工作统计（32 块中心 + 5 px halo）：唯一像素 8.24%，瓦片 13.36%（1090/8160），交点 13.0–13.7%；深度复杂度 = 交点/瓦片（如 bicycle 14K total isect 4.94M）。锚点 A=24.88 MB / B=1.56 MB，生成 + D2D 复制合计约 9 ms/次（按 lag16 摊销 h）。组件中位数 + 每次原始样本：`remote_outputs/cost_profile_*.json|csv`、`remote_outputs/m1_cost_model.csv`。

---

## EVIDENCE

- M0：`remote_outputs/m0_bicycle_30k.json`、`m0_bicycle_30k_addendum.json`、`m0_bicycle_it14000_bundle_parity.json`、`m0_bundle_noisefloor_bicycle_14k.json`、`local_logs_m0_test_core.txt`、`remote_logs/m0_*.log`
- M1 矩阵：`remote_outputs/probe_*_b32.json`（24 个，含 per_block 与 seed）、`remote_logs/probe_*_b32.log`
- 聚合与预注册评估：`remote_outputs/m1_aggregate.json`（bootstrap B=4000, seed=20260927）、`m1_matrix_table.csv`
- 补救：`remote_outputs/remedy_q2_*.json`、`remote_logs/remedy_q2_*.log`
- 预条件：`remote_outputs/precond_*_curv_b32.json`（12 个）
- 成本：`remote_outputs/cost_profile_*.json|csv`、`outputs/m1_cost_model.json|csv`、`remote_logs/cost_profile_*.log`、`remote_logs/cost_sweep_{a,b}.log`（首轮组件脚本缺陷与重跑，保留）
- 回放与状态身份：`remote_outputs/replay_{bicycle,garden}_14000/{results.json,timing.json,camera_sequence.npy}`、`remote_outputs/bundles_manifest_{bicycle,garden}.json`、`remote_logs/replay_*.log`
- 环境/占用：`ENVIRONMENT_AND_OCCUPANCY_M1.json`
- 裁决：`GRC3D_M1_VERDICT.json`；交付清单：`MANIFEST_M1_DELIVERABLES.json`

## HYPOTHESIS

- eta=1.2、delta·d、zeta：未测，主区间取 0；只可能使 T_ratio 变差。
- fine fwd/bwd 中可分离的光栅份额：冻结二进制内不可观测 → r 以 [0, f_hi×fine_share] 区间报告。
- 未来稀疏/共享内核的开销：无实测路径（per-tile launch/indirection 未测），不写成已测值。
- 共享投影 + 融合 SSIM 的潜在节省：未实现、未测量。

## TEST

- 判据与阈值在测试前写定于 `DSH_EXECUTION.md`（κ≤3/7、方差比 7κ/3、DROP 触发 κ≥1、1.3× gate、q=1/2 单次补救），本轮未放宽任何一条。
- 统计：按 64 块行重采样 bootstrap（B=4000, seed 20260927），失败样本（分母 <1e-30）显式计数。
- 相机为冻结 camera_sequence 的前两个不同索引 [163, 28]（预注册），不按 κ 挑选；锚点全部来自精确 replay（lag0/16/64 分别对应 iter、iter−16、iter−64 的快照），未用参数扰动模拟。
- 首轮失败与重跑全部保留（30K parity、14K parity、cost_sweep_a/b、模块路径纠错）。

## CONCLUSION

**DROP。** 依据：

1. **GO 的 κ 判据在两场景决定性失败**：lag16 下 means/quats 的 95% CI 下界全部高于 3/7（bicycle 14K means 0.864、quats 0.790；garden 14K means 0.740、quats 0.560），bicycle 另有 log_scales/opacity 失败。means 组主导总梯度能量（`all_raw_units` κ 与 means 几乎相同），因此这是主能量组上的失败。
2. **lag0 诊断排除缓存解释**：零陈旧时 κ 不变 → 粗分辨率代理的 Jacobian 无法解释细节参数，机制层面的「双分辨率梯度校正」前提在真实数据上不成立。
3. **唯一允许的补救 q=1/2 仍失败**（means 0.475/0.531 > 3/7）→ 按 §3「结束此候选」；不搜索新代理、训练超参或采样策略。
4. 成本侧同样不可行：14K 状态即使乐观界也未过 1.3× gate（c 随场景增长），且无实测路径量化稀疏内核开销 → 成本侧为 INCONCLUSIVE_COST，14K 读取为 DROP_COST。

诚实边界：预注册的 DROP 快速触发条件（κ≥1）**未**触发（κ 在 0.69–0.90 而非 ≥1）；本裁决依据是 GO 判据失败 + 补救失败 + 成本 gate 失败。sh0/shN（及 garden 的 log_scales/opacity）确实通过 3/7，说明低频谱颜色类参数适合粗代理；但它们在总梯度能量中占比很小，不足以支撑机制成立。

后续若要重启该方向，需要同样协议下重新预注册：新的代理形式（例如带高频校正的粗渲染）、dense-only ADC 校准对照、真实 cache policy、配对多场景多 seed 的 time-to-quality——本轮不启动。
