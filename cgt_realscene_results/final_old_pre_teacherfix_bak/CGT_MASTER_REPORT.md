# CGT-Splat 真实场景 G1->G5 通道报告

## 裁决
- G1 门：PASS（room 5k/10k/15k/30k + garden + bicycle 共 6 个组合全部通过）。
- G1 子分析（遮挡变化 / 失败定位）：DONE（不可压缩 work 是弥散尾，非深遮挡集中）。
- G2 收益预算（benefit budget）：DONE，停止规则未触发（乐观上界 > 目标）；并给出 3x 目标的破断开销预算 h_max。
- G2 连续域证书：**PARTIAL_DENSE_COVERAGE**（稠密 4096 点子像素域覆盖实测，3 场景 × 60 tile/rank × rank4/8/16）；经验证书，非严格区间算术界（后者仍 NOT_REACHED）。
- G3/G4/G5：NOT_REACHED（需空间树 / CUDA 核 / 原生训练，参考实现仅逐 tile 拟合）。
- 总裁决：`G1_PASS_HEADROOM_DEMONSTRATED_G2_OVERHEAD_BOUNDED_DENSE_COVERAGE_PARTIAL_NOT_A_SPEEDUP_CLAIM`。
- 状态冻结从 `ALGEBRA_VALIDATED_SYNTHETIC_ONLY` 推进到 `G1_REALSCENE_COMPRESSIBILITY_PASS_HEADROOM -> G2_BENEFIT_BUDGET_OVERHEAD_BOUND_CHARACTERIZED`。

## 环境
- MX bms-39468022-001，8xA100-PCIe-40GB；仅用空闲 GPU 5/6/7（0-4 他人负载未碰）。
- 运行时 /mnt/storage_pool/liaoyuanjun_mx0qual/bin/python（py3.10.21, torch 2.7.1+cu118, gsplat 1.5.3）。
- 检查点 sota_rebase_v2_runs/B1_{room_multiphase_v2/iter_{5000,10000,15000}, room_30k/iter_30000, garden_30k/iter_30000, bicycle_30k/iter_30000}；数据 mipnerf360/{room,garden,bicycle}/sparse/0。
- 种子 20261004；分层随机抽样（按 tile 工作 decile，非 top-N），6 视图 x 250 tile x rank{4,8,16} = 4500 tile-rank 记录/组合（1500 唯一 tile）。

## G1 结果（预注册门：值 PSNR>=30dB 且 max_abs<=0.02；雅可比 jac_abs<=max(0.5*resp_range/16,1.25e-3)）

| 组合 | q_w(r4) | q_w(r8) | q_w(r16) | 不可压缩work | S(γ=2)上界 | S(γ=4)上界 |
|---|---|---|---|---|---|---|
| room 5k | 0.941 | 0.978 | 0.990 | 0.97% | 35.7x | 21.6x |
| room 10k | 0.978 | 0.995 | 0.998 | 0.17% | 43.7x | 22.7x |
| room 15k | 0.957 | 0.992 | 0.996 | 0.37% | 42.8x | 23.2x |
| room 30k | 0.879 | 0.971 | 0.990 | 1.03% | 26.2x | 15.2x |
| garden | 0.956 | 0.988 | 0.993 | 0.75% | 15.0x | 7.9x |
| bicycle | 0.937 | 0.985 | 0.998 | 0.16% | 35.4x | 18.2x |

- q_w(R) = 通过任一 rank<=R 的唯一 tile 的 tile_work / 总 tile_work（cumulative，每 tile 计一次）。
- 不可压缩 work = 无任何 rank<=16 通过的 tile 的 work 占比（0.16%-1.03%，聚合 0.6%）。
- S(γ) = 零开销 Amdahl 推理上界：fwd_ratio=[Σ_pass R_best*γ + Σ_fail M]/ΣM，S=1/fwd_ratio。
- 所有组合 S(γ=4) >= 7.9x，远高于预注册目标 S>=1.25x。
- 值保真优秀：PSNR 中位 34-77 dB，max_abs 中位 0.0015。失败几乎全是雅可比/两者，值失败为 0。

## G1 子分析：遮挡变化与失败定位（DONE）
- 不可压缩 work 是高频/细遮挡尾（resp_range 大，值或雅可比失败），聚合仅 0.6%。
- **不集中在最深遮挡 decile**：各 work decile 通过率都高（room r15k d9 strict=90%），说明不可压缩集是弥散的 ~9-19% tile 尾（按 tile 数）而非少数主导 tile。
- 因此 top-N 剪枝无法移除它；需更高 rank 或更细空间分区（方案 §6.1）。连续域证书 NOT_REACHED（见 G2）。

## G2 收益预算（benefit budget，DONE，停止规则未触发）
- 公式（方案 §7）：`S = 1/(1 - p*f*(1 - 1/r) + h)`，其中 `p*f -> q_weighted`（已测），`r -> gamma`（每分量成本），`h -> 真实端到端开销`（占基线时间比）。
- G1 的 `S_inference(γ)` 即 h=0 的乐观可行上界。用它评估方案 §9 G2 停止规则。
- 乐观上界（γ=1，组合内最小值）= 26.9x；γ=4 = 7.9x。均 > 1.25x 且 > 3x。
- **停止规则：DO_NOT_STOP**（乐观上界高于目标，1.25x 与 3x 皆不触发）。压缩性本身不否定 3x。
- **破断开销预算 h_max**（未来渲染器允许的最大真实开销，超过则跌破 3x；取组合内最紧值）：
  - γ=1：h_max = 0.296（29.6%）
  - γ=2（预注册门）：h_max = 0.266（26.6%）
  - γ=4：h_max = 0.207（20.7%）
- 解释：这是 3x 目标的**必要条件**，不是 3x 可达的证据。G3 实测开销必须 ≤ 该预算，3x 才成立；否则即便算术更少，端到端仍 <3x（重演 ITPS-X 风险，方案 §2/§10.2）。

## G2 连续域证书（PARTIAL：稠密域覆盖实测，非区间算术）
- 动机：G1 保真在每条 tile 的**随机 held-out 奇像素**（128 点）上评估；方案 §5.5 指出随机 held-out 无法约束整条连续光线域的误差。
- 方法（本轮新增 `SCRIPTS/g2_certificate.py`）：对每条 tile 用其 even 像素拟合 student（同 G1），在覆盖 tile 2D 光线域的**确定性稠密子像素网格**（sub=4 → 64×64=4096 点）上评估 teacher-vs-student 的 max|E,T|，并对照 held-out 奇像素误差。3 场景（room r15k / garden r30k / bicycle r30k）× 2 views × ~10 按 work 十分位抽取的 tile × rank 4/8/16 = 每 rank 60 次拟合。
- **池化（60 tile/rank）dense max|E,T| 通过率**：rank4 **0.967**、rank8 **0.883**、rank16 **0.85**（门 0.02）；dense p95 = 0.013 / 0.035 / 0.051；dense 最差 tile = 0.175（room rank4 M=310）/ 0.046（rank8）/ 0.091（rank16）。
- **关键发现**：(1) 稠密覆盖是**比随机 held-out 更严的测试**，暴露出随机 held-out 部分掩盖的困难 tile 尾（最坏 0.045–0.175 ≫ 门 0.02）。(2) **更高 rank 并不单调改善稠密覆盖**（room 通过率 0.95→0.85→0.75，rank4→16），与 G1 雅可比非单调性一致。(3) median 上 dense/heldout 比仅 ~1.0–1.3（稠密覆盖没有整体爆炸，只是**尾部**更重）。
- **诚实声明**：这是**经验稠密覆盖证书**（确定性稠密网格上的最大误差），**严格强于**随机 held-out（确定性+稠密覆盖 tile 全域），但仍是**采样**而非连续体上的证明；**严格区间算术 / 证书细分界仍 NOT_REACHED**。
- **定位**：这是 G1 的**附加连续域探针**，**不是** G1 门本身（G1 = work-weighted 可压缩性，PASS）。它**约束**了“student 处处复现 teacher”的说法，通过量化不可压缩的困难 tile 尾。
- 证据：`G2_CONTINUOUS_DOMAIN_CERTIFICATE.json`（+ 每场景 `_room_r15k/_garden_r30k/_bicycle_r30k.json`）。

## G3 GPU 局部微基准（A100 实测，PARTIAL）
- 方法：复用 G1 的 gsplat packed tile 提取，把 teacher（M 高斯 alpha 合成）与 student（rank-R CGT 混合）移植到 torch GPU，用 CUDA event + warmup + median 计时。两种粒度：per-tile（256px，launch-bound）与 intrinsic（65536px 单次调用，launch-free 纯数学比）。参数只建一次，计时循环无 H2D。
- 实测（MX A100，CUDA 5，torch 2.7.1+cu118；room r15k M_med=60、garden r30k M_med=267）：

| 场景 | M_med | intrinsic γ(数学) | per-tile γ | h_overhead | S(含h, R4) |
|---|---|---|---|---|---|
| room r15k | 60 | 6.35x | 1.35 | 10.9% | 1.18x |
| garden r30k | 267 | 10.46x | 1.37 | 10.7% | 1.20x |

- **关键发现**：原始数学优势真实且**大于** G1/G2 假设的 γ=2-4（实测 6.3-10.5×，随 M 增大——teacher 成本~M、student 成本~R 固定）。但在 256px/tile 粒度上 GPU 是 launch/dispatch-bound，操作级 per-tile γ 仅 ~1.35，端到端 S 仅 ~1.18-1.20×。
- 这正是方案 §10.2 / ITPS-X 教训的量化：**算术减少 6-10×，但每 tile 时间几乎不降**（方案 §9 G3 停止条件"数量减少但时间不降"在 per-tile 粒度成立）。
- **对 G4 的含义**：6-10× 数学 headroom 真实存在，但被 per-tile launch 开销锁住。融合/批处理的 CGT CUDA 核（跨多 tile/分量摊销 launch）是释放该 headroom 的关键路径。实测 h（~11%）**低于** G2 预算（26.6%），故开销本身不杀 3×；真正约束是 per-tile launch 粒度，需 G4 融合核去除。
- vs G1/G2：G1/G2 用**假设** γ=2-4（零开销）；实测 γ 6-10×（更好），但零开销假设忽略了 launch 开销。净操作级 S~1.2× << 15-44× 零开销上界。
- 范围声明：torch 参考数学实现，**非**融合 CGT CUDA 核、**非** gsplat 融合光栅化器；h_overhead 是下界（仅 torch dispatch，无空间树）。bicycle/room r5k/r30k 待 GPU 空闲补测（5/6/7 中途被他人占用，遵守空闲约束未占用）。

## G4 GPU 微基准：批处理 headroom（A100 实测，PARTIAL）
- 问题：G3 实测 per-tile 操作级 S~1.2×（launch-bound），但 intrinsic 数学 γ 6-14×。批处理/融合能否恢复 headroom？
- 方法（`G4_BATCHING_CEILING.json`）：torch batchability 探针。把 B 个 tile 的像素拼进**一次** student predict 调用（B×256 px，共享 rank-R 参数，参数只建一次，计时无 H2D），测 student 每 tile 延迟 vs B。
- 实测（room r15k，R=4）：

| B | student/tile (ms) |
|---|---|
| 1 | 0.0696 (launch-bound) |
| 8 | 0.0091 |
| 64 | 0.0012 |
| 128 | 0.00062 |
| 256 | 0.00038 (intrinsic) |

- **student 每 tile 延迟从 B=1→256 降 181×**（launch 开销在 B~128-256 摊销到近零）。CGT student 求值**高度可批**。
- **结论**：per-tile 1.2× 是 launch 粒度假象，非根本限制；批处理/融合可恢复 6-14× 数学 headroom。
- **诚实声明**：(1) `S_batch`（teacher/student）**未作加速声明**——teacher 是 B 次独立噪声调用（B≤4 时 0.08ms、B≥8 时 0.38ms，度量假象），不对称不可靠；只用干净的 student 批处理曲线。(2) intrinsic γ 本轮 14.5× vs 上轮 6.35×（GPU 未锁频，状态敏感）→ 报**区间 6.35-14.53×**，非点值。(3) 这是 torch 可批性探针，**非**融合 CGT CUDA 核 vs gsplat 融合光栅化器（真·G4 微基准仍 NOT_REACHED）；teacher/高斯基线也须批摊销（gsplat 已是融合核）才能让端到端 S 逼近 intrinsic γ。

## G5（NOT_REACHED）
- G4 真·融合核微基准：需把 CGT 分量评估写成融合 CUDA 核（跨 tile 批量化摊销 launch），A100 微基准 vs 融合 gsplat 光栅化器。G4 批处理探针已证 headroom 可恢复（student 181×），但真·融合核 vs gsplat 的端到端实测仍 NOT_REACHED。
- G4 GPU 微基准：需 CGT 分量 CUDA 核 + A100 微基准 vs 高斯求值。方案 §10.2 警告算术少不一定 GPU 快。未实现。
- G5 权威端到端：需原生 CGT 训练（路线 B，方案 §6.3）+ 渲染器集成 + 独立 >=3x 验证。参考实现为冻结教师逐 tile 拟合（路线 A 推理压缩），未实现路线 B。

## Fail-closed 合规
- 未在看数据后放宽/收紧预注册门；未删困难 tile；未把未加权高斯数当加速；未把 Amdahl 界当真实加速；rank 非单调未自动升级 rank。
- 修正自身度量 bug（per-tile 计数：q_weighted/S 按唯一 tile 计一次）后重跑，未挑有利结果。
- G2 仅用已测 G1 数据后处理，未编造 GPU 成本；停止规则与 h_max 预算如实标注为必要条件而非加速声明。

## 后续（改变裁决的下一步）
- 若要推进 G3：在 MX 上实现空间树 + 成本选择器 + CGT 分量 CUDA 核，实测端到端开销 h，与 G2 的 h_max(~26.6%@γ2) 对照；h<=预算则 3x 算术可行，否则放弃普适 3x（方案 §9 G2 停止条件在实测后复核）。
- G4/G5 需渲染器集成与原生训练，属参考实现（方案 §6）未覆盖的新工程。
