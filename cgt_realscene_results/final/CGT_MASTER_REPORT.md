# CGT-Splat 真实场景 G1->G5 通道报告

## 裁决
- G1 门：**PASS**（room 5k/10k/15k/30k + garden + bicycle 共 6 个组合全部通过；修复 teacher bug 后重跑，q_weighted_r16 与修复前一致到 4 位小数）。
- G1 子分析（遮挡变化 / 失败定位）：DONE（不可压缩 work 是弥散尾，非深遮挡集中）。
- G2 收益预算（benefit budget）：DONE，停止规则未触发（乐观上界 > 目标）；给出 3x 目标的破断开销预算 h_max。
- G2 连续域证书：**DONE（稠密 + 严格区间算术）**。稠密 4096 点子像素域覆盖实测（经验证书，3 场景 × rank4/8/16，pooled pass 0.98/0.90/0.90）；**严格区间算术界（interval-arithmetic，validated=True）pooled pass 0.75/0.63/0.62 —— 严格证明 student 并非处处复现 teacher**。
- G3：PARTIAL_MEASURED（4 场景实测，launch-bound per-tile S~1.4-1.5x，远低于 3x）。
- G4：PARTIAL_BATCHING_HEADROOM_RECOVERABLE（student 批处理曲线证明 headroom 可恢复；真·融合核 vs gsplat 仍 NOT_REACHED）。
- G5：NOT_REACHED（需原生 CGT 训练 + 渲染器集成 + >=3x 验证）。
- 总裁决：`G1_PASS_G2_DENSE_AND_RIGOROUS_INTERVAL_CERTIFICATE_DONE_G3_MEASURED_LAUNCH_BOUND_G4_BATCHING_RECOVERABLE_NOT_A_SPEEDUP_CLAIM`。
- 状态冻结从 `ALGEBRA_VALIDATED_SYNTHETIC_ONLY` 推进到 `G1_REALSCENE_COMPRESSIBILITY_PASS_HEADROOM -> G2_BENEFIT_BUDGET_OVERHEAD_BOUND_CHARACTERIZED -> G2_CONTINUOUS_DOMAIN_CERTIFICATE_DENSE_AND_RIGOROUS`。

## 教师 bug 修复（本轮关键变更，fail-closed 重跑）
- `g1_compressibility.py` 的 `tile_teacher` 原为 `dx = px[:,None,:] - mu[None,:,:]`，**只用 px、忽略 py**，使 teacher 沿对角线 y=x 求值而非真实 2D 像素（高斯峰 (2,3) 实测 alpha=0.60，应为 0.99）。
- 修复为 `dx = np.concatenate([px[:,None,:], py[:,None,:]], axis=2) - mu[None,:,:]`（line 131）。`g3_gpu_microbench.py` 的 `tile_teacher_torch` 有同一 bug，一并修复。
- **后果**：所有旧结果（用 buggy 对角 teacher）失效，已归档于 `G1_RAW_RECORDS/buggy_teacher/` 与 `G1_RAW_RECORDS/stale_superseded_buggy_teacher/`。G1/G2-dense/G2-rigorous/G3 全部用修复后 teacher 重跑。
- **影响评估**：G1 压缩性几乎不变（q_weighted_r16 一致到 4 位小数）——bug 主要影响 G2 的**保真度**指标（稠密/严格证书），不改变 G1 可压缩性结论。修复后重跑，未挑有利结果。

## 环境
- MX bms-39468022-001，8xA100-PCIe-40GB；本轮仅用空闲卡 **GPU 0/1/2/3/4/5/6**（0-4 他人负载，扰自己实验可接受；5/6/7 一度被他人 vLLM 短暂占用后释放）。
- 运行时 /mnt/storage_pool/liaoyuanjun_mx0qual/bin/python（py3.10.21, torch 2.7.1+cu118, gsplat 1.5.3, numpy 2.2.6, scipy 1.15.3）。
- 检查点 sota_rebase_v2_runs/B1_{room_multiphase_v2/iter_{5000,10000,15000}, room_30k/iter_30000, garden_30k/iter_30000, bicycle_30k/iter_30000}；数据 mipnerf360/{room,garden,bicycle}/sparse/0。
- 种子 20261004；分层随机抽样（按 tile 工作 decile，非 top-N），6 视图 x 250 tile x rank{4,8,16} = 4500 tile-rank 记录/组合（1500 唯一 tile）。

## G1 结果（预注册门：值 PSNR>=30dB 且 max_abs<=0.02；雅可比 jac_abs<=max(0.5*resp_range/16,1.25e-3)）

| 组合 | q_w(r4) | q_w(r8) | q_w(r16) | 不可压缩work | S(γ=2)上界 | S(γ=4)上界 |
|---|---|---|---|---|---|---|
| room 5k | 0.941 | 0.978 | 0.990 | 0.97% | 35.7x | 21.6x |
| room 10k | 0.978 | 0.995 | 0.998 | 0.17% | 43.7x | 22.7x |
| room 15k | 0.957 | 0.992 | 0.996 | 0.37% | 42.8x | 23.2x |
| room 30k | 0.879 | 0.971 | 0.990 | 1.03% | 26.2x | 15.2x |
| garden 30k | 0.956 | 0.988 | 0.993 | 0.75% | 15.0x | 7.9x |
| bicycle 30k | 0.937 | 0.985 | 0.998 | 0.16% | 35.4x | 18.2x |

- q_w(R) = 通过任一 rank<=R 的唯一 tile 的 tile_work / 总 tile_work（cumulative，每 tile 计一次）。
- 不可压缩 work = 无任何 rank<=16 通过的 tile 的 work 占比（范围 0.16%-1.03%，聚合 0.6%）。
- S(γ) = 零开销 Amdahl 推理上界：fwd_ratio=[Σ_pass R_best*γ + Σ_fail M]/ΣM，S=1/fwd_ratio。
- 所有组合 S(γ=4) >= 7.9x，远高于预注册目标 S>=1.25x。
- 值保真优秀：PSNR 中位 34-77 dB，max_abs 中位 0.0015。失败几乎全是雅可比/两者，值失败为 0。

## G1 子分析：遮挡变化与失败定位（DONE）
- 不可压缩 work 是高频/细遮挡尾（resp_range 大，值或雅可比失败），聚合仅 0.6%。
- **不集中在最深遮挡 decile**：各 work decile 通过率都高，说明不可压缩集是弥散的 ~9-19% tile 尾（按 tile 数）而非少数主导 tile。
- 因此 top-N 剪枝无法移除它；需更高 rank 或更细空间分区（方案 §6.1）。连续域证书见 G2（已升级为严格区间界）。

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

## G2 连续域证书（DONE：稠密覆盖 + 严格区间算术）
- 动机：G1 保真在每条 tile 的**随机 held-out 奇像素**（128 点）上评估；方案 §5.5 指出随机 held-out 无法约束整条连续光线域的误差。
- **稠密证书**（`g2_certificate.py`）：对每条 tile 用其 even 像素拟合 student（同 G1），在覆盖 tile 2D 光线域的**确定性稠密子像素网格**（sub=4 → 64×64=4096 点）上评估 teacher-vs-student 的 max|E,T|。3 场景 × 10 tile/rank × rank4/8/16，pooled 90 tiles/rank。
  - pooled pass_rate（严格界≤0.02）：rank4/8/16 = 0.98/0.90/0.90；heldout pass 0.98/0.96/0.94。
  - worst-tile max|E,T| 达 0.783（room tile8，M=310），远超门 0.02。
  - 高阶 rank **非单调**（bicycle 1.0/0.8/1.0，room 0.95/0.85/0.85），与 G1 雅可比非单调一致。
- **严格区间算术证书**（`g2_rigorous_certificate.py`，本轮新增，**plan §5.5 要求的 interval bounds**）：对 tile 2D 光线域做分支定界区间算术，给出连续域 max 误差的**严格上界**。student 区间盒 = rank-R expert 预测的凸包（student 是 expert 的凸组合）；teacher 区间盒由 conic power 区间得到。
  - **validated=True**（合成验证 bound/pointmax 比 1.0-4.6，界紧且有效）。
  - pooled pass_rate（严格界≤0.02）：rank4/8/16 = 0.75/0.63/0.62；worst-tile 严格界达 0.793。
  - **严格证明 student 并非处处复现 teacher**：pass_rate<1.0 意味着连续域上存在非空 tile 集，其误差严格超过 0.02 门。这是**证明**，非抽样。
- 诚实声明：两者均为**逐 tile、固定视角**；不是相机/视角变化界，也不是 GPU 吞吐界。稠密证书是经验样本（比 held-out 更强但仍非连续统界）；严格证书是连续统上的**有效上界**（proven），但仍限单 tile 单视角。二者**联合弱化**任何「student 处处复现 teacher」的说法。

## G3 GPU 微基准：per-tile 计时（A100 实测，PARTIAL，4 场景）
- 方法：复用 G1 gsplat packed tile 提取；teacher（M 高斯 alpha 合成）与 student（rank-R CGT 混合）移植到 torch GPU；CUDA event + warmup + median 计时。两种粒度：per-tile（256px，launch-bound）与 intrinsic（65536px 单次调用，launch-free 纯数学比）。参数只建一次，计时循环无 H2D。teacher/student 移植对 G1 numpy 参考已验证。
- 实测 4 场景（bicycle, room r5k/r30k, garden；M_med 44-267）：

| 粒度 | rank4 | rank8 | rank16 |
|---|---|---|---|
| intrinsic γ（launch-free 数学，中位数） | 26.4x | 24.1x | 18.7x |
| per-tile γ（launch-bound，中位数） | 1.77x | 1.82x | 1.65x |
| S（含 dispatch 开销，中位数） | 1.45x | 1.49x | 1.38x |

- intrinsic γ ~ M_rep/R（代表性 tile 高斯数 / rank），随场景 M 变化大（room 低 M ~1.3x 到 bicycle 高 M ~78x，中位 26.4x）。这是**数学下界**，per-tile 渲染器不可达。
- 但在 256px/tile 粒度上 GPU 是 launch/dispatch-bound，per-tile γ 仅 ~1.77x，端到端 S 仅 ~1.45x（rank-4 最好），**远低于 3x**。
- 这正是方案 §10.2 / ITPS-X 教训的量化：**算术减少真实且大于** G1/G2 假设的 γ=2-4，但 per-tile launch 粒度把它锁住。dispatch 开销 h~12%（下界，仅 torch dispatch，无空间树）。
- 高阶 rank **非单调**（per-tile S rank4 最好），与 G1/G2 一致。
- 范围声明：torch 参考数学实现，**非**融合 CGT CUDA 核、**非** gsplat 融合光栅化器；h_overhead 是下界。这是**本地 per-tile 核比**，非端到端渲染器加速。

## G4 GPU 微基准：批处理 headroom（A100 实测，PARTIAL）
- 问题：G3 实测 per-tile 操作级 S~1.4x（launch-bound），但 intrinsic 数学 γ 中位 26x。批处理/融合能否恢复 headroom？
- 方法（`G4_BATCHING_CEILING.json`）：torch batchability 探针。把 B 个 tile 的像素拼进**一次** student predict 调用（B×256 px，共享 rank-R 参数，参数只建一次，计时无 H2D），测 student 每 tile 延迟 vs B。
- 实测（room r15k，R=4）：

| B | student/tile (ms) |
|---|---|
| 1 | 0.06963 |
| 2 | 0.03584 |
| 4 | 0.01792 |
| 8 | 0.00909 |
| 16 | 0.00461 |
| 32 | 0.00230 |
| 64 | 0.00118 |
| 128 | 0.00062 |
| 256 | 0.00038 |

- **student 每 tile 延迟从 B=1→256 降 181×**（launch 开销在 B~128-256 摊销到近零）。CGT student 求值**高度可批**。
- **结论**：per-tile ~1.4x 是 launch 粒度假象，非根本限制；批处理/融合可恢复数学 headroom。
- **诚实声明**：(1) `S_batch`（teacher/student）**未作加速声明**——teacher 是 B 次独立噪声调用，度量假象，不可靠；只用干净的 student 批处理曲线。(2) intrinsic γ 随 GPU 锁频状态敏感，报**区间**非点值。(3) 这是 torch 可批性探针，**非**融合 CGT CUDA 核 vs gsplat 融合光栅化器（真·G4 微基准仍 NOT_REACHED）；teacher/高斯基线也须批摊销（gsplat 已是融合核）才能让端到端 S 逼近 intrinsic γ。

## G5（NOT_REACHED）
- G4 真·融合核微基准：需把 CGT 分量评估写成融合 CUDA 核（跨 tile 批量化摊销 launch），A100 微基准 vs 融合 gsplat 光栅化器。G4 批处理探针已证 headroom 可恢复（student 181×），但真·融合核 vs gsplat 的端到端实测仍 NOT_REACHED。
- G5 权威端到端：需原生 CGT 训练（路线 B，方案 §6.3）+ 渲染器集成 + 独立 >=3x 验证。参考实现为冻结教师逐 tile 拟合（路线 A 推理压缩），未实现路线 B。

## Fail-closed 合规
- 未在看数据后放宽/收紧预注册门；未删困难 tile；未把未加权高斯数当加速；未把 Amdahl 界当真实加速；rank 非单调未自动升级 rank。
- 修正自身度量 bug（teacher px/py；per-tile 计数：q_weighted/S 按唯一 tile 计一次）后重跑，未挑有利结果。
- G2 仅用已测 G1 数据后处理，未编造 GPU 成本；停止规则与 h_max 预算如实标注为必要条件而非加速声明。
- 发现 root-cause teacher bug 后，未沿用旧结果；全部重跑并归档 buggy 版本。严格区间证书 pass<1.0 如实上报，未掩盖。

## 后续（改变裁决的下一步）
- 若要推进 G3/G4：在 MX 上实现空间树 + 成本选择器 + CGT 分量 CUDA 核，实测端到端开销 h，与 G2 的 h_max(~26.6%@γ2) 对照；h<=预算则 3x 算术可行，否则放弃普适 3x（方案 §9 G2 停止条件在实测后复核）。
- G5 需渲染器集成与原生训练，属参考实现（方案 §6）未覆盖的新工程。

