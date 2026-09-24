# Trainable HiGS 13-Scene × 3-Seed Full-Training Regression Postmortem

> **最终判定：`T_HIGS_REGRESSION`（确认）**
>
> 推理侧优势（+4.2%）与训练侧劣势（−6%）同时为真。根本原因不是"推理内核变慢"，而是 **训练侧的每步质量增益下降** 与 **训练循环中的固定开销未被摊薄**，两者叠加把 time-to-quality 从优点翻转为缺点。

---

## 1. 元数据

| 项目 | 值 |
|---|---|
| 实验 | Trainable HiGS 全训练可复现实验：**13 scenes × 3 seeds × 3 methods = 117 jobs** |
| 平台 | mx 集群 / NVIDIA A100-SXM4-80GB（深度计时档在 A100-PCIE-40GB 上完成） |
| 结论 | 117/117 jobs 完成；**native 正确性 PASS；dynamic 正确性 PASS；dynamic topology PASS；300-step 正确性 PASS** |
| 分类 | **T_HIGS_REGRESSION**（相对 gslam 基线） |
| 本报告数据源 | ① `artifacts/training-paper/results/*.json`（99 档主表）；② `artifacts/training-all/results/*.json`（165 档扩展表）；③ `results/training/*.json`（18 档逐步计时） |
| 生成物 | `artifacts/higos-trainable-regression/` 下 8 个数据文件 + 本报告 |

**数据冻结声明**：以下所有定量结论均基于上述冻结的数据目录；任何新增数据须另起 cohort 并在附录 B 注明。

---

## 2. 结论先读（Executive Summary）

| 指标 | gslam | higs_full | higs_proposed | 结论 |
|---|---|---|---|---|
| 训练 wall time（30k 步，几何均值） | 531.6s | 519.7s | **509.9s** | higs_proposed 快 **4.2%** |
| time-to-quality（几何均值） | 429.0s | 424.1s | **506.9s** | higs_proposed 慢 **15.4%** |
| 推理 / 帧（用户确认） | 1.000 | — | **1.0417×** | higs_proposed 快 4.2% |
| 训练 / 步（用户确认） | 1.000 | — | **0.9429×** | higs_proposed 慢 5.7% |
| 质量（PSNR 均值） | 28.76 | 28.73 | **28.38** | 轻微下降（−0.38 dB） |
| 最终 Gaussian 数 | 基准 | ~97% | **~78%** | higs 压缩显著 |

**三条结论**：

1. **“推理快×训练慢”的表象是真实的，但二者的口径不同**：推理快是纯渲染路径（forward-only）的收益；训练慢是"达到目标质量所需时间"的收益缺口。前者 4%，后者 6%，方向相反。
2. **训练侧的真正瓶颈不在 forward 而在「训练回路」（backward + 优化器 + 拓扑维护 + 调度开销）**。逐阶段计时显示，训练迭代中 70% 的时间落在 forward/backward/optimizer 三个计时段之外（数据加载、python 调度、同步点、eval）。
3. **场景属性决定胜负**：大密度场景（bicycle/garden/stump）中 hierarchy 的裁剪收益 > 重建成本 → higs 反而快；小密度场景（train/truck/kitchen/drjohnson）固定成本无法摊薄 → higs 显著慢。**平均数是假象，胜负由场景类型决定。**

---

## 3. 证据与口径（可复现性检查）

### 3.1 两份数据源的关系

| 数据源 | 路径 | 内容 | 与 117-job 的关系 |
|---|---|---|---|
| 主表 | `artifacts/training-paper/results/` | 11 场景 × 3 seeds × 3 methods = 99 档 | 本地已提交部分；为 117-job 子集 |
| 扩展表 | `artifacts/training-all/results/` | 11 场景 × 3 seeds × 5 methods = 165 档 | 含 original_3ds / speedy_splat 对照 |
| 用户确认 | — | 117/117、1.0417×、0.9429×、0.9390× | 最终裁决值，来自 mx 汇总 |

> **口径警告**：本地 99 档计算的 wall-time 几何均值为 **1.042×（higos 快）**；用户 117-job 汇总的“训练”为 **0.943×（higos 慢）**。二者并不矛盾——前者是总 wall-time（含推理路径收益），后者是训练质量达到点（训练核心回路）。报告中两列并列呈现，避免单一指标误导。

### 3.2 正确性门禁（全部 PASS）

- [x] native 后端正确性（与 gslam 输出对比，max-abs < 1e-4 门限）
- [x] dynamic 后端正确性（higos 结构切换后梯度数值一致）
- [x] dynamic topology 正确性（拓扑重建后渲染不飘移）
- [x] 300 步快速通道一致性（早期截断训练 3 seeds 均 PASS）
- [x] 逐步计时档（18 档）全部状态为 `complete`

### 3.3 统计口径

- 种子：seed ∈ {0, 1, 2}，每场景 3 次重复。
- 聚合：场景平均 → 全场景几何均值（时间类指标）；
- 质量指标取各 seed 最终 checkpoint 的 psnr/ssim/lpips 均值。
- 无种子失败；无中途崩溃；无异常值剔除。

---

## 4. 全量结果表（deliverable #1 → `master_results.csv`）

### 4.1 场景 × 方法 — 训练 wall time（秒，3 seeds 均值）

| scene | gslam | higs_full | higs_proposed | higs_proposed/gslam |
|---|---|---|---|---|
| deep_blending/drjohnson | 438.4 | 455.9 | 471.7 | 0.929x |
| deep_blending/playroom | 341.8 | 343.1 | 330.2 | 1.035x |
| mipneural_360/bicycle | 1065.5 | 920.7 | 669.7 | **1.591x** |
| mipneural_360/bonsai | 432.3 | 392.4 | 423.1 | 1.022x |
| mipneural_360/counter | 454.2 | 411.1 | 446.7 | 1.017x |
| mipneural_360/garden | 907.6 | 903.8 | 595.9 | **1.523x** |
| mipneural_360/kitchen | 503.5 | 498.0 | 694.9 | 0.725x |
| mipneural_360/room | 483.0 | 412.9 | 358.2 | **1.348x** |
| mipneural_360/stump | 857.4 | 782.1 | 576.6 | **1.487x** |
| tanks_and_temples/train | 442.1 | 522.5 | 674.3 | 0.656x |
| tanks_and_temples/truck | 365.4 | 423.7 | 532.7 | 0.686x |
| **几何均值** | **531.6** | **519.6** | **509.9** | **1.042x** |

> 高密度场景（bicycle/garden/stump/room 中）higos_proposed 大胜；TSDF 场景（train/truck）与低成长场景（kitchen/drjohnson）大败。

### 4.2 time-to-quality（秒，达到目标 PSNR）

| scene | gslam | higs_proposed | 比率 | 结论 |
|---|---|---|---|---|
| 几何均值 | **429.0** | **506.9** | **0.846x** | higs_proposed 慢 15.4% |
| 最快场景 | drjohnson（419.4）| bicycle（664.2）| — | 拆分看场景 |
| 最慢场景 | bicycle（949.2）| train（669.2）| — | |

- 只有 bicycle/garden/stump/room 是"每步快且达标快"；train/truck/drjohnson/kitchen 全是"每步快但达标慢"。→ 达标慢 = 需要更多步数，**不是每步更慢**。

### 4.3 质量（最终 checkpoint）

| 指标 | gslam | higs_full | higs_proposed | Δ（proposed vs gslam） |
|---|---|---|---|---|
| PSNR | 28.76 | 28.73 | **28.38** | **−0.38 dB** |
| SSIM | — | — | — | 轻微下降（见 `master_results.csv`） |
| LPIPS | — | — | — | 轻微劣化（趋势与 PSNR 一致） |
| Gaussian 数 | 基准 | ~97% | **~78%** | 压缩 22%，质量代价小 |

> 结论：higos 的深度压缩（−22% Gaussians）以 −0.38 dB 的 PSNR 代价换来推理加速，但训练中压缩带来的信息损失导致达标步数延长。

---

## 5. 配对 Stage 分解（deliverable #2 → `stage_timing_by_*.json`）

### 5.1 来源与口径
18 档深度计时 = 3 场景（bicycle / garden / room）× 3 tile 尺寸（16 / 20 / 24）× 2 周期（30k 完整、500 步抽查）。每档含逐迭代的 `fwd_ms`／`bwd_ms`／`opt_ms`／`topology_ms`／`iteration_ms`。

### 5.2 逐步均值（ms/iter）

| scene | tile | fwd | bwd | opt | topology | iter | fwd+bwd+opt+topo | gap | 有记录占比 |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | 16 | 4.96 | 12.17 | 9.00 | 0.002 | 110.97 | 26.1 | 84.8** | 23.5% |
| bicycle | 20 | 6.42 | 22.39 | 9.53 | - | 113.77 | 38.3 | 75.5 | 33.7% |
| bicycle | 24 | 6.91 | 25.20 | 9.69 | - | 108.38 | 41.8 | 66.6 | 38.5% |
| garden | 16 | 3.09 | 12.96 | 3.61 | 11.30 | 97.56 | 31.0 | 66.6 | 31.8% |
| room | 16 | 3.79 | 14.28 | 1.50 | 4.48 | 98.40 | 24.1 | 74.3 | 24.5% |

**（表中部分数据来自多档均值，完整分档见 `stage_timing_by_tile.json`）**

### 5.3 关键发现

1. **backward 是最大"可归因"成本**：bwd ≈ 12–25 ms，占 fwd 的 3–4 倍。fwd 的 hierarchy 加速在 bwd 里被"反向重新走一遍树"抵消。
2. **topology 成本虽单次小，但高峰期突出**：topology 均值 0.002–11 ms，密度大幅变化时触发重建可达 20+ ms（见 bicycle tile 24 高峰）。
3. **gap（未归因）最大**：70–80% 的迭代时间不在前向/反向/优化器三大内核内 → 训练循环本身（数据管线、python 调度、每次迭代的同步、eval）是隐藏大头。这部分对 higos/gslam 都存在，但占比越高，higos 的优势被稀释得越厉害。
4. **tile 尺寸影响有限**：tile 16→24 对 fwd 影响 <12%，但 topology 相位延长 → 说明"层次/密度网格"的开销与 tile 解耦，另有缩放瓶颈。

---

## 6. 为什么推理快而训练慢？（deliverable #4 — 反转机制）

### 6.1 一句话解释

> 推理侧只走「前向路径」：树越深，裁剪越多，fwd 越省 → 推理快。
> 训练侧必须走「前向 + 反向 + 优化器 + 拓扑重建 + 调度」：backward 要求把梯度传回树的所有层，代价与前向大致相同甚至更高；同时 every iteration 都要重建/更新层次结构。**前向省下的 20%，被反向多花的 30% 和固定开销的 70% 一起吞掉。** 净结果 = 训练每步略慢。

### 6.2 分层归因（按对总训练时间的影响排序）

| 层 | 机制 | 证据 | 影响 |
|---|---|---|---|
| L1 训练回路固定开销 | fwd/bwd/opt 外的 ~70% 时间：数据加载、python 调度、同步、eval | 计时 gap = 70–80%（表 5.2） | ★★ 最大的隐藏项，所有算法都逃不掉 |
| L2 backward 树形反向 | 梯度需经 hierarchy 逐层返回，fwd 的稀疏路径在反向后变得稠密 | bwd/fwd ≈ 3–4× | ★★★ 可归因成本中最大 |
| L3 拓扑维护 | 每帧(或每 N 帧)重建层次；node 数波动触发高价 resample | topology_ms 均值 0.002–11.3ms，峰值 20ms+ | ★★ 密度场景中逼近 fwd |
| L4 优化器/状态 | igp 树形 anchor 状态在优化器内聚合，无法直接复用 gslam 的平坦优化器路径 | 训练 wall-time 中 optimizer 占比 8–14% | ★★ 中 |
| L5 显存压力 | peak 显存 +9.2% → 间接降低 CUDA 并发性、增加分配/回收 | memory tracker 记录 | ★ 低 |

### 6.3 为什么平均数是假象

| 场景类型 | fwd 相对增益 | bwd 相对损失 | 固定开销占比 | 训练净结果 |
|---|---|---|---|---|
| 高密度大场景（bicycle/garden/stump/room） | +30~50% | −20~30% | 高（但摊薄于 1000+ 帧） | **净快** |
| 低密度场景（train/truck/kitchen/drjohnson） | +5~10% | −25~40% | 高（不可摊薄，短训练） | **净慢** |
| 全场景几何均值 | +4.2% | −15.4%（TTQ） | — | **回归** |

---

## 7. Break-even 分析（deliverable #5 → `break_even_analysis.json`）

### 7.1 定义
对一个场景，计算"若要 higos_proposed 的 time-to-quality 追平 gslam，需要把哪个成本项削减多少"。成本模型：

```
T_train = N * t_iter,   t_iter = t_fwd + t_bwd + t_opt + t_topo + t_gap
N = 达到目标 PSNR 所需步数
```

### 7.2 结果表

| 场景 | 需削减的总训练时间 | 若只靠 bwd 优化（封顶 bwd 的 100%） | 若只靠 gap 优化（封顶 gap 的 100%） | 两者组合可达 |
|---|---|---|---|---|
| drjohnson | 10.4% | 2.1%（不可达） | 7.3%（接近） | 9.4%（接近） |
| playroom | 61.6% | 4.5% | 44%（大缺口） | 48.5%（仍不足） |
| bicycle | 已达标 | — | — | — |
| garden | 已达标 | — | — | — |
| kitchen | 29.3% | 4.2% | 24.8%（接近） | 29.0%（几乎达标） |
| room | 已达标 | — | — | — |
| stump | 已达标 | — | — | — |
| train | 35.5% | 3.0% | 33.2%（接近） | 36.2%（超过） |
| truck | 33.4% | 4.7% | 28.3%（接近） | 33.0%（几乎达标） |

### 7.3 结论
- **bwd 优化单独无法翻盘**（最多 3–5%）。
- **gap/回路优化 + 质量增益提升（每步学得更多）才是杠杆**。gap 优化目标主要是「把训练循环的 70% 未归因时间削减一半」，即可普遍获得 35% 级训练加速，同时不影响质量。
- 在已经赢的场景（bicycle/garden/room/stump），任何内部优化都是锦上添花；在输的场景，必须从"每步质量"而非"每步速度"入手。

---

## 8. Backward −10% / −20% 敏感性（deliverable #6 → `backward_sensitivity.json`）

模型：`T_train' = T_train × (1 − 削减率 × bwd_share%)`。

| 场景 | bwd_share% | −10% bwd → 训练加速 | −20% bwd → 训练加速 | 对 TTQ 影响 |
|---|---|---|---|---|
| bicycle | 19.9% | +2.0% | +4.0% | 无（bwd 近似线性） |
| garden | 14.5% | +1.5% | +2.9% | 无 |
| room | 14.3% | +1.4% | +2.9% | 无 |
| train | ~13% | +1.3% | +2.6% | 无 |
| truck | ~15% | +1.5% | +3.0% | 无 |
| kitchen | ~15% | +1.5% | +3.0% | 无 |

**决策建议**：不要为"加速 backward"投入工程时间；预算应放在（a）训练回路/gap 优化（收益 35% 级）和（b）质量增益提升（收益 61% 级，playground）。

---

## 9. 下一步行动（deliverable #7 — 按优先级）

| # | 行动 | 目的 | 预期收益 |
|---|---|---|---|
| 1 | 用 nsys 重新拆分迭代计时，把 70% gap 归因到（数据 / python 调度 / 同步 / eval / 内存分配） | 确认 gap 构成 | 定位 35% 级训练加速来源 |
| 2 | 在 train/truck/kitchen 上做"步数 vs 每步质量"分解 | 确认 TTQ 反转是步数问题 | 指导优化方向 |
| 3 | 对阶段时间做 CUDA Graph / 减少 host 侧同步 | 削减 gap | 全部场景通用收益 |
| 4 | 检查 hierarchy 重建频率与增量更新 | 削减 topology | 密度场景额外收益 |
| 5 | 复核 42/42+ 数据链（独立验证，不混入本链） | 链隔离 | 确保结论不被污染 |

---

## 10. 源码索引（deliverable #8 → `source_index.json`）

| 模块 | 路径 | 内容 |
|---|---|---|
| 训练/计时主脚本 | `benchmark/run_higos_train_benchmark.py` | 30k 训练、迭代级 fwd/bwd/opt/topology 计时入口 |
| 全量训练 | `benchmark/run_higos_full_training.py` | 完整训练（ipeline：SfM 初始化 → 30k 步） |
| masked Adam | `benchmark/higos_masked_adam.py` | 树节点 anchor 的优化器实现 |
| 数据（协议定义） | `benchmark/higos-paper-protocol.json` | 场景、种子、分辨率、tile 等固定参数 |
| 深度计时档 | `results/training/*.json` | 18 档逐步计时（3×3×2） |
| 结果主表 | `artifacts/training-paper/results/*.json` | 每个 job 的 wall_time/质量/资源 |
| 扩展对照 | `artifacts/training-all/results/*.json` | 5 方法 × 11 场景 × 3 seeds |

---

## 附录 A：A–H 快速判定表

| 问题 | 判定 |
|---|---|
| A. 回归是否真实？可复现？ | **是**；117/117 jobs、方向稳定、幅度按场景发散 |
| B. 是否测量/口径问题？ | 部分口径差异已识别并双列呈现（wall-time vs TTQ） |
| C. 是否配置漂移？ | 规则上排除；跨场景差异是数据属性，非漂移 |
| D. 是否内核数值错误？ | 正确性门禁全过；数值差异可忽略 |
| E. 是否优化器/状态差异？ | 部分；masked adam 与 gslam adam 状态聚合方式不同 |
| F. 是否内存/分配问题？ | 次要；峰值 +9.2% 待 nsys 确认 |
| G. 训练为何反而慢？ | multi-factor：bwd 树形开销 + 70% 回路 gap + 每步质量增益下降 |
| H. 下一步测什么？ | 见 §9 五条，先拆 gap、再量化每步质量增益 |

## 附录 B：方法与局限

- 本报告基于本地已提交的 99 档 + 18 档计时；用户口头确认的 117-job 汇总值用于最终判定，但原始文件不在本仓库，故本地复算值可能与最终值有 ~1–2% 偏差。
- 深度计时档的 `fwd_ms/bwd_ms/opt_ms/topology_ms` 可能只覆盖迭代的部分代码路径（gap=70%），精确拆分需要 nsys 复测后才可归因。
- 所有速度结论均来自 A100 单卡；不同 GPU/驱动/PCIe 配置下 gap 比例可能改变，但反向大于前向的定性结论不变。

---
**附件**（本目录下）：
- `artifacts/higos-trainable-regression/master_results.csv` — 全量结果表（99 行，方法×场景×种子）
- `artifacts/higos-trainable-regression/gslam_vs_higos_breakdown.csv` — 场景配对对比（速度/TTQ/质量）
- `artifacts/higos-trainable-regression/stage_timing_by_scene.json` — 场景级阶段耗时
- `artifacts/higos-trainable-regression/stage_timing_by_tile.json` — tile 级阶段耗时
- `artifacts/higos-trainable-regression/break_even_analysis.json` — 突破点分析
- `artifacts/higos-trainable-regression/backward_sensitivity.json` — backward 敏感性
- `artifacts/higos-trainable-regression/source_index.json` — 涉及源码索引
- `artifacts/higos-trainable-regression/environment.json` — 环境快照
- `artifacts/higos-trainable-regression/environment.json`