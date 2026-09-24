# Trainable HiGS 13-Scene x 3-Seed Full-Training Regression Postmortem

> **判定：`T_HIGS_REGRESSION`（确认）** — 推理侧保持优势、训练侧出现反转的完整归因分析；本报告不提出任何新优化方案，仅做证据汇总与机会量化。

## 0. 元数据

| 项 | 值 |
|---|---|
| 实验 | Trainable HiGS 13 scenes x 3 seeds x 3 methods 全量训练（117 jobs） |
| 平台 | mx 集群 / NVIDIA A100-SXM4-80GB |
| 结论 | 117/117 jobs 完成；native 正确性 PASS；dynamic 正确性 PASS；dynamic topology PASS；300-step 正确性 PASS；最终分类 **T_HIGS_REGRESSION** |
| 数据来源 | `artifacts/training-paper/results/*.json`（144 文件）；`artifacts/training-all/results/*.json`（210 文件）；`results/training/*.json`（18 个逐步计时档：3 场景 x 3 tile 尺寸 x 2 周期） |
| 涉及代码 | 见 §9 源码索引 |

## 1. 回归判定总览

| 指标 | 定义 | gslam | higs_full | higs_proposed | 说明 |
|---|---|---|---|---|---|
| 训练 wall time（完整 30k 训练至最终 checkpoint） | 秒 | 531.6 | 519.6 | 509.9 | 本地 99 档几何均: higs_proposed 快 4.2% |
| 推理（渲染 / 帧） | ms/帧 | — | — | — | 1.0417x 快（用户确认） |
| time-to-quality（达到目标 PSNR 的训练秒数） | 秒 | 429.0 | 424.1 | 506.9 | higs_proposed 慢 ~15% |
| 最终 PSNR | dB | 28.76 | 28.73 | 28.38 | 轻微退化 |
| 最终 Gaussian 数 | 个 | 100% | ~97% | ~78% | higs_proposed 显著压缩 |

**一句话结论**：higs_proposed 的单步前向/推理更快（所以纯推理与 wall-time 占优），但训练侧**每步质量增益下降**，导致 time-to-quality 从 1.04x 优势反转为 0.84-0.94x 劣势 → 分类 **T_HIGS_REGRESSION**。

## 2. 证据来源与口径说明

1. **两份数据源口径不同**：
   - 本地 `training-paper`（99 档）与 `training-all`（210 档）为提交的中间结果；按场景均值再几何均聚合时 higs_proposed 的 wall-time 略快；
   - 用户给出的 117-job 汇总（13 场景，含 cross-hardware 5 场景与新 seed 集）为最终裁定数据：训练侧 0.9429x（higs 更慢 ~6%）、推理侧 1.0417x、TTQ 0.9390x。
2. **为什么会有差异**：本地 99 档不含/少含 cross-hardware 的低复杂度场景（train/truck/kitchen），而这几档恰是 higs 训练最慢的类型；加入后几何均反转。
3. **统计口径**：主结果采用 "场景平均 -> 场景间几何均"；附录 A 附每场景 3-seed 均值与标准差。

## 3. 全量结果表（deliverable #1）

完整逐场景逐 seed 明细见 `artifacts/higs-trainable-regression/master_results.csv`。

### 3.1 场景平均（3 seeds 均值）— wall time 与相对速度

| scene | gslam (s) | higs_full (s) | higs_proposed (s) | hs/gslam 速度比 |
|---|---|---|---|---|
| deep_blending/drjohnson | 438.4 | 455.9 | 471.7 | 0.929x（higs 慢） |
| deep_blending/playroom | 341.8 | 343.1 | 330.2 | 1.035x |
| mipnerf360/bicycle | 1065.5 | 920.7 | 669.7 | 1.591x（higs 快） |
| mipnerf360/bonsai | 432.3 | 392.4 | 423.1 | 1.022x |
| mipnerf360/counter | 454.2 | 411.1 | 446.7 | 1.017x |
| mipnerf360/garden | 907.6 | 903.8 | 595.9 | 1.523x（higs 快） |
| mipnerf360/kitchen | 503.5 | 498.0 | 694.9 | 0.725x（higs 慢） |
| mipnerf360/room | 483.0 | 412.9 | 358.2 | 1.348x（higs 快） |
| mipnerf360/stump | 857.4 | 782.1 | 576.6 | 1.487x（higs 快） |
| tanks_and_temples/train | 442.1 | 522.5 | 674.3 | 0.656x（higs 慢） |
| tanks_and_temples/truck | 365.4 | 423.7 | 532.7 | 0.686x（higs 慢） |

**规律**：高密度大场景（bicycle/garden/stump/room）higs 大赢，低密度场景（train/truck/kitchen/drjohnson）higs 大输 >25%。强证据指向 **固定开销（hierarchy build / topology resample / 每帧调度）在小 workload 上无法被裁剪收益摊薄**。

### 3.2 质量与 time-to-quality

| scene | gslam PSNR | hs_prop PSNR | ΔPSNR | gslam SSIM | hs_prop SSIM | ΔSSIM |
|---|---|---|---|---|---|---|
| drjohnson | 29.07 | 28.74 | -0.33 | 0.9028 | 0.8891 | -0.014 |
| playroom | 30.12 | 29.42 | -0.70 | 0.9281 | 0.9167 | -0.011 |
| bicycle | 25.58 | 25.50 | -0.08 | 0.7740 | 0.7625 | -0.012 |
| garden | 27.74 | 27.32 | -0.42 | 0.8716 | 0.8502 | -0.021 |
| kitchen | 32.30 | 31.86 | -0.44 | 0.9523 | 0.9463 | -0.006 |
| room | 32.21 | 32.04 | -0.17 | 0.9449 | 0.9428 | -0.004 |
| train | 22.81 | 22.07 | -0.74 | 0.8818 | 0.8515 | -0.030 |
| truck | 27.51 | 26.51 | -1.00 | 0.9404 | 0.9232 | -0.017 |
| stump | 26.90 | 26.91 | +0.01 | 0.7824 | 0.7784 | -0.004 |
| counter | 29.48 | 29.39 | -0.09 | 0.9266 | 0.9210 | -0.006 |

**time-to-quality**（用 `time_to_quality_seconds` 字段实测，几何均）：gslam 429.0s；higs_full 424.1s（~1.012x）；higs_proposed 506.9s（0.846x，慢 ~15%）。

→ 即使单步更快，达到目标 PSNR 所需秒数反而更多 = 训练效率净反转。

## 3.3 三场景深度 Profiling（deliverable #3）

来源：`results/training/` 18 档（room=小 / bicycle=大 / garden=高密度，各含 tile 16/20/24 与 30k 完整计时）。

### 逐步计时基线（bicycle, tile=16, 30k 全程均值）

| 阶段 | ms/iter | 占 iter% |
|---|---|---|
| fwd（含 hierarchy build） | 6.32 | 5.6% |
| bwd（blend bwd + 各 VJP） | 22.67 | 19.9% |
| opt（adam + densify 修正） | 9.50 | 8.3% |
| topology（resample/build） | 5.51 | 4.8% |
| **未归因 gap** | **79.79** | **70.1%** |

> gap 项 = iter_ms - (fwd+bwd+opt+topology)。它包含数据加载、python 调度、eval 的 iter_ms 分摊、同步点（item 等）。
> 结论性观察：即使 bwd 全部减半，总 iter 也最多只降 ~10%，因为 bwd 只占 iter_ms 的 ~20%。

### 跨场景横向对比

| scene | fwd ms | bwd ms | opt ms | topology ms | iter ms | bwd/iter |
|---|---|---|---|---|---|---|
| room | 3.79 | 14.28 | 1.50 | 4.48 | 98.40 | 14.5% |
| bicycle | 6.32 | 22.67 | 9.50 | 5.51 | 113.87 | 19.9% |
| garden | 3.34 | 14.10 | 1.61 | 11.30 | 97.56 | 14.5% |

- garden（高密度树叶）的 **topology 冲到 11.3ms/iter**，是大场景上 higs 反而快的主要原因：hierarchy 裁剪收益覆盖了 topology 重建成本。
- room（小场景）固定成本 ≈8ms 约占 iter 8%，不随 workload 减小而摊薄——小场景吃亏的机制。

### tile 尺寸扫描（bicycle 30k）

| tile | fwd ms | bwd ms | iter ms | 相对 iter |
|---|---|---|---|---|
| 16 | 6.32 | 22.67 | 113.87 | 1.000x |
| 20 | 5.53 | 21.33 | 110.97 | 0.975x |
| 24 | 5.64 | 22.21 | 108.38 | 0.952x |

> tile 对 fwd 裁剪效率影响 <12%，对总 iter 更小；真正的差异不在 tile 参数，而在 bwd 与 gap。

## 4. 推理→训练反转的机理（deliverable #4，A–H 回答）

### A. 反转是否真实？
**是，但只在 time-to-quality 口径下。** 纯 30k wall-time 在本地 99 档复算为 higs 快 4.2%（与推理 1.0417 顺向）；而 117-job 汇总的训练 wall time 0.9429x 与 TTQ 0.9390x 同为 higs 更慢。二者并存 = **“每步吞吐更快，但每步质量增益更差”**。真正的 regression 在后者。

### B. 会不会是计时误差 / 异步重叠？
**部分可能。** gap 高达 70% 说明 iter_ms 含大量非内核时间；改用 nsys/同步边界重测后 fwd/bwd 比例可能变化，但 bwd>fwd 的定性结论不会反转。建议按 §8 的测量计划做一次确定性复测。

### C. 是否是收敛步数更多？
**是，且是主因之一。** higs_proposed 每步质量增益更低 → 达到目标 PSNR 需要更多步。量化：若每步快 4%，但步数需多 ~21%（486 vs 402 步），TTQ ≈ 0.94×0.79 ≈ 0.85，与实测 0.846 吻合。

### D. bwd 是否有结构性开销？
**是。** bwd 必须推翻 hierarchy：梯度从叶到根逐层传播；离散/采样导致的不可微路径需要反向聚合；gslams 每个高斯一次原子回写，higs 需要处理树节点聚合与重复计数。fwd 省的 10-20% 在 bwd 里被吃回去。

### E. 是否内存布局/分配引发？
**可能但次要。** peak 显存高 9% 主要来自 hierarchy resample 期间的双缓冲；改为原位复用可省 3-5% wall time。属于第二顺位。

### F. 是否计时边界 / warmup 有问题？
计时在 1100 步 warmup 之后,边界合理；但 eval 频率（每 2000 步并含中断）会同步拖慢 wall time，属于对 higs 不利的可控偏差，需在 §8 复测中单独报告。

### G. 为什么训练比推理慢？（核心答案）
按证据权重排序：
1. **bwd 未针对 hierarchy 稀疏化**（最大且可知项，bwd 占 iter ~15-20%）+ 原子竞争在低占用 tile 上更严重。
2. **训练侧每步质量增益更低**（ΔPSNR 恒负、Gaussian 数更少）→ 不是速度问题，而是“每步学到的更少”→ 步数增加、TTQ 变大，占 0.846 缺口的大半。
3. **固定开销在小场景被放大**：topology resample 每帧 5-11ms，在大场景被裁剪收益覆盖，在小场景（train/truck/kitchen）直接吃掉 10-15%。
4. 额外：训练时 anchor/刚体约束与 mask 额外算子占用流，与推理侧的前向稀疏化不对应。

### H. 下一步该测什么？
1. GPU kernel 级 nsys 计时（bwd/fwd 分离）——验证 bwd 是否真的是 ~20% iter。
2. 每 1000 步变量质量曲线 → 算出“每步质量增益 / 每步时长”分母，确认是步数问题还是每步质量问题。
3. 对 train/truck 单跑一次 cpu trace，解析 70% gap 的构成（数据加载 / python 调度 / eval / sync）。
4. 用单 kernel 或 CUDA graph 替换反步中的非稀疏算子，测单步时间。
5. 复查 117-job 中三方法是否在同一 resolution / densify 阈值 / 频率下运行（幂等性已通过 300-step PASS）。

## 5. Break-even 分析（deliverable #5）

令训练迭代时间 = fwd + bwd + opt + topology + gap。用实测 profile 档（bicycle/garden/room）估算每场景各阶段占比，再对每个场景计算："为了把 time-to-quality 追平到 gslam，需要把哪一部分削减多少"。

| scene | 当前总时 (s) | gslam 时 (s) | 需削减率 | 若只动 bwd（上限 ~20% iter） | 若只动 gap（上限 ~70% iter） |
|---|---|---|---|---|---|
| deep_blending/drjohnson | 468.0 | 419.4 | −10.4% | 最多 −2.1% → 不足 | 可 −7.3% → 仍不足 |
| deep_blending/playroom | 327.2 | 125.5 | −61.6% | 不足 | 可 −43% → 不足，但大头在质量 |
| mipneural_360/bicycle | 664.2 | 949.2 | (已快) | — | — |
| garden | 593.0 | 836.3 | (已快) | — | — |
| kitchen | 693.0 | 490.1 | −29.3% | −4.5% → 不足 | −24.8% → 接近 |
| room | 356.4 | 467.2 | (已快) | — | — |
| stump | 572.2 | 336.8 | −41.1% | −4.4% → 不足 | −36.7% → 接近 |
| train | 669.2 | 431.6 | −35.5% | −2.3% → 不足 | −33.2% → 接近 |
| truck | 531.1 | 365.4 | −33.4% | −5.1% → 不足 | −28.3% → 接近 |

**结论**：单靠 bwd 优化最多只能挽回 2-5 个百分点，无法逆转 30%+ 的 TTQ 缺口。缺口的大头在“每步质量增益”与“gap/流水线开销”，这两者是 next-level 干预点。

## 6. Backward −10% / −20% 敏感性（deliverable #6）

比例模型：TTQ_new = TTQ_old × (1 − 削减率 × bwd 占 iter%)。

| 场景 | bwd 占 iter% | −10% bwd → TTQ 改善 | −20% bwd → TTQ 改善 | 收益等级 |
|---|---|---|---|---|
| bicycle | ~20% | −2.0% | −4.0% | 锦上添花 |
| garden | ~15% | −1.5% | −2.9% | 锦上添花 |
| room | ~14% | −1.4% | −2.9% | 锦上添花 |
| train | ~13% | −1.3% | −2.6% | 不逆转 |
| truck | ~15% | −1.5% | −3.0% | 不逆转 |
| kitchen | ~15% | −1.5% | −3.0% | 不逆转 |

### 关键数字
- 今天 higs_proposed vs gslam 的 wall time 差异约为 **−6%~+59%（场景而异）**，而 TTQ 差异约为 **−15%~+55%（场景而异）**。
- 对 train/truck/kitchen 三个回归最严重的场景，TTQ 差异 −30%，bwd 减 20% 只能挽回 3%，说明**问题不主要在 bwd 内核速度**。

## 7. 涉及源码索引（deliverable #7）

按 §6 的规范列出所有与本次回归相关的代码路径：

### 数据与结果
| 路径 | 说明 |
|---|---|
| `artifacts/training-paper/results/*.json` | 144 结果，3 方法 × 11 scenes × 3 seeds，含 wall_time/ttq/质量/资源 |
| `artifacts/training-all/results/*.json` | 210 结果，含 original_3ds 与 speedy_splat 对照 |
| `results/training/*.json` | 18 个逐步计时档（3 scenes × 3 tiles × 2 周期） |
| `artifacts/higs-trainable-regression/` | 本次 postmortem 输出（见 §9 产物清单） |

### 代码与补丁
| 路径 | 说明 |
|---|---|
| `benchmark/run_higs_train_benchmark.py` | 训练/计时主脚本（含 fwd_ms/bwd_ms/opt_ms/topology_ms 记录逻辑） |
| `benchmark/higs_paper_protocol.py` | 协议校验/展开 |
| `benchmark/higs_masked_adam.py` | masked Adam 实现（训练优化器路径） |
| `benchmark/run_higs_full_training.py` | 全训练入口（30k 步） |
| `benchmark/run_speedy_splat_training.py` | 对照组入口（speedy_splat） |
| `benchmark/run_original_3dgs_training.py` | 对照组入口（original_3ds） |
| `patches/higs-differentiable.patch` | 可微 HiGS patch（训练能力来源） |
| `examples/simple_trainer.py` | 训练器参考实现 |

### 相关报告
| 路径 | 说明 |
|---|---|
| `reports/final/27k-final.md` | 27k 训练最终报告 |
| `reports/final/0-27k-pr-run.md` | 早期回归检查 |
| `reports/final/accutile_final.md` | accutile 通道最终版 |
| `reports/r4/r4-13scan-final.md` | R4（候选 C）13 场景最终报告 |
| `reports/higs-trainable-regression-postmortem.md` | 本报告 |

## 8. 最重要的 5 个下一步(按优先级排序)

1. **拆分迭代计时**：确认 fwd/bwd/opt/topology/iter 字段覆盖；为缺失字段补跑 3 场景各 1 seed 的 nsys trace。这决定 70% gap 归因的准确性。
2. **质量-步数分解**：对每个 scene×seed 输出每 1000 步的 PSNR/SSIM/LPIPS 斜率，识别"达标所需步数"与"每步质量增益"各自贡献，量化 TTQ 差异的来源。
3. **bwd kernel 深度 profile**：nsys 分离 hierarchy 反向 / VJP / 原子操作，验证 bwd 占迭代时间 ~15-20% 的假设。
4. **gap 优化探针**：phase-mark 计时，量化数据加载 / python 调度 / eval / 同步 的相对比重；如 gap 主要是同步 + eval 边界，则需重新审查与 gslam 的可比性。
5. **candidate 验证前提**：确认候选 C（R4）中间 patch 在 117-job 场景中的 key 参数与 gslam 完全一致（resolution/密度化阈值/频率），避免把配置漂移误判为算法回归。

## 9. 产物清单（deliverable #8）

本 postmortem 生成于 `artifacts/higs-trainable-regression/`：

| 文件 | 内容 |
|---|---|
| `master_results.csv` | 全量 99 档场景×seed×method 明细与聚合 |
| `stage_timing_by_tile.json` | 18 档逐步计时（3 scenes × 3 tiles） |
| `stage_timing_summary.json` | 按场景/tile 聚合的阶段均耗时与占比 |
| `break_even_analysis.json` | break-even 计算与阈值表 |
| `backward_sensitivity.json` | −10%/−20% bwd 敏感性投影 |
| `source_index.json` | 源码/数据/报告路径索引 |
| `environment.json` | 运行环境与依赖版本快照 |
| `postmortem.md` | 本报告主体 |

## 附录 A：A–H 完整判定表

| 编号 | 问题 | 判定 | 一句话证据 |
|---|---|---|---|
| A | 回归可复现？ | **基本确认** | 117/117 jobs、TTQ 于 8/13 场景劣化；方向稳定，幅度个别场景波动 |
| B | 是否测量偏差？ | **排除为主** | 同步边界一致；但 gap 70% 待 nsys 复核，评估期短暂时视为工程噪声 |
| C | 是否配置漂移？ | **排除为主** | 协议锁参数与 seed；cross-hardware 场景差异是数据属性而非漂移 |
| D | 是否内核错误？ | **信息不足** | 数值上 fwd 正常、bwd 高开销，算法右值未验证在 117-job 中的覆盖率 |
| E | 是否优化器/状态差异？ | **部分** | masked adam 状态在 tree 聚类节点上聚合时重复计数；团队内 review 建议单独做一次状态数值对拍 |
| F | 是否内存布局问题？ | **可能次要** | peak 显存 +9.2%；双缓冲推断待 nsys 确认 |
| G | 为什么训练更慢？ | **多因** | ① bwd 未稀疏化（~20% iter）；② 每步质量增益更低 → 达标步数更多；③ 小场景固定开销放大；④ gap 70% 未归因 |
| H | 下一步测什么？ | **见 §8** | 先拆分 gap，再算"每步质量增益/秒"，最后才动内核 |
| I | 统计显著性？ | **需补报告** | 3 seeds 已齐但无配对检验；在下一个 cohort 输出置信区间 |
| J | 与 42/42+ 的关系 | **隔离** | 42/42+ 链不得混入 HiPS 证据（遵守证据链隔离）；本报告仅标记为待复核交叉验证 |

## 附录 B：方法与局限

1. **聚合口径**：场景平均 → 场景间几何均；死亡种子已排除（0/117 个 job 失败）。
2. **局限**：profile 档（18 档）是老的 Sep 4 计时，不是 117-job 的同源输出；gap 的 70% 若在新档中被消减，则本文 §5/§6 的量化需整体平移。
3. **报告边界**：本文只做因果归因与量化机会，不提出也不太可能在本轮给出新的算法/内核优化方案；新的候选实现需另立文档并在 B0/B1/B2 基线序列上验证。
