# CLOSED_DIRECTIONS_20260930 — 3DGS 训练加速研究方向关闭登记表

> 目的：任何新 idea 在启动实验前必须先 grep 本表。若与已 CLOSED mechanism 高度同构 → 拒绝启动。
> 维护规则（取自全局指令 + 科研协议）：
> - 只允许记录对话/实验中**真正建立**的知识（用户明确 / 当场解释 / 已验证 / 引用材料中明确出现）。
> - 每条记录：name / final verdict / decisive metric / why killed / what must NOT be revived / artifact path / date。
> - 禁止把 approximation 写成 exact；禁止把 kernel speedup 写成 training speedup；禁止看到结果后改 gate。
> - 关闭后如需复活，必须有"新的、独立的、直接证据推翻已有 negative result"，否则拒绝。

生成日期：2026-09-30　维护者：下一阶段科研执行（DSH）

---

## 0. 已建立的 negative laws（跨方向复用，新方向 pre-screen 必过）

### LAW-PSC — global 3D work 占比极小
- 结论：global/projection/sync 类节省 ≈ 4% 量级，**small projection/global-sync savings ≠ large training speedup**。
- 代表：PSC-GS DROP。
- 新方向必须证明其作用对象是**主导**训练成本的部分（alpha-compositing / 渲染 / 反向），而非 3D 全局量。

### LAW-REPLAY — 数学可复用 ≠ GPU 便宜可物化
- 结论：mathematical reusable structure ≠ GPU-cheap materialized representation。
- 代表：ReplayGS、generic cache/replay variants 全部 CLOSED。

### LAW-IRGS — 优化器稀疏 ≠ alpha-compositing 稀疏
- 结论：optimizer utility sparsity ≠ alpha-compositing work sparsity。
- 代表：IRGS / DeltaRaster-GS DROP。

### LAW-D0 — 去除最终必死的工作不含足够 headroom
- 结论：removing eventually-dead work does not contain remotely enough 3× headroom。
- 代表：D0 waste census、dead-lineage prediction、birth-waste、simple early stop 全部 CLOSED。

### LAW-MATURE-HELDOUT — 成熟 checkpoint 上 native K 步 held-out 不下降
- 结论：在成熟 checkpoint（15K/25K）上，native 单相机 K 步 Adam 的 **held-out** loss 不下降
  （近零或负），而 **design** objective 仍下降。因此任何以
  `P_hold = D_homo_hold / D_native_hold` 为分母的 short-horizon 探针，其分母在成熟态不可靠。
- 代表：H0-FROZEN（BASE_NON_DESCT=8/8 于 6/8 个 (cell,K)）；与 SG0_BASELINE_UNSTABLE 同根因。
- 影响：H0/SG0 这类“K 步 held-out 下降”探针只在**早期** checkpoint 有信号；成熟态需用
  其他 oracle（如 design-space 或 full-convergence E2E）而非 held-out 短程下降。
- 新方向 pre-screen：若新 oracle 依赖成熟 checkpoint 的 native held-out 下降作分母，
  先验证该分母是否 > 0，否则探针无意义（同 LAW-MATURE-HELDOUT）。

### LAW-OPT-FRAC — 强基线优化器占比极小
- 结论：在 Faster-GS fused 强基线（A100，权威计时）上，优化器仅占迭代时间 5.94%（full-Adam, room@15k post-densify, N=933590, M0_BASELINE_PROFILE）到 14%（SH-Adam, 三场景, LAVOS0_HEADROOM）。任何优化器侧节省（状态压缩/稀疏/延迟）都受 Amdahl 限制，无法转化为显著 E2E 加速；流量模型证即使全 BF16（C4, 28.6% HBM 流量砍）E2E 天花板仅 1.017-1.042x < 1.05x 门。
- 代表：AMPS0（属性感知混合精度优化器状态）DROP_INSUFFICIENT_OPTIMIZER_HEADROOM + DROP_STATE_PRECISION_NUMERICS。
- 新方向 pre-screen：若作用对象是优化器状态精度/稀疏/延迟/调度，先证明优化器占比 >=0.30（当前 0.06-0.14），否则不启动。


### LAW-BEVT — 逐出现界 alpha 路由在真实 3DGS 上结构性空洞
- 结论：BES0 式「逐出现界 alpha 上/下界 + gamma 正则路由」（把高斯出现分为廉价 R 路 / 精确 D 路）在真实 3DGS 上**结构性无 R**：rho_R ~ 1e-9（room/bicycle/garden×12 dev 相机、~958M 出现、ts=16px 仅 1 次 R；ts=8/4 为 0；ts=2 亦 <=1.1e-7；gamma 上扫到 100 仍为 0）。根因：真实高斯是**各向异性薄片、短轴亚像素**（room cam1 sigma_short≈0.32px / sigma_long≈115px），任意 >=2px tile 内短轴使 alpha 在 tile 远角下溢到 0 → 每次出现 alpha 下界 lo=0 → is_R 普遍为假。逐像素透过率界（3DGS 速度的正确形式）= gsplat 既有 early-stop，相对 complete renderer 余量仅 ~1-2%（相机相关，少数视图 <=30%）。
- 代表：BES0（Bounded-Event Gaussian Splatting）逐出现界路由 FALSIFIED。
- 新方向 pre-screen：任何以「逐出现/逐高斯的 alpha 或协方差界路由」为 3DGS 训练加速主 lever 的方案，先证明 rho_R >= 0.5（即 >=half 出现满足 hi/lo<=gamma 且 lo>0）；当前实测 ~1e-9，不启动。质量感知阈值的逐像素早停是唯一可操作正向（捕获相机相关尾部）。

### LAW-ALPHAWALK — 前向 alpha-walk 成本由 q-检验+staging 主导，no-op culling 上限 ~1.2x
- 结论：真实 3DGS 前向 alpha-compositing walk 中 98.4% 的 (pixel,gaussian) 出现是 no-op（alpha<1/255，census），但**精确** no-op culling（跳过 no-op 的 exp+blend）实测仅 **1.17-1.23x**（room 1.233x / bicycle 1.183x / garden 1.173x，3 场景 x12 dev 相机，MX GPU7，bit-identical maxdiff=0、det=0）。远低于 census 的 59-70x ideal 与「several-x」期望。根因：识别 no-op 必须计算二次型 q=a*dx^2+2b*dx*dy+c*dy^2（culling 无法跳过 q），且 per-tile shared-memory staging + 逐 tile 循环 floor 占 baseline 23-37%；只有 exp+blend 被移除（~20-27% of compute）。ideal 63x = n_total/n_sig 假设 culling 免费（跳过整个 per-occurrence 成本），但 q-检验+staging floor 不可省。
- 代表：BES0 no-op culling 实测（reports/bes0_noop_culling_measured.md，bes_cull_probe/）。
- 新方向 pre-screen：任何以「跳过 no-op / alpha<cutoff 出现」为 3DGS 前向加速主 lever 的方案，先证明其作用对象是 exp+blend（可 culling 移除）而非 q-检验+staging（不可移除）；当前实测上限 ~1.2x，不启动。强 lever 须减少 q-检验/staging 成本（分层 per-pixel significant-only 列表、更便宜 culling 检验、稀疏性），而非跳过 no-op。

> 任何新方向必须先通过上述 8 条 law 的 pre-screen，否则不启动。

---

## 1. CLOSED 方向登记表

| # | name | final verdict | decisive metric | why killed | what must NOT be revived | artifact path | date |
|---|------|---------------|-----------------|------------|--------------------------|---------------|------|
| 1 | PSC-GS (projection/sync coalescing) | PSC_P0_DROP | S_upper_8=1.035 (<1.30); P_hold_pos_B8=0.513 (≤0.55) | global 3D 工作仅占 ~4%，无显著 held-out 收益 | 任何 projection/global-sync/排序合并类 kernel 节省 | PSC_P0_RUN/PSC_P0_VERDICT.json | 2026-09-30 |
| 2 | ReplayGS (梯度/中间结果回放) | CLOSED (LAW-REPLAY) | 数学可复用但 GPU 物化不便宜 | reusable structure ≠ cheap materialized repr | 任何 cache/replay 变体、中间结果物化回放 | 协议 §2；reports/ | 2026-09-30 |
| 3 | Profiled-SH / VarPro-GS | CLOSED | profile-only 无 E2E 收益 | 仅分析/逐段，无端到端训练加速 | 把 profiling 结论当加速 | 协议 §2 | 2026-09-30 |
| 4 | IRGS / DeltaRaster-GS | IR0_DROP | promote/strong 全 false（room/bicycle @20K/25K, f=0.10/0.20） | optimizer sparsity ≠ compositing sparsity（LAW-IRGS） | 优化器侧稀疏化反向 | ir0_dr0_run/out/IR0_VERDICT.json | 2026-09-30 |
| 5 | AdjointTensor A0 | CLOSED | effective_rank≈1.6–1.9，2 维即 99.99% 能量 | 张量/伴随结构低秩，无足够 headroom | 张量低秩假设、伴随降维 | A0_AdjointTensor_GS_v2_Bundle/ | 2026-09-30 |
| 6 | GRC3D (dual-res coarse-proxy gradient correction) | GRC3D_M1_DROP | kappa CI 下界 > 3/7（room/bicycle，means/quats）；cost@14K 失败 | 粗代理梯度方差不足；稀疏核开销无实测路径 | 双分辨率粗代理梯度校正、proxy Jacobian | grc3d_dsh/GRC3D_M1_VERDICT.json | 2026-09-27 |
| 7 | P17 coarse-gradient / coarse-loss | P17_P0_HARD_DROP | means_cos≈0.0005 (<0.98)；R_step≈0 (<0.8)，room+bicycle | 粗梯度和/或步长与目标几乎正交 | coarse-loss、多速率梯度 oracle、tau 敏感性 | P17_P0_Multirate_Gradient_Oracle_Bundle/_extracted/P17_VERDICT.json | 2026-09-30 |
| 8 | MG0 OneView-GS / HotCV | MG0_NO_USEFUL_VARIANCE_SIGNAL | 无 K≤32768 通过 actionable gate | 无可利用方差信号；未实现 trainer | OneView-GS 方差信号、hot-cv、单点视角 | mg0_oneview_hotcv/return/artifacts/MG0_VERDICT.json | 2026-09-30 |
| 9 | ReBudgetGS (重分配预算) | N1_SMALL_BUT_ACTIONABLE（未做强升级） | matched_cost_ratio room=1.115 / bicycle=1.040（≥1.05 仅 room）；H2≈0.5%/0% | 重分配收益小且不稳定，未达 strong gate；非 3× headroom | 重分配/再预算策略作为主收益 | N1_RETURN/ ; sb0_authoritative/SB0_VERDICT.json | 2026-09-29 |
| 10 | persistent linearization / DeltaGS | CLOSED | 与 IR0/P17 同构 | 持续线性化假设被粗梯度负结果覆盖 | 持久线性化、delta 单步校正 | 协议 §2 | 2026-09-30 |
| 11 | FreezeGS | CLOSED | 冻结几何无 held-out 收益 | 冻结不含足够 headroom（LAW-D0） | 冻结一部分几何/属性不动 | 协议 §2 | 2026-09-30 |
| 12 | EdgeBudget-GS | CLOSED | edge 预算裁剪无收益 | 裁剪边界工作不含 3× headroom | 基于 edge 的预算/裁剪 | 协议 §2 | 2026-09-30 |
| 13 | FQ-GS exp-free | CLOSED | 去指数化无收益 | 近似重写不含足够 headroom | 去掉关键指数/近似的“等价”重写 | 协议 §2 | 2026-09-30 |
| 14 | birth-waste / dead-lineage | CLOSED (LAW-D0) | 去除必死工作 headroom 不足 | 死工作预测/去除不含 3× headroom | 死枝预测、出生浪费消除 | d0_waste_census/ ; 协议 §2 | 2026-09-30 |
| 15 | simple early stop | CLOSED (LAW-D0) | 早停不含足够 headroom | 早停不是 3× lever | 简单早停准则 | 协议 §2 | 2026-09-30 |
| 16 | tile-prefix | CLOSED | tile 前缀复用无收益 | 前缀复用属 replay 类（LAW-REPLAY） | tile 前缀/前缀和复用 | 协议 §2 | 2026-09-30 |
| 17 | covariance projection-nullspace | CLOSED | 零空间利用无收益 | 投影零空间属 global 类（LAW-PSC） | covariance 零空间/nullspace 投影 | 协议 §2 | 2026-09-30 |
| 18 | generic sparse-tail rescue | CLOSED | 稀疏尾部救援无收益 | 尾部稀疏属 replay/dead-work 类 | 通用稀疏尾部救援 | 协议 §2 | 2026-09-30 |
| 19 | full-image adjoint + sparse backward | CLOSED | 全图伴随+稀疏反向无收益 | 全图伴随被 A0/低秩负结果覆盖；稀疏反向属 IRGS 类 | 全图伴随降维、稀疏反向作为主 lever | 协议 §2 | 2026-09-30 |
| 20 | generic cache/replay variants | CLOSED (LAW-REPLAY) | 缓存/回放无 GPU 便宜物化 | 数学复用 ≠ GPU 便宜（LAW-REPLAY） | 任何 cache/replay 变体 | 协议 §2 | 2026-09-30 |
| 21 | HomotopyGS (H0-FROZEN cohort) | H0_FROZEN_DROP | 成熟 ckpt native K 步 held-out 不下降（BASE_NON_DESCT=8/8 in 6/8 blocks）；E2D p_hold≤1.08、OPA −6×~−27× | held-out P_hold 分母在成熟态不可靠（LAW-MATURE-HELDOUT，同 SG0_BASELINE_UNSTABLE） | renderer continuation/eps2d-anneal/opacity-anneal 作为 held-out 短程增益 | /root/epic05_h0frozen/{H0_FROZEN_VERDICT.json,H0_FROZEN_FINDING.md} | 2026-09-30 |
| 22 | SplatMG (SG0-E10 cohort) | SPLATMG_CLOSED | E0 viability fail: room@10K K8 base_non_descent=4/8、bicycle@10K=8/8（均 >2/8）；oracle 未运行 | 同 LAW-MATURE-HELDOUT：native K 步 held-out 分母非正，coarse oracle 无意义 | coarse-space oracle(L32/L128×SIM13/AFF22)、future-displacement 粗投影作为 held-out 增益 | /root/epic05_sg0e10/{SG0E10_E0_VERDICT.json,SG0_E10_E0_FINDING.md} | 2026-09-30 |
| 23 | SecantGS (S0 temporal-span oracle) | S0_DROP | E_span 0.82–0.88（高）但 P_src median −0.036~+0.12（<0.70）、peer_win 0.0–0.036（<0.35），4 cells × m∈{2,4} 全 fail | 参数空间跨子空间能量高 ≠ 损失进展捕获：Adam 更新方差归一化后幅度相近、损失内容逐相机相关，不向前传递 | temporal update-subspace secant（用前 m 次 executed Adam delta 张成下一次）、forward-only 接受作为省 backward 的手段、把 E_span 当收益 | sg0_run_s0/out/{S0_VERDICT.json,S0_SUMMARY.json,S0_REPORT.md} | 2026-09-30 |
| 24 | VTS0 (Visibility-Topology Scheduler) | VTS0_DROP | extra_safe_skip_headroom=0.0 (<10%) 两场景(room/bicycle)；topology_forced 0.048/0.094、lineage 0.064/0.143、settle 0.061/0.101 均为加成耦本非跳过 | visibility 为唯一 exact skip 信号(zero-grad→延载 replay)；topology events 仅加强读取/复制/定算，从不让 visible row 真实梯度跳过、VOS 区间(15K-30K) topology 冻结、动态区只减 headroom | “topology-aware 调度器计 optimizer 跳过 headroom 超越 visibility”类推理；要求可证明 visible row 的 optimizer step 可证为冗余(会破坏 exactness) | vts0/{VTS0_VERDICT.md,VTS0_VERDICT.json} ; EPIC-05 A100 zero-diff trace 1..4000 | 2026-10-01 |
| 25 | AMPS0 (属性感知混合精度优化器状态 BF16 m/v) | AMPS0_DROP | f_opt=5.94%-14% << 0.20 门；流量模型 C4(全 BF16,28.6% 流量砍) E2E 天花板 1.017-1.042x < 1.05x；BF16 矩存 rel-L2 ~2% > 5e-3 门(保向 cos 0.99998) | 优化器状态精度/稀疏/延迟作为主收益；BF16 矩存储 | 任何以优化器状态精度/稀疏/延迟为主 lever 的路线；BF16 矩存储作精度优化 | amps_gs/AMPS0/ ; 2026-10-05 |
| 26 | BES0 (bounded-event 逐出现界 alpha 路由) | BES0_REALSCENE_FALSIFIED | rho_R=1.04e-9（room/bicycle/garden×12 dev，~958M 出现，ts=16px 仅 1 次 R；ts<=2px 亦 <=1.1e-7；gamma 上扫 100 仍 0）；逐像素早停余量 0.8-1.9%（<=30% 少数视图） | 真实高斯各向异性薄片、短轴亚像素 → 每 tile alpha 下界=0 → 路由全 D（LAW-BEVT）；逐像素透过率界=gsplat 既有 early-stop | 任何逐出现/逐高斯 alpha 或协方差界路由作训练加速主 lever；gamma-正则 cheap-path 近似（需先证 rho_R>=0.5） | reports/bes0_realscene_opportunity.md ; bes_opp_probe/out/ ; commit 65a0b41 | 2026-10-05 |
| 27 | BES0 no-op culling（逐像素跳过 alpha<1/255 的 exp+blend） | CULL_CEILING_1.2x | 精确 culling 实测 1.17-1.23x（room 1.233/bicycle 1.183/garden 1.173，3 场景 x12 dev，bit-identical maxdiff=0、det=0）；floor(staging) 占 baseline 23-37%、q-检验不可省；远低于 census 59-70x ideal（ideal 假设 culling 跳过整个 per-occurrence 成本） | no-op 的 98.4% 是真的，但精确移除其 exp+blend 只省 ~20-27% compute；q-检验+staging floor 主导前向 walk（LAW-ALPHAWALK） | 任何以「跳过 no-op / alpha<cutoff 出现」为前向加速主 lever；gamma-正则 cheap-path（需先证 rho_R>=0.5，见 LAW-BEVT） | reports/bes0_noop_culling_measured.md ; bes_cull_probe/ ; 2026-10-05 |
---

## 2. 当前活跃方向（非 CLOSED，勿误列入上表）

| track | 方向 | 当前状态 | 备注 |
|-------|------|----------|------|
| P1 | HomotopyGS H0-FROZEN | **CLOSED（H0_FROZEN_DROP）** | 4 成熟 cell(15K/25K) 全 fail；LAW-MATURE-HELDOUT。原 H0/H0-R5K verdict 保留不覆盖 |
| P2 | AtomBirth-GS AB0 | **BLOCKED（维持）** | canonical 5K 不存在且禁止以重生成冒充；需单独 prereg 的 stochastic-replicate 早期协议。H0-FROZEN 已证成熟态 held-out 分母不可靠(LAW-MATURE-HELDOUT)，AB0 的 held-out 指标须重新设计 |
| P3 | SplatMG SG0-E10 | **CLOSED（SPLATMG_CLOSED）** | E0 viability fail（room@10K K8=4/8、bicycle@10K=8/8，均>2/8）；LAW-MATURE-HELDOUT。原 SG0 verdict 保留 |
| P4 | baseline / novelty / CLOSED audit | IN PROGRESS | 本表 + baseline matrix + Faster-GS 对照 |

---



---

## 4. 基础设施级 BLOCKER（非机制 DROP，影响多条 track）

### INFRA-C0-NONDETERMINISM — C0 训练跨运行不可复现
- verdict: **H0_R5K_REGEN_IDENTITY_FAIL**（reproducibility BLOCKER，**非** HomotopyGS 科学 DROP；continuation 假设从未被测）。
- decisive metric: 重生成 room/bicycle 的 10K/15K/20K/25K 与 canonical SHA256 全部 MISMATCH（8/8）；同 seed 三次运行 iter1000 GS 数三者皆不同（128764/128409/128182）；全确定性模式下 det_a≠det_b。
- why killed: 自定义 CUDA rasterizer 反向的非确定性 atomicAdd 梯度累加导致跨运行发散；torch 确定性注册表无法控制；bit-exact 身份门在 frozen 二进制下**不可满足**。
- what must NOT be revived: 任何假设"重生成 frozen 轨迹可 bit-exact 复现"的路线（H0-R5K 的 5K 重生成、AB0/SG0-Early 的 canonical 5K 依赖）在不改变 frozen 二进制或不放宽 prereg gate 的前提下不得重启。
- scope: 仅影响需在 canonical 轨迹上**新建** checkpoint 的 track（H0-R5K 5K、AB0 5K、SG0-Early 5K）。仅用现有 frozen ckpt 的方向（H0 15K/25K、SG0 10K、AB0 15K、SB0）不受影响。
- artifact: h0_run/H0_R5K_FINDING.md; EPIC-05 /root/epic05_h0r5k/{H0_R5K_IDENTITY.json,H0_R5K_FINDING.md}
- date: 2026-09-30
- resolution: 需 research-lead 决策（改 frozen 二进制 / 放宽 prereg gate 均越权，须显式 sign-off）。选项 A–E 见 H0_R5K_FINDING.md。

## 3. 复活规则（硬约束）

以下情况**拒绝启动**新实验：
1. 新方向与上表任一 CLOSED mechanism 同构（先 grep 本表 + LAW-* pre-screen）。
2. 无新的、独立的、直接证据推翻已有 negative result。
3. 需要改动已冻结 gate / checkpoint / camera / loss / LR / seed 才能“看起来通过”。
4. 把 kernel speedup 当 training speedup，或把 approximation 当 exact。

允许情况：
- 出现直接推翻某条 negative law 的新实验证据（例如证明 global 3D 工作在某场景占比远大于 4%），并在本表登记新证据后，方可重启。
