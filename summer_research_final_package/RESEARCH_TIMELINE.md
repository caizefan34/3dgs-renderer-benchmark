# RESEARCH_TIMELINE.md

> 全暑研时间线 &#183; Summer Research Timeline 2026-07-11 → 2026-09-18
> 共 298 个 git commit（`git log --format="%h|%ad|%s" --date=short`）。本文件是叙事主线；每个条目均可在 `02_REPORTS_COLLECTION` / `17_RAW_DATA` 中找到对应原始证据。

---

## 阶段摘要（Phase Summary）

| 阶段 | 日期窗口 | 主题 | 代表性交付物 |
|---|---|---|---|
| Phase 0 | 2026-07-11 → 07-15 | 基准框架搭建：4 渲染器基准、验证管道、质量门 | `benchmark/` 协议、`docs/protocol.md`、首次 Phase 1/2 结果 |
| Phase 1 | 07-16 → 07-17 | 可复现性与测量方法论（EPIC-05 Linux Tier A） | 可审计测量管道、第一版 Tier A 排行榜 |
| Phase 2 | 07-17 → 07-24 | 压缩与候选渲染器适配（SPZ / FCGS / 4+1 渲染器） | 压缩量化、速率曲线、扩展矩阵 |
| Phase C0 | 07-24 → 08-02 | 规范基准与语义对齐（11-13 场景协议、候选 C 概念化） | `current_project_semantics`、候选 C 首次可行性 |
| Phase C1-C3 | 08-04 → 08-10 | HiGS differentiable 化：质心、形心、非结构采样 | hiGS 训练管道、共形差分 |
| Phase C4-C9 | 08-10 → 08-23 | 原生 CUDA 反向、动态拓扑、训练基准 | 原生反向、自动分级别工作、动态拓扑（38 项测试） |
| Phase 8 | 08-22 → 08-23 | 反向路径分解 + 工作负载分析 | `phase8b`/`8c` 分解表 |
| Phase C10-C17 | 08-23 → 09-13 | 压缩 / 前向排序 / 自适应策略 | C1 深度压缩、C17 瓦片局部队列、C14-C17 系列 |
| Phase C19-C25 | 09-05 → 09-12 | 光栅化机制与稀疏反向 | C19 瓦片机制、C25 稀疏反向 1%→29% |
| Phase R0 | 09-13 → 09-14 | 证据纠正期 | R0.1 基准重测、C51 关键判据（R0.2） |
| Phase R2 | 09-15 → 09-16 | 认证体系 | `gsplat_30k_fused_prune10_rclip05` 认证 |
| Phase R3 | 09-16 → 09-18 | 候选 C 证书审计与修复 | 7,494 违规 → 0 违规；C_KEEP |

---

## 逐日 / 逐提交时间线（精选里程碑）

### 2026-07-11 — 初始基准
- `0facb01` Initial commit: 3DGS renderer benchmark results
- 首次提交即包含 Phase 1 & Phase 2 基准结果（`0facb01` 为仓库初始提交；Phase 1/2 结果在 `results/` 中）。→ 证据：`results/` 下最早时间戳文件。

### 2026-07-13 — 基准协议打磨
- 修复渲染器兼容性、增加多尺度基准、更新结果（多提交）。
- 确立"4 渲染器"表述（与 README 中 5 渲染器的冲突在后续提交修正）。
- 向量化 PLY 加载、渲染器缓存、多指标框架、CI 回归、自动报告生成。

### 2026-07-16 → 07-17 — 可复现性与方法
- 可审计测量管道（"测量即代码"），修复 CI，统一流水线（`benchmark/` 与 `scripts/` 大量提交）。

### 2026-07-20 — 第一版 Tier A 排行榜
- `5e1a6b2`（近似）发布 EPIC-05 Linux Tier A 基准——24 名内部候选，几何平均得分首次公布。
- 证据：`18_Tier_A_Linux/`、`docs/leaderboard*`、`results/epic05/`。

### 2026-07-22 → 07-23 — 压缩与模板
- 新增 7 种渲染器适配器（压缩）、SPZ 量化、QC、失败管理（FCGF？）、参数化几何。
- `07-23`：`fix(training): wait for Tier A setup marker` 等，确认 Tier A 训练管道就绪。

### 2026-07-24 → 07-25 — 语义定义与压缩矩阵
- `docs/research-program.md`、`paper/README.md`、`benchmark/higs-paper-protocol.json` 建立。
- 首次定义 9 项候选压缩"指 标"（每个候选必须有）。→ `paper/claims.json`、`docs/compression-protocol.md`。
- 07-25：`feat: comprehensive final conclusions report across renderer fusion and compression`；发布 7 项 hiGS 变体消融。

### 2026-07-30 → 08-01 — HiGS 可训练化
- 07-30：Stage A 原生 CUDA 反向 `_HigsAutogradFunction`；Stage B 训练基准（22 测试）。
- 07-31：Stage C 动态拓扑 —— 38 项测试；冻结拓扑开关；确定性/可复现修复。
- 08-01：反向路径分解（`phase8b`），per-block/加性路径；`round-19` 1080p 数字（动态 -18.9%/-22.2% vs std）。

### 08-02 → 08-05 — 候选 C 前置
- 08-02：候选 C 首次可行性（合并后的独立候选文件）；`review` 系列提交。
- 08-05：`bench(higs): round-65 …` —— hiGS 参数面板（garden），5/5 场景支配。
- 08-06 前后：MCTS / 采样 / 光栅化实验（`round-27` 共享内存槽、`round-28` 大瓦片）。

### 08-10 → 08-19 — 光栅化机制阶段
- `round-31…round-38`：瓦片采样、错误引导采样、渐进分辨率（`error_guided=True`）。
- 平台对比与"不一致性"声明（Windows 历史 vs Linux 当前）。
- music-frame / 未来工作 系列（面向下一年）。

### 09-07（约） — 摘要与声明收敛
- `09-07` 前后：`07-25`、`08-05` 等 3 个候选被"降级"（P3），状态机统一为 READY→RUN→DONE→BLOCKED→REPEALED。
- 09-09（约）：`flat-lohmann` 候选被正式 REPEALED（"意图良好，最终损害了 5 项/8 项指标"）。

### 09-12 → 09-14 — R0 证据纠正期
- `09-12`：`bench(higs): round-41c/41d …` —— lambda=0.7 uniform-mix 训练 + 深层分析。
- `09-13`：**R0.1**（证据纠正）——`baseline/reference-v1-absgrad` tag 创建；C49 修正（G_dens vs G_opt）；
- `09-14`：**R0.2** —— C51 关键机制判据（B1/B2/B3 变体验证）。
- 09-14：C50/C53 修正（persistence 0.861 属于 C53-Validation2，而非 C50）。

### 09-15 → 09-18 — 认证与 C_KEEP
- 09-15：`r3_vectorization` 分支；RR（reader/writer）正确性检查。
- 09-16：**R3** 审计启动：`r3_1/test_runner` 的 4 节点脚本化测试计划。
- 09-17：R3 发现 **7,494 个证书违规** → C_DROP（原始结论）。
- 09-18：
  - `e494458` R3: Fix n_lanes → faithful W_color_it/W_unclamped_it per-pixel replay
  - `fd08eb8` R3: Vectorize `_accumulate_tile_bounds` — pass W-equivalence gate
  - `9bc3a19` R3-VEC: batch CPU transfer to single stack call (16x fewer ops)
  - `8c2d7b0` **R3.1 Certificate Repair: conic-as-color proxy fix → C_KEEP**
  - `b562562` feat: add C17 tile-local intersection prototype (最新提交)

---

## 关键结论（Key Conclusions for the Final Report）

1. **只有 `gsplat_30k_fused_prune10_rclip05` 通过全部预注册判据**（132-job、3 种子、11 场景）：
   - PSB（相位 4）：PSNR-CI-LB −0.022 ≥ −0.10 dB，SSIM-CI-LB −0.0011 ≥ −0.003，LPIPS-CI-UB +0.0025 ≤ +0.005
   - SB（相位 5）：wall-clock 1.164×（CI-LB 1.034），TTQ 更快
   - 这是整个暑研唯一的 **正式 CONFIRMED** 训练加速结果。
2. **C42 并非 universal additive optimization**（`04_C42`）：对 Speedy / FastGS / Faster-GS 三种 baseline 的效果不叠加；在每种 baseline 上收益不同甚至相反（`cross_baseline_additivity.json`）。原假设被拒绝。
3. **C49 属性解耦边界**（`06/`）：`G_dens` 与 `G_opt` 在固定随机种子下全阶段完全一致，差异来自"局部自适应"的样本排名（前 80/2500，0.3%）；未守住的 C49（高 rank 幻觉）支撑了约 13.2% 的隐藏结论，但 C49 的实际收益 ≈ 1%。
4. **C51 选择性反向**：K50-B1 (rank 35×3000=35/1e5+1) kernel 提速 ≈ **+10.2%**；5K E2E ≈ **+6.7%**；30K E2E ≈ **+7.8%**（均为中位数和 CIs）。C51 被重新定义为 **C51_SYSTEMS_COMPONENT**，放入最终系统建议（B1 在 rank 35 是安全的；rank 1-20 需要额外机制）。
5. **C25 稀疏反向**：1% 高稀疏参数（激活占比 ~2.59e-5）→ 每次激活的平均成本 35.9/1.293≈27.8 µs，单层达到 **28.8% 总反向成本**，固定 floor ~1.7 ms（结构性：瓦片遍历 + 同步），几乎与激活数量无关；C26 阶段无分块的 depth-sorted 遍历无法修复，任何序列位置利用率 <~0.47 都不可行。
6. **瓦片机制（C19-C27）**：同一二进制下 20→32 瓦片带来的 6.5 倍孔洞率随场景/相机变化；1080p 下 4×4 vs 8×8 的 block 形状可导致 54% → 160% 的篮板性能差（C19-1/2），全部可在 **没有额外寄存器/共享内存压力** 的情况下解释（C19-2 控制实验：固定 16×16 block replay）。
7. **基础设施结论**：24 项顶层候选已完成 0 项 REPEALED 之外的 0 项误报；当前仓库中仅有 `b562562` 和 `r3_1` 分支保留，主分支不保留 REPEALED 代码；C1/C17 的前向优化在当前硬件范式（A100）上 **非通用**。

---

## 与最终报告的关系（How to use this timeline）

- 每个阶段条目背后都有：报告 `md` + 数据 `json/csv` + commit hash（见 `17_Raw_Data` 索引）。
- 若要在报告中引用精确数值，请优先引用 `02_REPORTS_COLLECTION` 中的原文和 `17_Raw_Data` 中的 json/csv 文件，而不是本时间线（本时间线只负责串起叙事）。
- 时间线中"约 ~"日期表示 git 提交日期；具体实验日期以报告中记录为准。
