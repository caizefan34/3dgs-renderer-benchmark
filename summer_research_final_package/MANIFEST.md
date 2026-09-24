# MANIFEST.md — Summer Research Final Evidence Package

> 本文件是证据包的**目录索引**。它告诉你每一份证据在哪、是什么、以及如何与最终报告对应。
> 所有路径相对本文件所在目录 `summer_research_final_package/` 解释；
> 原始仓库路径以 `→ <repo>/...` 标注，供在仓库内交叉验证。
> **最后更新：2026-09-18（v2）** — 目录已按实际内容重新对齐。

---

## 0. 包结构总览（实际布局）

```
summer_research_final_package/
├── 00_README_FIRST.md              ← 给撰写 AI 的使用说明（先读这个）
├── 01_PROJECT_OVERVIEW/            ← 项目概述与研究问题
│   └── project_overview.md
├── 02_TIMELINE/                    ← 时间线叙事
│   ├── RESEARCH_TIMELINE.md
│   └── PHASE_LOG.md
├── 03_FINAL_MATRIX/                ← 最终结果矩阵
│   ├── FINAL_RESULT_MATRIX.csv
│   └── MATRIX_DICTIONARY.md
├── 04_CRITICAL_FINDINGS/           ← 关键发现速览
│   └── CRITICAL_FINDINGS.md
├── 05_PROTOCOL_AND_DATASETS/       ← 实验协议、数据集、硬件队列（48 文件）
├── 06_Backdrop_C25/                ← C25 反向/深度分解 证据
│   └── EVIDENCE_DIGEST.md
├── 07_TILE_GEOMETRY_HARDWARE/      ← 瓦片尺寸/硬件机制研究（EVIDENCE_DIGEST + README）
├── 08_C42/                         ← C42 研究（公式 + 13 场景 + 跨 baseline）
├── 09_FORWARD_SORT/                ← C1 深度压缩、C17-1、C18 增量排序
├── 10_C51_SELECTIVE_BACKDROP/      ← C51 选择性反向（B1/B2/B3、门 A/B/C/D）
├── 11_C49_ATTRIBUTE_DECOUPLED/     ← C49 属性解耦（G_dens vs G_opt）
├── 12_CANDIDATE_C/                 ← 候选 C 证书（R3 → R3.1 → R4）
├── 13_TRAINABLE_HIGS/              ← 可训练 HiGS 架构与 13 场景
├── 14_NEGATIVE_RESULTS/            ← NEGATIVE_RESULTS.md（被否定的假设）
├── 15_APPENDICES/                  ← 附录（原始数据索引、工具说明）
├── 16_REPRODUCABILITY/             ← REPRODUCABILITY.md（环境、命令、种子、哈希）
├── 17_RAW_DATA/                    ← 原始数据索引（INDEX.md）
├── MANIFEST.md                     ← 本文件
├── version-info.json               ← 快照元数据
├── SHA256SUMS.txt                  ← 包内文件校验和
└── README_BRIEF.md                 ← 摘要：给撰写 AI 的概要
```

> 说明：部分早期目录名（如 `06_Backdrop_C25`、`13_TRAINABLE_HIGS`）在迭代中出现
> 了编号漂移；本清单以文件实际位置为准。撰写报告时**以本清单 + 各目录内
> `EVIDENCE_DIGEST.md` 的文件指针为准**，不要依赖旧版目录编号。

---

## 1. 顶层文件

| 文件 | 说明 |
|-----|------|
| `00_README_FIRST.md` | **撰写 AI 首先阅读。** 包含报告写作规则、证据使用纪律、禁止事项。 |
| `MANIFEST.md` | 本文件。 |
| `README_BRIEF.md` | 摘要：项目一句话、三条 track、必引文件清单。 |
| `RESEARCH_TIMELINE.md` | 全暑研时间线（2026-07-11 → 09-18，298 commits）。 |
| `FINAL_RESULT_MATRIX.csv` | 机器可读的最终结果矩阵。 |
| `version-info.json` | 打包元数据（时间、repo HEAD、格式版本）。 |
| `SHA256SUMS.txt` | 包内全部文件的 SHA-256 校验和。 |

---

## 2. 分目录说明

### `01_PROJECT_OVERVIEW/`
- `project_overview.md` — 研究画像：3 条 track、24 位候选、里程碑、结论摘要。

### `02_TIMELINE/`
- `RESEARCH_TIMELINE.md` — 叙事时间线（阶段、日期、提交锚点如 `b562562`）。
- `PHASE_LOG.md` — 逐阶段日志：每个阶段的关键动作、产出、判定。

### `03_FINAL_MATRIX/`
- `FINAL_RESULT_MATRIX.csv` — **核心交付物**：每候选一行：判定 / 速度 / 质量 / 证据。
- `MATRIX_DICTIONARY.md` — 字段字典：判定枚举（KEEP / DROP / FALSIFIED / REGRESSION /
  BLOCKED / CONFIRMED）、速度分母定义（kernel / forward / e2e / wall）、质量门定义。

### `04_CRITICAL_FINDINGS/`
- `CRITICAL_FINDINGS.md` — 9 条最重要的发现，每条带证据路径（见下节摘要）。

### `05_PROTOCOL_AND_DATASETS/`
- `EVIDENCE_DIGEST.md` — 实验协议、数据集定义、硬件 cohort 规则。
- 内含 `docs/` 与 `configs/` 的镜像：协议文档与参考配置。

### `06_Backdrop_C25/`
- `EVIDENCE_DIGEST.md` — C25 反向/深度分解（1% 参数 ≈ 29% 成本、1.7ms 地板）。

### `07_THE_GEOMETRY_HARDWARE/`
- `EVIDENCE_DIGEST.md` — tile size / 占用率 / 缓存行研究。
- `README.md` — 目录导航说明。

### `08_C42/`
- `EVIDENCE_DIGEST.md` — C42 定义、公式、13 场景原始表。
- `C42_final_decision.md` — **最终判定：非 universal、非叠加（保留为 tuning tool）**。

### `09_FORWARD_SORT/`
- `EVIDENCE_DIGEST.md` — C1 深度压缩 + C17/C18 排序研究（<1% 收益）。

### `10_C51_SELECTIVE_BACKDROP_*` / `11_C49_ATTRIBUTE_*`
- 见 `EVIDENCE_DIGEST.md` 与 `README.md`。C51 选择性反向最终接入建议。

### `12_CANDIDATE_C/`
- `R3_audit.md` — 7,494 违规计数方法与完整清单。
- `R3.1_root_cause.md` — conic-as-color 缺陷根因链与修复公式。
- `R3.1_validation.md` — 90 次测量，0 违规，最坏比率 0.9998，VEC pass。
- `R4_cuda_status.md` — CUDA 实现已完成，编译/训练验证未完成（HONEST 状态）。

### `13_TRAINABLE_HIGS/`
- `architecture.md`、`correctness.md`（67/67、112/112、117/117 测试）、
  `results_13scene.md`、`T_HIGS_REGRESSION.md`。

### `14_NEGATIVE_RESULTS/`
- `NEGATIVE_RESULTS.md` — 所有被否定的假设：C1、C17-0/-1/-2/-3、C18、
  C43、C49 独立、C51-B2、Turbo-GS、SkipGS、flat-lohmann（REPEALED）、BloomGPU（REPEALED）等。

### `15_APPENDICES/`
- 工具脚本与辅助材料（`_tools` 的历史版本说明）。

### `16_REPRODUCABILITY/`
- `REPRODUCABILITY.md` — 环境（EPIC-05 A100 vs RTX 5070 Laptop）、命令、种子、哈希。

### `17_RAW_DATA/`
- `INDEX.md` — 原始数据索引：`results/`、`reports/` 中每个 JSON/CSV 的用途与来源。
- `version-info.json` — 快照元数据副本。

---

## 3. 原始仓库 ↔ 包内映射（快速交叉验证）

| 仓库路径 | 包内位置 |
|---|---|
| `reports/reference-v1-baseline-lock.md` | `06_Backdrop_C25/EVIDENCE_DIGEST.md`（C25 目录中）等 |
| `reports/phase-c42/*.md` | `08_C42/` |
| `reports/phase_c49_*.md` | `11_C49_ATTRIBUTE_DECOUPLED/` |
| `reports/r3_1/*.md`、`candidate_c_source_audit/` | `12_CANDIDATE_C/` |
| `reports/phase_c51_*.md` | `10_C51_SELECTIVE_BACKDROP/` |
| `results/training/`、`results/a100/` | `17_RAW_DATA/results/` |
| `paper/higs/*` + `paper/claims*.json` | `13_TRAINABLE_HIGS/` |

---

## 4. 一致性说明（已知坑，写报告时注意）

1. **速度语义**：`speedup` 必须写分母（kernel / forward / e2e / wall）。本项目 1.164x 是
   **wall**（EPIC-05，gsat_30k_fused_prune10_rclip05）。
2. **C42 判定**：C42 不是 universal additive optimization。全包 13 场景 × 3 种子结果
   显示其效果随 baseline 变化，**必须作为非普遍结论呈现**。
3. **历史数值 vs 修正数值**：C49 82.0%（未修正）与 75.8%（修正后）不同；报告引用一律
   用修正后数据。C19 早期均速数字以 C19-2 受控重放结果为准。
4. **R3 与 R3.1**：7,494 违规是 R3 阶段的发现；R3.1 修复后 0 违规（90 项测量）。
   写报告时不可把 R3 违规当作最终结果。
5. **R4 状态**：CUDA 内核已写（`r4/*.cu` / `ext_skip.cpp`），**未编译、未 GPU 验证**；
   只能写“实现完成”，不能写“实现了加速”。
6. **日期/时区坑**：一些日志中的日期用 UTC，部分文件名用本地时间；引用文件时以
   `git log` 的 commit hash 和文件 mtime 为准。
7. **数据有限性**：部分数值由日志推算得出（如 1.164x 出自 CI 的 wall-time 聚合），
   引用时必须标注来源文件；若报告需要新数字，必须按 03_DIRECTORY 中的指针去原始
   JSON 里核对。
8. **tile**：tile size 是运行时参数，不是代码版本；tile16 为双平台最优，但切换成本为零。

---

*本文件由 evidence-collection agent 于 2026-09-18 生成。*
