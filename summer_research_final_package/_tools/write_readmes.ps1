$root = "C:\Users\36570\3dgs-renderer-benchmark\summer_research_final_package"

function Write-ReadmeFile([string]$dir, [string]$title, [string[]]$lines) {
    $path = Join-Path $root $dir
    New-Item -ItemType Directory -Force -Path $path | Out-Null
    $file = Join-Path $path "README.md"
    $content = @("# $title`n") + $lines + @("`n---", "*本文件由 evidence-package 生成。*`n")
    $content | Set-Content -Path $file -Encoding UTF8
}

Write-ReadmeFile "00_README_FIRST" "00 · README_FIRST（本节说明）" @(
    "本节包含给撰写 AI 的使用说明。正式文件位于包根：",
    "- **`../00_README_FIRST.md`** - 必读。写作规则、证据纪律、禁止事项。",
    "- 本目录仅作占位；如需把说明拆分，可在此放置补充。",
    "",
    "要点速览：报告中每个数值必须能回溯到 `17_Raw_Data` 或 `02_Reports_Collection` 中的文件；不同硬件 cohort 的绝对耗时禁止混用；速度提升必须标明分母。"
)

Write-ReadmeFile "02_RESEARCH_TIMELINE" "02 · RESEARCH_TIMELINE（本节说明）" @(
    "本节包含时间线。正式文件位于包根：",
    "- **`../RESEARCH_TIMELINE.md`** - 全暑研时间线（阶段、日期、提交锚点、里程碑）。",
    "- 补充：`PHASE_LOG.md`（阶段明细）。"
)

Write-ReadmeFile "03_FINAL_RESULT_MATRIX" "03 · FINAL_RESULT_MATRIX（本节说明）" @(
    "本节包含最终结果矩阵。正式文件位于包根：",
    "- **`../FINAL_RESULT_MATRIX.csv`** - 每候选一行：判定/速度/质量/证据。",
    "- **`MATRIX_DICTIONARY.md`** - 字段字典与速度语义。"
)

Write-ReadmeFile "04_CRITICAL_FINDINGS" "04 · CRITICAL_FINDINGS（本节说明）" @(
    "本节收录必须写进 final report 的关键发现，均带证据路径。",
    "",
    "核心内容（详见 critical_findings.md）：",
    "1. 唯一 CONFIRMED 训练加速：`gsat_30k_fused_prune10_rclip05` - 1.164x wall（CI LB 1.034），11 scenes x 3 seeds，132-job。",
    "2. C42 不是 universal additive optimization（见 08_C42/）。",
    "3. C1（深度压缩）在 RTX 5070 上 <1%，在 A100 上 ~8-9% 估计但未验证。",
    "4. C49 修正后上限约 1.13% - C49/C50 被 C51 方法论吸收。",
    "5. 瓦片机制：占用率坍塌 + 场景/硬件依赖，见 07_Tile_Geometry/。",
    "6. R3 -> R3.1 证书修复：7,494 违规 -> 0 违规，见 13_Candidate_C/。"
)

Write-ReadmeFile "05_PROTOCOL_AND_DATASETS" "05 · PROTOCOL_AND_DATASETS（本节说明）" @(
    "本节包含实验协议、数据集与硬件队列定义。详细内容见：",
    "- `docs/protocol.md` - 通用协议",
    "- `docs/hardware.md` - 硬件队列及不同 GPU 不做绝对对比规则",
    "- `docs/datasets.md` - 数据集清单与校验哈希",
    "- `docs/evaluation_methodology.md` - 测量与统计方法",
    "- `docs/compression-protocol.md` - 压缩轨道协议",
    "- `docs/official_dataset_training.md` - 官方数据训练",
    "- `configs/*` - 各阶段配置",
    "",
    "参考：S1 摘要（_drafts/s1_protocol.md）提炼了 Reference V1 精确定义。"
)

Write-ReadmeFile "06_BASELINE_REFERENCE_V1" "06 · BASELINE REFERENCE V1（本节说明）" @(
    "本节包含 Reference V1 基准锁定的全部证据：",
    "- `baseline_12unit_tests.md` - 12 个单元测试",
    "- `baseline_30k_run.md` - 30K 训练结果 (room, seed 42/10, 复现)",
    "- `baseline_checkpoint_manifest.md` - 检查点与哈希",
    "- 相关源文件在 src/gaussian_renderer/、src/utils/（见 14 节索引）"
)

Write-ReadmeFile "07_TILE_GEOMETRY_HARDWARE" "07 · TILE GEOMETRY / HARDWARE（本节说明）" @(
    "内容：",
    "- `tile_size_sweep.md` - t8/t16/t20/t24/t32 的 fwd/fwd+bwd 曲线（A100 与 RTX 5070 Laptop）",
    "- `occupancy_block_geometry.md` - C19-2 固定几何重放：占用率坍塌机制",
    "- `cache_line_quarter.md` - C19-0/1 缓存行四分与 ~22.9% 提升",
    "- 结论：tile_size 是运行时参数，不重新编译；同一二进制 fwd 输出 bit-identical；RTX 5070 上 tile16 是 fwd+bwd 最优（room），A100 上也是，但 tile20 训练异常（561.8 min）已单独追踪。"
)

Write-ReadmeFile "08_C42" "08 · C42（候选 C42 全证据）" @(
    "- `C42_formulation.md` - C42 精确定义（DSIM/downscale 机制公式）",
    "- `C42_13scene_raw_results.md` - 13-scene 原始表格",
    "- `C42_baseline_comparison.md` - 与 Reference V1 逐场景对比",
    "- `C42_vs_Speedy_FastGS_FasterGS.md` - 跨 baseline 对照（非叠加性证据）",
    "- `C42_adaptive_scale.md` - 自适应尺度可行性研究与阈值扫描",
    "- `C42_final_decision.md` - 最终判定（DROP as universal; KEEP as tool for baseline-specific tuning）",
    "",
    "数据：`17_Raw_Data/results/c42_adaptive/`、`17_Raw_Data/results/a100/phase-c42/`"
)

Write-ReadmeFile "09_FORWARD_SORT" "09 · FORWARD SORT / C1（本节说明）" @(
    "- `C1_depth_compression.md` - 32->16 位深度键；12->8 遍（未验证）；RTX 5070 <1% 改善",
    "- `C17_1_tile_queues.md` - 每瓦片队列；完整度状态：CAUTION（原型差异，未通过判定）",
    "- `C18_1_incremental_sort.md` - 增量排序：GO -> C18-2 原型",
    "- `C17_2.md` - 交叉瓦片差分可行性：CAUTION",
    "- 提交锚点：C17 原型 b562562、机制修正 fc756f4、R3 矢量池化 e494458 等"
)

Write-ReadmeFile "10_BACKDROP_DEPTH_C25" "10 · BACKDROP / DEPTH（C25 数据）" @(
    "C25/C26 数据。摘要位于 _drafts/s5_backward_c25.md。",
    "1% 高稀疏参数激活占比 ~2.59e-5，占 ~29% 总反向成本；固定 floor ~1.7ms。"
)

Write-ReadmeFile "11_C51_SELECTIVE_BACKWARD" "11 · C51 选择性反向（本节说明）" @(
    "- `C51_variants.md` - V1/V2、B1/B2/B3、门 A/B/C/D 定义",
    "- `C51_k50_b1_results.md` - K50-B1：kernel +10.2%",
    "- `C51_30k_results.md` - 30K 训练结果：Room +8.2%、Bicycle +24.4%、Garden +28.9%（wall-clock）",
    "- `C51_status.md` - 最终状态：KEEP，作为 _SYSTEMS_COMPONENT",
    "- 警告：B2 失败原因 = 部分更新不一致（-0.49 dB）；B1/B3 一致性成立。"
)

Write-ReadmeFile "12_ATTRIBUTE_DECOMPOSED_C49" "12 · C49 属性解耦（本节说明）" @(
    "摘要位于 _drafts/s7_c49.md。完整证据在仓库 reports/phase_c49_*。",
    "U_loss = -g^T dtheta；top-10% = 82.0% (uncorrected) / 75.8% (corrected)；SH 20% -> 95%；max E2E ~1.13% => DROP。"
)

Write-ReadmeFile "13_CANDIDATE_C" "13 · CANDIDATE C（本节说明）" @(
    "- `R3_audit_with_perf_impact.md` - 7,494 违规计数",
    "- `R3.1_root_cause_conic_as_color.md` - 根因链",
    "- `R3.1_certificate_spec.md` - 修复后证书规范",
    "- `R3.1_validated_results.md` - 90 项测量，0 违规，最差比率 0.9998",
    "- `R4_cuda_implementation.md` - CUDA 内核与绑定；编译未验证",
    "- `decision.md` - 最终 C_KEEP"
)

Write-ReadmeFile "14_HIGS_TRAINABLE" "14 · 可训练 HiGS（本节说明）" @(
    "- `architecture.md` - 质心 / 形心描述符、委托渲染、反向委托",
    "- `correctness.md` - 单元测试与数值（67/67、112/112、117/117）",
    "- `results_13scene.md` - 13-scene 结果表",
    "- `T_HIGS_REGRESSION.md` - 回归记录",
    "- 附：paper/higs/tables/* 为报告最终版数据，请优先引用"
)

Write-ReadmeFile "15_NEGATIVE_RESULTS" "15 · NEGATIVE_RESULTS（本节说明）" @(
    "本目录的 NEGATIVE_RESULTS.md 是撰写 final report 时的必读清单：",
    "C1（RTX 5070）<1%、C17-0 分段排序更慢、C17-3 元数据缓存被证伪、",
    "C43 自适应瓦片尺寸被证伪、C49 独立候选被降级、C50 持久性假设被修正、",
    "C51-B2 门失败、Turbo-GS/SkipGS 上不可移植、border 约束失败等。",
    "每个条目都包含为什么失败的证据文件。"
)

Write-ReadmeFile "16_REPRODUCIBILITY" "16 · REPRODUCIBILITY（本节说明）" @(
    "本目录的 REPRODUCIBILITY.md 依据包根 REPRODUCIBILITY.md。",
    "内容包括 Python/CUDA 版本矩阵、seed 配置、数据集哈希、构建命令。"
)

Write-ReadmeFile "17_RAW_DATA" "17 · RAW_DATA（本节说明）" @(
    "本目录是从仓库 results/ 中抽取的结构化数据（JSON/CSV）。",
    "索引见 index.json 与 RESULTS_INDEX.md。"
)

Write-Host "All README files written."
