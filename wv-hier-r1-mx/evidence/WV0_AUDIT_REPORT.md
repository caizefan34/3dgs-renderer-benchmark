# WV-HIER-ORACLE0-MX 原始证据恢复与科学审计

审计日期：2026-10-08（Asia/Shanghai）。**VERDICT：BLOCKED_RAW_EVIDENCE。**

**回答：现存汇总支持 G4 在相同像素采样预算下有更好的冻结梯度估计质量；还没有足够的可独立复核证据，将历史 GO 升级为真实分层 CUDA 执行的性能依据。** 分层共享排序是合理的待验证机制，但本轮不晋级 READY_FOR_CODEX_FEASIBILITY 或 CONDITIONAL_CODEX_GO。缺失逐样本记录、绝对成本并未实测，无法给统计显著性或真实时间优势。也没有依据把证据缺口直接判成算法 DROP。

## OBSERVED — 本轮范围与完整性

- 在 MX `bms-39468022-001` 执行只读搜索、哈希检查、CPU mask 重建及统计/成本审计。
- 新目录：`/mnt/storage_pool/liaoyuanjun/wv0_raw_audit_20261008_2148`。原始 `/dev/shm/wv0_out/` 和既有源码/环境/checkpoint 均保留。
- 指定 home/pool 深度 9 搜索、原始 `/dev/shm`、两个 MX 归档、本机交付文件及归档均检查；manifest 和源码/日志给出一致的“仅保存汇总”证据链。
- 原 manifest **94/94 匹配**，checkpoint **9/9 SHA 匹配**。这验证文件身份，不补足不存在的 raw arrays。
- 本轮 GPU compute **0 秒**，CUDA 修改 **0**，编译 **0**，训练 **0 步**；没有 3K/30K 或小规模 replay。原 WV0 日志累计 25,685.7 单 GPU 进程墙钟秒（7.1349 小时），包含 CPU/I/O，不是实测 GPU 活跃秒或账单。GPU 金额未知。

## OBSERVED / RECOMPUTED — 关键裁定

| 项目 | 裁定 | 证据和边界 |
|---|---|---|
| 9,216 样本 | CORRECTION / BLOCKED | 源码循环暗示 18,432 camera-level 估计；缺原始索引，不能实数调用 |
| RAW 文件 | BLOCKED | `*_per_mask` 是统计字典；9 个原 `.pt` 同样没有逐 seed 数组 |
| 保存的数据量 | OBSERVED | 72 条 condition、360 条 group、1,080 个 metric-stat 对象；逐 camera/seed 原始行=0 |
| 损失与 HT | 源码数学审计通过，标签需纠正 | RGB SSE 非 MSE；固定 p 的 tile Bernoulli 1/p 权重；不等于 CUDA backward 验证 |
| Bias | CORRECTION | 报告误用单 mask relative error；真正 mean-gradient error 的 geometry macro 为 .03834–.23724 |
| Cosine | PASS_SOURCE_AUDIT | 相似度增加，1−cos 减少；符号一致 |
| Variance / ESS | CORRECTION | 混合 camera/mask 方差；同 seed 跨 camera/state 复用；32 seed clusters |
| G4 同像素预算 | RECOMPUTED | 汇总 18/18 更低 NMSE，16/18 至少降低 30%；配对显著性 BLOCKED |
| G4 同绝对结构预算 | BLOCKED_MEASUREMENT | 没有匹配预算观察；有限模型预测更差，但依赖代理/外推，不能当实测 |
| G4 runtime / F+B | STRUCTURAL_ORACLE_ONLY | 没有对应实现与有效实测；所有时间字段 NOT_MEASURED |
| Hierarchical 16→4 | HYPOTHESIS | 可共享父排序，但 .25 抽样约 99% 完整父块仍激活 |

## RECOMPUTED / DERIVED — 18 个状态预算比较

下面质量是保存的 8-camera geometry macro 均值；成本是 camera0 的 bbox 代理乘 p，**不是实测 CUDA intersections**。

| state | p | NMSE G16 | NMSE G4 | G4/G16 NMSE | E[G16 proxy] M | E[G4 proxy] M |
|---|---:|---:|---:|---:|---:|---:|
| bicycle__iter15000 | 0.25 | 1.9651 | 1.1544 | 0.5875 | 1.0041 | 6.0383 |
| bicycle__iter15000 | 0.40 | 0.9800 | 0.5800 | 0.5918 | 1.6066 | 9.6614 |
| bicycle__iter30000 | 0.25 | 1.8116 | 1.0422 | 0.5753 | 0.9156 | 5.3602 |
| bicycle__iter30000 | 0.40 | 0.8889 | 0.5231 | 0.5885 | 1.4650 | 8.5763 |
| bicycle__iter5000 | 0.25 | 1.8545 | 0.8543 | 0.4607 | 0.6035 | 4.8017 |
| bicycle__iter5000 | 0.40 | 0.9416 | 0.4234 | 0.4496 | 0.9655 | 7.6827 |
| garden__iter15000 | 0.25 | 2.3535 | 1.6417 | 0.6976 | 0.6763 | 3.2640 |
| garden__iter15000 | 0.40 | 1.1830 | 0.8209 | 0.6939 | 1.0821 | 5.2224 |
| garden__iter30000 | 0.25 | 1.9028 | 1.3455 | 0.7071 | 0.6705 | 3.2387 |
| garden__iter30000 | 0.40 | 0.9649 | 0.6779 | 0.7025 | 1.0728 | 5.1820 |
| garden__iter5000 | 0.25 | 2.4629 | 1.6091 | 0.6533 | 0.6377 | 2.9701 |
| garden__iter5000 | 0.40 | 1.2427 | 0.8002 | 0.6439 | 1.0204 | 4.7521 |
| room__iter15000 | 0.25 | 1.8124 | 0.9113 | 0.5028 | 0.3741 | 2.9776 |
| room__iter15000 | 0.40 | 0.9004 | 0.4536 | 0.5038 | 0.5985 | 4.7642 |
| room__iter30000 | 0.25 | 1.4503 | 0.6851 | 0.4724 | 0.3387 | 2.7695 |
| room__iter30000 | 0.40 | 0.7305 | 0.3428 | 0.4692 | 0.5419 | 4.4312 |
| room__iter5000 | 0.25 | 1.8001 | 0.7470 | 0.4149 | 0.3562 | 3.2906 |
| room__iter5000 | 0.40 | 0.8987 | 0.3741 | 0.4163 | 0.5699 | 5.2649 |

原生 G4 在相同 p 下交数代理是 G16 的 **4.6572–9.2388 倍**。固定 p 的四个粒度呈误差/成本权衡，72/72 点都在代理非支配集；放开两个已有 p 一起比较时仅 54/72 非支配，其中 G4 p=.25 在 9/9 个状态被较便宜候选支配。完整各粒度数据及支配者见 `WV0_PARETO_RECHECK.csv`。

相同绝对代理成本下，Bernoulli 理论配合有限样本 K_G 估计得到 G4/G16 NMSE **3.8391–6.3482**。该值是 **DERIVED 外推**，不能冒充实验；更不能换成训练时间。主要限制：成本 camera0 与质量 camera-mean 不完全匹配、bbox 边界公式不同于 CUDA、外推到未测 p。详见成本模型。

## OBSERVED — 精确源码、二进制与相机身份

- 观察到项目 `master`，HEAD `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`；已有三个 tracked 修改未触碰。
- 实际 gsplat 来源是 user-site 1.5.3，不是该 repo 的 Git tree；保存 1,729 个源码/头文件等文件的清单。当前树 SHA256：`767430bb11d9dcdf70f91dc001f2aaa547b5ad0434c0ec87434cef61c4786f04`。
- 现有 `gsplat_cuda.so`：`405d4bbdcf168ebedc566b0f064785722cc33c9dd1665c0393880f9ba06538a0`，23,069,312 bytes，原路径 `/dev/shm/txext/gsplat_cuda/gsplat_cuda.so`。本轮只复制/哈希，没有加载执行或重编译。
- Python 3.10.19 / torch 2.4.1+cu124 / CUDA compiler 12.4.131 / C++11 ABI=0 / driver 595.71.05。未混用其他 HiGS/cu128 环境。
- 旧 WV0 没保存历史 source/binary SHA 对；当前哈希不是对历史运行的追溯认证。
- camera IDs、原图 SHA、w2c/K、边界尺寸及 train/eval 限制见 `WV0_STATISTICS_AUDIT.md` 和 `raw/evidence/camera_image_identity.json`。
- 768 个唯一 mask 是本次 CPU 重建，72/72 selected-pixel 均值匹配；不是历史 mask 文件。完整 bits/hash 在证据目录中。

## BLOCKED — 最小补证/补测清单（仅建议，未执行）

首选提供旧逐 camera/seed 指标、原始 projected/isect 数组和历史源码/二进制身份。当前所有已定位副本及归档都没有这些记录，不能从分位数恢复数据。

若另行批准新的取证 replay，先限制为 **1 个现有高 N checkpoint × 2 cameras × G16/G4 × 两个 p × 32 seeds = 256 次 sparse gradient**，加 2 个 dense reference；无 optimizer、无训练。保存逐 camera/seed/group 指标、dense/estimate norm 和 dot、mask bits/hash、全部输入 SHA、源码/.so SHA，以及固定 camera 的在线梯度均值/方差累积量。需要独立验证 bias 时应明确保留足够的向量累积量或可重复输入，不能仅靠标量 summary。

结构成本最低补证是在相同投影上取得 G16/G8/G4/G1 的真实 per-tile counts（可分块计数，避免盲目物化 G1 的大排序数组），并把每 camera 的成本与它自己的误差配对。此项不自动假设存在相应 sparse CUDA renderer，也不产生 G4 F+B 速度结论。

这个 256 次 pilot 只能检查取证链与初步重复性，不能替代原 9-state 结论。若要完整重建原统计，需另行批准 18,432 次 sparse 与 72 次 dense，并先评估成本；本轮不启动。真实 native G4 / hierarchical CUDA 比较仍要在以后单独通过代码正确性、跳过工作证据和 paired timing 门槛。

## VERDICT — 为什么是 BLOCKED，为什么没有 DROP

保留下来的描述性信号一致，HT 公式也成立；但关键 raw 数据从保存阶段就被丢弃，跨粒度绝对成本的旧 GO 理由无效，时间证据为空。故当前实现决策为 **BLOCKED_RAW_EVIDENCE**。共享父列表有减少 native G4 重复 bin/sort 的机制依据，尚无证据证明总收益；不能据代理外推判 DROP，也不能据历史 GO 开始实现或训练。

## 交付与复现

八项指定文件全部生成：`WV0_RAW_INVENTORY.json`、`WV0_DATA_COMPLETENESS.json`、`WV0_STATISTICS_AUDIT.md`、`WV0_ABSOLUTE_COST.csv`、`WV0_PARETO_RECHECK.csv`、`WV0_HIERARCHICAL_COST_MODEL.md`、`WV0_CODEX_READINESS.json`、`WV0_AUDIT_REPORT.md`，另有 manifest、raw/、commands/、logs/、gpu_snapshots/。

`commands/05_recovery_search.sh` 是保存的只读搜索及证据复制命令；`commands/03_cpu_oracle_audit.py` 是此前 mask 审计命令（内部输出路径保留原样）；`commands/build_audit.py` 从本审计目录的证据复算八项结果。`SHA256_MANIFEST.json` 校验全部交付文件，不包含自身和外部 ZIP，外部 ZIP 单列 SHA。历史证据文件一律不覆盖。
