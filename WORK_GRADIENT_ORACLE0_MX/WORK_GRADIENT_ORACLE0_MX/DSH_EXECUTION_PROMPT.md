# DSH EXECUTION DIRECTIVE — WORK-GRADIENT-ORACLE0-MX

**你是 MX 实验执行代理，不是自主更换科研主线的论文构思代理。**

立即读取同包 `PROTOCOL_LOCK.json` 与 `MATH_AND_PITFALLS.md`，按以下阶段开展，所有测量必须是 fresh MX raw evidence。禁止用旧报告、synthetic Gaussian、mock timings 冒充本次实验。保持阶段状态可审计，遇到阻断立刻停并交付诊断；无需用户二次授权即可进行下述*只读审计和通过门槛后的原协议轻量测试*；不得自动开始 30K 训练、修改 CUDA kernel 或删除数据。

## S0. MX 能力和源码审计（先完成，未通过禁止运行）

1. 收集 MX node/GPU A100 40GB 实际 GPU ID、`nvidia-smi`、Torch+CUDA+driver、有效 conda/env、可用存储、其他进程快照。不要 kill 他人进程，不要清理未授权科学目录。root 与 storage pool 磁盘不足则 `BLOCKED_ENVIRONMENT`。
2. 定位真实 checkout，记录 `git rev-parse HEAD`、`git status --porcelain`、`git diff` SHA256、所有改动源文件 SHA256，native dynamic `.so` 的真实绝对路径和 SHA256、JIT 编译目录。**GitHub master 不是 MX runtime evidence**。
3. 审计 `benchmark/run_higs_train_benchmark.py`：有 `higs_dynamic_ts` + `make_forward_fn(...tile_mask=...)`，运行时验证实际 `.so` 支持外部 tile mask，且未选 tile 真跳过 intersection/binning、raster 和 native backward。低层 `std_ll` 不会自动提供等价 sparse kernel，不可偷换成 mask-loss-after-dense。
4. 定位原始 MX 数据和**真实预先存在的训练 checkpoint**：`mipnerf360/garden` 与 `mipnerf360/bicycle` 各 early 和 late 至少两种、每个至少 2 TRAIN camera。记录绝对路径、SHA256、iteration、N、来源 run/seed、完整模型状态来源、camera manifest。任何 N 误标（如 580416 作为 6M）直接阻断。**不要为了满足状态门槛训练新 checkpoint**；不足=>`BLOCKED_MISSING_REAL_STATES`，给清单和可继续的部分。
5. 训练/评估拆分严格限定：采样分数与 proxy 构建不可查看 eval/test camera、GT、未来数据。检查 `--masked-adam-union-decay-eval-proj` 等数据流，全部 OFF；如不可隔离则 `BLOCKED_LEAKAGE`。
6. 做一个 16×16 显式 tile mask 的 correctness/cost smoke。锁定 background、eps2d、SH degree、camera、loss、optimizer-independent backward config。检查 mask 生效是否只渲染 selected tiles、RGB/inactive layout、gradient zeros、border semantics；若 Python 小 hook 缺失可提交最小可审计 patch + unit tests。CUDA/ABI 不兼容 => `BLOCKED_NEEDS_CODEX_CUDA`，STOP，给精确函数和报错。

交付 S0 `SOURCE_IDENTITY.json`, `STATE_INVENTORY.json`, `CAPABILITY_AUDIT.json`, `gpu_snapshots/`, `commands/`, `logs/`, `RUN_STATUS.json`。每个所宣称的可用路径须有实际文件和哈希。

## S1A. 真正固定状态的 per-tile gradient oracle（主科学试验）

1. 用 garden/bicycle early+late 各 2 TRAIN camera 的真实冻结状态；原 renderer/backend same-state。每个 scene/state/camera 从全图预先固定 RNG **分层抽取 64 个不同候选 tile**（覆盖纹理/深度/成本的不同区间，但分层依据不允许用 eval GT），提供 `candidate_tile_ids.json` 与代码+随机种子。此 64-tile pool 只能支持 pool-level 结论；不可说它等于全帧。
2. 在同一冻结状态上，以 *additive full-image-normalized L1* 的 64 tile 子集构成梯度目标。记录 full reference `G_pool=Σ g_j` 与逐 tile `g_j` 的 means/quats/scales/opacity/SH；使用同一输出归一化和共同 `eps2d`。每个 g_j 可以离线昂贵地通过单 tile backward 收集，收集成本单独上报、并标记 `OFFLINE_ORACLE_ONLY`，不得并入 deployable performance。
3. 计算 `sum(g_j)` vs 同一 pool 单次 masked reference gradient 的误差、cosine/relL2。不能直接拿 pool gradient 与全帧梯度比较并声称可加性通过；若相加测试失败，`BLOCKED_NONADDITIVE_OR_ABI`。
4. 为每个 tile 收集真正有意义的 cost PROXY（Gaussian-tile intersections、真实有效像素数、blend interactions，能测则分列），固定 `c_j` 定义。任何 per-tile time 当可加真实成本都属不合格。用额外独立随机 mask sets 测量**集合级** CUDA/F+B 和 wall，标定 proxy 与真实成本的关系、共享固定部分与 kernel occupancy；如果排序失真，标记 `NO_GO_BAD_COST_MODEL`。
5. 比较 5 个 arm：U-uniform Bernoulli；E-历史训练误差驱动 Bernoulli（不能用未来 gradient 或同一 target 的离线 GT 当 online proxy）；T-历史时间互补作诊断（如已可用）；O-利用真实 `||Pg_j||` 做理想 oracle（严格 `OFFLINE_ORACLE_ONLY`）；P-历史可用梯度信息代理调度（没有历史梯度的 tile 使用先验/探索地板）。解析对照必须使用**同一种 Bernoulli HT loss**以及同 expected additive proxy budget。原生 error_guided `torch.multinomial(replacement=True)` 另作为生产 API 基线，不能与 E 的解析结果混成同一个 arm。
6. 对 Bernoulli HT 梯度估计器输出 32 次独立 masks 的 empirical cosine、relL2、geometry NMSE（以及各组），并对照 `Σ(1-p_j)/p_j ||Pg_j||²` 的解析 variance。若 32 次样本太少以估计置信区间，显示不确定性而非断言无偏性已由实证证明；数学无偏仅在已列前提成立。
7. 逐 cell 保存 raw tile-gradient norms/vectors 或无损分组摘要、真实错误图历史来源、tile cost features、time budgets、p_j、mask draws、image/optimizer/state identities。**所有 deployable 特征必须是过去训练步骤当时可获取、预算内可计算的信息**，不得用 g_j 真值训练 P 或决定 P 当前 mask（可用完全分离的 prior train states 拟合规则，再冻结参数，在其他训练 states/cameras 前瞻评估）。
8. 运行本包 `scripts/variance_oracle.py` 处理实测候选表，仅用于分析成本模型中的 ideal/analytic variance bound，结果文件必须带 `DERIVED_OFFLINE_PROXY_COST_NOT_RUNTIME` 标识。

S1A 预注册 threshold：理想 O 的 geometry 解析 NMSE/variance 相对 U/E 中**表现更好的**可比较方法减少 ≥30%，在两必需场景均出现、且不少于 3/4 early/late scene-stage；若 O 都无效，直接 `DROP_NO_INFORMATION_HEADROOM`，不做 P。注意不同 camera 的结果要展示，不按 tile 当独立 replicate。

## S1B. Online-available proxy 与真实成本（必须通过才可继续）

1. 在未用于设计/拟合 P 的 TRAIN states/cameras 上评估廉价 P。至少对比 U / E / P；T 可加。P 特征及超参在评估前锁定，严禁试十种再选一个报道。不允许用 test/eval camera score。
2. P 相对最强 U/E，geometry NMSE 减少 ≥20%，且其他 gradient groups 不可出现严重恶化（预注册 SH 和 opacity 相对 NMSE 恶化上限 10%）；两必需场景均有改善。若未通过，`DROP_PROXY_UNPREDICTIVE`。
3. 在真实 sparse CUDA 路径按相同图像、相同真实 frozen state、matched actual selected pixels/intersection bucket 测 F、B、F+B CUDA events 与**scheduler-included iteration wall**，30 warmup+100 timed ×3 paired alternate order，顺序轮换 U/E/P。GPU 污染观测保留并重跑，不允许删失败记录。
4. 核查策略成本：P 在线计算（含 visible Gaussian map、feature fetch、p normalization、mask generation、cache update、必要同步）wall-overhead <=5% of U；如果比 U/E 明显慢、shared cost 导致 F+B 无优势，则 `DROP_COST_OVERHEAD` / `DROP_NO_REAL_WORK_WIN`。只在实际成本可比时报告 gradient-error Pareto，不以解析 `Σp c` 冒充 GPU 工作。
5. 粗 tile 路径必须保留真实 skip，不能把 dense RGB render 后的 loss mask 冒充 sparse rendering。满足 S1A/B = `GO_TO_SEPARATE_3K_PILOT`，**不能直接宣称 CVPR 方法成立**。

## S2. 有条件 3K pilot（单独冻结后才可执行）

如果 S1A/S1B 双过门，产生 `PROPOSED_3K_PROTOCOL.json`：garden+bicycle，2 seeds，3K/arm，训练/评估严格分离，质量 vs REAL wall/TTQ，同 recipe 4 arms U/E/P/Dense；记录增密轨迹、LPIPS/SSIM/PSNR 和 wall。**不要在本 campaign 自动运行 3K**，需要先审核协议是否与当前 full-training baseline 兼容。

## 结果、裁决和投稿边界

必须按 OBSERVED / DERIVED / HYPOTHESIS / NEXT_TEST 标记每条主张。必须报告 source SHA, dynamic `.so` SHA, camera+checkpoint SHA, N, GPU contamination, raw/commands/logs/gpu snapshots，所有 CSV/JSON，报告不允许由旧表重写当成新测。

目录：`WORK_GRADIENT_ORACLE0_MX/`，必须含 `SOURCE_IDENTITY.json`, `STATE_INVENTORY.json`, `CAPABILITY_AUDIT.json`, `FROZEN_PROTOCOL.json`, `TILE_GRADIENT_REFERENCE.json`, `COST_PROXY_CALIBRATION.json`, `ANALYTIC_ORACLE.json`, `EMPIRICAL_MASK_VARIANCE.json`, `PROXY_GENERALIZATION.json`, `GPU_TIMING_REAL.json`, `VERDICT.json`, `REPORT.md`, `RUN_STATUS.json`, `MANIFEST.sha256`, 以及 `raw/ logs/ commands/ scripts/ gpu_snapshots/`（运行阶段不能空、不能仅有占位）。

终态允许：`GO_TO_SEPARATE_3K_PILOT`, `DROP_NO_INFORMATION_HEADROOM`, `DROP_PROXY_UNPREDICTIVE`, `DROP_COST_OVERHEAD`, `DROP_NO_REAL_WORK_WIN`, `NO_GO_BAD_COST_MODEL`, `BLOCKED_MISSING_REAL_STATES`, `BLOCKED_ENVIRONMENT`, `BLOCKED_SOURCE_IDENTITY`, `BLOCKED_LEAKAGE`, `BLOCKED_NO_REAL_SPARSE_RENDERER`, `BLOCKED_NEEDS_CODEX_CUDA`, `BLOCKED_NONADDITIVE_OR_ABI`, `BLOCKED_TIMEOUT_OR_RESOURCE`。

**若 S0 成功但 S1 算力/磁盘不足，允许 `BLOCKED_TIMEOUT_OR_RESOURCE`，保留已观测证据，不得填充缺失。**

最终向用户只报告：真正执行到哪一阶段、实验是否为 fresh MX、哪些 cell 完成/缺失、预注册 gate pass/fail、artifact 和哈希路径，下一步 Codex 修改是否必要。不准发表 GO 论文、1.5× 加速或“首次”的未证实结论。
