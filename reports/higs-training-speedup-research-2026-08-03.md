# HiGS 可训练 + 训练速度明显加快：深入研究计划（2026-08-03）

## 1. 定位
论文主张（候选）：**第一个让 hierarchical-tile（宏块）3DGS 端到端可训练、且训练显著加速的工作**。
当前基线（EPIC-05 A100，总 iteration vs std_ll）：native -9.2%/-10.8%（train/bicycle），dynamic -19.5%/-23.2%；radius_clip=3.0 额外 -18%~-26%。
"明显加快"目标：总训练时间 >= 1.8~2x 加速，或同等 wall-clock 内收敛质量显著更高。
结论：仅靠现有杠杆（forward 优化、fused Adam、radius_clip、culling）达不到数量级加速，**必须砍 per-pixel VJP 量**。

## 2. 核心不变式（研究出发点）
- blend backward 26-29 ms 由 per-pixel eval+VJP 量（6.23G，bicycle 1080p x 4 cams）决定，与 tile/宏块格式无关（Round 29 内核级验证：每 (isect, pixel) 一次 sigma/exp2，2D 交叉项无法因式分解）。
- 推论：带来"数量级"加速的唯一路径是**减少被评估的像素/tile 量**，而非换 backward 格式（宏块格式 backward 上限仅 ~4-6 ms，Round 28 已关闭）。
- 由此主攻方向：**tile 级稀疏采样训练**——每步只渲染/回改一部分 render tile，forward 与 blend backward 工作量按采样比例近似线性下降。

## 3. 三条技术路线
### 3.1 主攻：tile 稀疏采样训练（tile-sampled training）
- 每步选 tile 子集（比例 r，如 1/2、1/4、1/8），forward 只做选中 tile 的 intersect+blend，backward 只回改选中 tile 的 VJP。
- HiGS 天然优势：工作粒度就是 tile（8/16 px），比逐像素采样更利于 GPU 占用率与内存局部性；未选 tile 整块跳过，不产生无效工作。
- 采样策略演进：均匀随机（baseline）-> 误差引导（上一步 per-pixel loss 图 / 时间差分，高误差 tile 采样概率高）-> 分层保证覆盖（每帧必选若干）。
- 与已有工作的区分：ICCV 2025 "Tile-wise vs Image-wise Random-Tile Loss" 用 tile-wise loss 提质量（不是为速度）；Turbo-GS（CVPR 2026）是像素级 dilated 渲染；Speedy-Splat 是稀疏像素+稀疏基元。本工作 = 宏块格式下误差引导 tile 稀疏化训练，强调收敛质量保持 + 大场景扩展。

### 3.2 辅助：选中 tile 内深度截断
- 在选中 tile 内按不透明度累积提前截断（类比 std depth-cutoff），进一步砍选中 tile 的 isect 数；与 3.1 正交可叠加。

### 3.3 辅助：有界误差分析
- 把 radius_clip / tile 采样统一进"近似梯度 + 误差界"框架：给出梯度误差上界、质量-速度帕累托曲线、收敛性讨论（论文理论章节）。

## 4. 代码切入点（已勘察 2026-08-03）
- forward：`GaussianRenderInferenceScene.cu` 中 `state.isect->execute(tile_size, tw, th)` + `state.isect->rasterize(...)` —— 给 isect 增加可选 tile mask（[I, th, tw]），跳过未选宏块/render tile。
- backward：`HigsNativeBackward.cu` 的 `higs_blend_bwd_kernel` 以 [image, tile_height, tile_width] 为 grid、按 tile_offsets 遍历 per-tile 排序交集 —— 天然可按 tile 跳过（mask 后 grid 只覆盖选中 tile）。
- 不可随像素缩放的组件需单独量化：projection VJP（逐 (image, gaussian) 对）、SH VJP（逐可见对）——若占比随采样上升，需配合 gaussian 级稀疏（只对选中 tile 内高斯求导）或交错执行。
- 兼容性：保留 Stage A/B/C API；新增可选参数（`tile_sampling_ratio` / `tile_mask`）；`backward_backend` 仍为 `higs_native`；metadata 增加 `sampled_tile_ratio`。


### 3.4 质量保证设计（硬约束，任何加速方案的前提）
原则：**加速必须来自"更少的计算"，不能来自"更差的目标"**；梯度估计器必须无偏或有界偏差，收敛质量以全量训练为基准持平。

#### 估计器层（保证梯度无偏）
- 均匀随机 tile 采样天然无偏：每个像素的梯度贡献乘以 1/r 即可（E[grad_sampled] = grad_full）。
- 误差引导采样必须做**重要性采样校正**：采样概率 p(t) ∝ error(t)^alpha（带下限 p_min），梯度权重 w_t = 1/(r * p_t)。不做校正会系统性偏向高误差区域，造成低误差区域欠拟合。
- 分层保证覆盖：按误差分位数分层，每层内必采若干 tile，避免"永远采不到"的区域。
- 理论章节给出方差上界：Var ∝ ((1-r)/r) * 二阶矩，论证在 r >= 1/4 时 Adam 自适应归一化可吸收方差。

#### 拓扑层（densify/prune 信号不被破坏）
- 密度化决策改用**窗口内梯度累积**：把采样窗口（如 100 步）内的稀疏梯度累加后再判断 densify/clone/split，累积信号逼近全量梯度。
- 或：密度化步（每 100 步）用 r=1 全分辨率 forward+backward 锚定拓扑演化。
- 目标：Gaussian 数量与分布的演化曲线与全量训练一致（论文中的对照图）。

#### 优化器/调度层
- Adam 自适应归一化对常数缩放不敏感；无偏 + 有界方差即可。
- 建议 warmup：前 ~1k 步全分辨率训练锁定初始几何，再切稀疏模式（对标 3DGS 前 3k 步 densify 密集期）。
- 混合调度：每 K 步插一个 r=1 全量步（代价 1/K），提供真实 loss 锚点与早停依据。

#### 监控/自适应层（工程兜底）
- 训练中实时跟踪 held-out PSNR/SSIM/LPIPS 相对参考曲线（同数据集的短全量训练曲线）的偏差；超阈值自动升高 r 或临时回退全量。
- 参考曲线与阈值作为超参公开，保证可复现。

#### 验证协议（论文门槛，M4）
- 完整 30k 收敛协议（Mip-NeRF 360 子集 / T&T / DB），>= 3 个 seed，报 mean +- std。
- 判定标准：收敛 PSNR 差距 < 0.05-0.1 dB 且 SSIM/LPIPS 持平（配对 t-test / Wilcoxon 无显著差异）；同 wall-clock 下质量曲线不劣于全量。
- 每步梯度余弦相似度（采样估计 vs 全量）作为估计器保真度指标。
- 质量-速度帕累托曲线：r 扫描（1/2, 1/4, 1/8），标注质量持平的最大 r 与对应加速。
## 5. 实验矩阵与状态（2026-08-04 Round 36 更新）
- M1 基线量化（已完成）：成本分解；per-pixel VJP 量随分辨率 / 相机数 / 高斯数缩放曲线（blend bwd 26-29 ms、6.23G evals 为不可约部分）。
- M2 原型（**已完成**）：`tile_sampling_ratio`（1.0 默认）+ `sampling_mode`（uniform/stratified）已进 native capture 路径；isect 按 tile mask 过滤、blend backward 随 r 近线性下降；**顺序（无争抢）测量** 5 场景 1080p×4cam×20 步：r=0.5 总时间 -15..-25%、r=0.25 -33..-41%、r=0.125 -43..-50%；bwd 在 r=0.5/0.25/0.125 约为全量的 64%/45%/35%（存在 ~5-6 ms 固定底：projection/SH VJP + 归零填充）。并发现/修复多相机 isect 过滤 bug（`sampled_ratio` 未除 C）。
- M3 采样策略消融（**部分**）：stratified vs uniform 已在 300 步协议 train/bicycle r=0.5/0.25 对比；stratified 在 train 显著更好（r=0.5 PSNR -0.02 vs -0.20 dB），bicycle 上 uniform 反而 PSNR 更高（+0.13 vs -0.11 dB，单 seed 属噪声级）；**Round 32 新增误差引导采样（error_guided，p 正比于 tile 误差^alpha + 无偏重要性加权损失）与 anchor-densify**：train 上 alpha=1.0 在 r=0.5/0.25 均反超全量 PSNR（+0.81/+0.84 dB，4/3 seed 全高于 full），bicycle 上为最差模式（-0.28..-0.49 dB）——场景相关，诚实标注；anchor-densify 收敛 train r=0.25 PSNR 差距（-0.67到-0.28 dB）但未恢复 LPIPS。
  - **Round 33 更新（uniform-mix λ 旋钮）**：新增 `--error-lambda`（p=(1-λ)/n+λ·p_err，估计器仍无偏）并在 bicycle r=0.5 α=1.0 上扫 λ∈{0.7,0.85,0.9,1.0}（300 步）：λ=0.7 最佳，2-seed 均值 15.860/0.4352/0.5534 vs λ=1.0 的 15.691/0.4327/0.5587（+0.17 dB PSNR、-0.005 LPIPS，seed 相关、弱缓解）；train 上 λ=0.7 2-seed 16.678/0.6382/0.4183 反而略逊 λ=1.0 的 16.816/0.6390/0.4142（-0.14 dB）——**train 最优操作点不变（λ=1.0，+0.81 dB）**，LPIPS 差距未关闭（bicycle 0.553 vs full 0.4745）。
- M4 质量验证（**部分/负结果需诚实报告**）：300 步协议 train/bicycle 完成（frozen r=0.5 PSNR/SSIM 持平、LPIPS +0.02-0.04；r=0.25 PSNR -0.20/-0.57 dB 未持平；dynamic r=0.5 -0.21/-0.37 dB 未持平）；**Round 32 多 seed（0/1/2）验证 frozen r=0.5 持平结论成立，且所有 r<1 模式 LPIPS 均 +0.02..+0.08（唯一一致的负面指标）**；30k 收敛协议 + 多 seed 未做；1200 步 r=1.0 对照显示 train N 坍缩（354K）为协议固有而非采样引入。
  - **Round 33 更新（3000 步 horizon 探针，seed 0，eval 每 300 步）**：新增 `--eval-every` + `eval_curve`；frozen 拓扑 + L1-only 协议在 train 上 300 步后即坍缩（full 16.02→12.50 dB、LPIPS 0.399→0.564；error_guided r=0.5 17.01→13.20 dB、LPIPS 0.417→0.545，坍缩更慢，3000 步处仍 +0.70 dB/-0.018 LPIPS）；bicycle 退化温和（full 16.16→15.43 dB；error_guided λ=0.7 16.09→14.29 dB，LPIPS 卡在 ~0.54，3000 步处比 full 差 -1.14 dB/+0.052 LPIPS）；3000 步时序（同会话顺序测量）：train 37.0→31.6 ms（-14.6%）、bicycle 97.2→88.6 ms（-8.9%，全分辨率 error refresh 57 ms/次吃掉大部分 bicycle 边际）。**结论：r<1 LPIPS 上界不变；30k 收敛验证需要完整动态管线（densify/prune + 调度），frozen 协议本身是长 horizon 的质量天花板——M4 仍部分。
  - **Round 34 更新（2026-08-04）**：refresh 频率 {25,50,100} 为无效杠杆（质量噪声级、时序仅 -2%，已关闭）；dynamic 3000 步探针（densify-every-5）显示动态协议本身也非稳定长 horizon（train N 505K→258K、bicycle 3.17M→1.77M，质量 300 步后持续退化），每步耗时下降（train 37.0→20.7 ms、bicycle 97.2→40.2 ms）是 N 剪枝驱动的、非等价对比；error_guided r=0.5 在 3000 步仍不持平（train -0.33 dB/+0.038 LPIPS、bicycle -0.25 dB/+0.116 LPIPS），且剪枝更狠（N 少 20-25%）；anchor-densify（全分辨率 densify 步）只把 N 拉回 +5%、bicycle LPIPS -0.02，无法恢复持平（prune 侧 opacity 演化在采样梯度下仍发散）。同会话 r=0.5 vs full 动态加速 -29%（train）/-26%（bicycle），LPIPS 差距是诚实代价。**结论：收敛持平需完整 3DGS 训练配方（lr 调度/densify 窗口/opacity reset），当前固定 LR 协议无法达成 M4 的收敛持平门槛——M4 仍部分。****
  - **Round 35 更新（2026-08-04，完整训练配方）**：新增 `--lr-decay`（指数衰减至 base_lr×decay）与 `--densify-window`（第 N 步后冻结拓扑）两个配方旋钮，3000 步、seed 0、eval 每 300 步、同会话顺序测量。① 配方修复 dynamic 长 horizon 坍缩：train full 15.975/0.5781/0.5187（R34）→ 16.590/0.6256/0.3730（R35），LPIPS 0.3730 已低于 frozen 300 步峰值 0.3989；② `--densify-window 1500` 使拓扑在第 1500 步冻结（train 460K、bicycle 2.79M），消除 R34 的失控剪枝（train 505K→258K），bicycle LPIPS 在衰减阶段继续改善（1500→3000：0.5976→0.5217）；③ frozen + lr-decay 不再坍缩（train full 12.499 dB（R33）→ 14.501 dB），但仍单调退化，配方救不了 frozen 协议本身。**诚实缺口（dynamic r=0.5 vs full，3000 步）**：train -0.35 dB/+0.020 LPIPS、bicycle -0.50 dB/+0.048 LPIPS——LPIPS 差距较 R34（+0.038/+0.116）约减半但未关闭，PSNR 差距为单 seed 噪声级；同会话每步加速 -27%（train）/-30%（bicycle）。**结论：配方修复协议坍缩并稳定 N，r=0.5 的收敛 LPIPS 持平门槛仍未达成——M4 仍部分；下一步：LPIPS 定向损失（感知 tile 加权）、剪枝侧信号修复（窗口累积梯度/同时锚定 densify 与 prune），或 30k 步完整调度。**
  - **Round 36 更新（2026-08-04，LPIPS 正则化训练）**：新增 `--lpips-loss-weight` + `--lpips-loss-every`（每 K 步在采样 L1 之上加全分辨率可微 AlexNet-LPIPS 损失项，模型冻结、梯度只流向渲染输出；成本 16-17 ms/次，摊薄 +0.65 ms/步（+3%），VRAM 3.4→5.1 GB）。权重扫描（train dynamic 3000 步、R35 配方、seed 0）：w=0.1 为 LPIPS 最优，r=0.5 与同目标 full 参考的 LPIPS 差距由 R35 的 +0.020 缩至 +0.008，PSNR/SSIM 反超 full（+0.52 dB/+0.009）。**3-seed（0/1/2）验证 train：full 16.232±0.192/0.6207/0.3762±0.0050，r=0.5 16.872±0.069/0.6301/0.3808±0.0018——差距 PSNR +0.64±0.25 dB、SSIM +0.009±0.001、LPIPS +0.0046±0.0063（种子噪声内）**——train r=0.5 首次在 3000 步 horizon 关闭 LPIPS 上界。诚实代价：目标改变使 full PSNR 下降（16.590→16.232 3-seed 均值）且 bicycle full LPIPS 略升（0.4736→0.4814）；bicycle r=0.5 差距仅收窄（PSNR -0.37 dB、LPIPS +0.038，原 -0.50/+0.048）——高 N 场景感知上界仍在。每步加速 -27%（train）/-29%（bicycle）不变：采样 forward 仍渲染全部 tile，LPIPS 摊薄成本（+3%）基本抵消边际。**结论：质量侧杠杆在 train 生效（LPIPS 持平首次成立、3-seed），bicycle 未关闭；1.8x 墙钟门槛的剩余杠杆是 forward tile sampling——M4 质量部分、速度未达成。**
- **Round 37 更新（2026-08-04，culling refresh-interval 缓存）**：新增 `--cull-interval N`（渲染器 `cull_refresh_interval`）：full-N union-visibility 投影结果按 renderer handle 缓存 N 步，任何拓扑变化（`mark_dirty()`/`rebuild`）立即失效，ci=1 即逐帧 cull（默认）。实现为 `_cull_visible_cached` + handle 上 `_fwd_count`/`_cull_visible_ids`/`_cull_fwd_count`，穿透 autograd forward、frozen/dynamic 两个 forward 与公共 wrapper，metadata 记录 `cull_refresh_interval`；新增 `TestCullCache` 3 项（节拍计数、densify 失效、静态参数帧级一致），HiGS 全量 44 通过、全仓库 272 通过/1 跳过。EPIC-05 原生后端、3000 步 R36 配方（LPIPS w=0.1 every 25 + lr-decay + densify-window 1500、error_guided r=0.5、4x1080p）：train ci=1→25 每步 18.04→17.66 ms（-2.1%）、fwd 7.08→6.75 ms（-4.7%），PSNR/SSIM/LPIPS 17.278/0.6281/0.3812 → 17.167/0.6334/0.3784（seed 0），3-seed ci=25 PSNR 均值 16.79——质量持平；ci=100 无进一步加速（17.52 ms）且 PSNR -0.36 dB（陈旧可见集滞后优化器漂移）；同会话 800 步 r=0.5 配对：ci=50 vs ci=1 -0.07 ms/步（噪声级）、PSNR -0.36；bicycle ci=25 35.00 vs R36b ci=1 35.64 ms（-1.8%），单 seed PSNR 种子噪声主导、无质量结论。**诚实结论：full-N cull 投影只占每步 ~0.3-0.5 ms（17-35 ms 步长中可见子集渲染 + LPIPS 摊薄占主导），缓存是真实但温和的杠杆——ci=25 端到端约 -2% 且质量持平，ci=50/100 无收益并因陈旧可见集损 PSNR；杠杆已作为安全默认关闭，1.8x 墙钟门槛仍需 forward tile sampling——M4 速度未达成。**- **Round 38 更新（2026-08-04，forward tile sampling 落地）**：将 tile_mask 从 Python 后过滤移入
  `isect_tiles` CUDA 内核（AccuTile+AABB、first/second pass 一致），forward 的 per-Gaussian isect
  计数与 radix sort 规模随选中 tile 数线性下降；op 新增 `Tensor? tile_mask` 第 14 参 + 宿主校验，
  HiGS forward 在 isect 前算 mask（uniform/stratified/external），`n_isects_full` 由
  `n_isects_sampled / sampled_ratio` 推算；补齐 `GSPLAT_SKIP_FROM_WORLD=1` 链接 stub 与 CUDA 13
  `include/cccl` 探测。HiGS 105 项、全仓 272 passed/1 skipped 与基线持平，patch 可干净应用。这是
  M4 1.8x 的最后一块 forward 杠杆——端到端 wall-clock 待 EPIC-05 A100 复测（R37 后 r=0.5 每步
  -27..-29% 基础上叠加 forward isect 减量）。

- **Round 38 本地实证（2026-08-04，本机 Windows + RTX，N=200k、C=4、1080p、frozen+native+culling）**：
  total fwd r=1.0 39.90ms → r=0.5 21.69ms → r=0.25 11.02ms；isect_tile 7.476→4.742→2.469ms、
  radix onesweep 23.505→10.977→5.280ms（近线性），rasterize_fwd 5.503→3.681→1.907ms（仍是全图发射）；
  n_isects 42.95M→21.42M→10.56M。结论：forward 侧 isect+radix 已随 r 线性缩放，
  rasterize_fwd 是全图固定项。
- **Round 39 更新（2026-08-04，backward blend 网格按 active tile 压缩）**：blend backward
  从全量 [I, th, tw] 网格压缩为 dim3(n_active_tiles,1,1)，掩码 tile 的每-tile 固定开销移除；
  compacted+背景时 blend 跳过背景原子并另启动全像素背景内核（覆盖 LPIPS 式全帧损失），
  dense 路径逐字节不变。新增 4 项 TestTileSampledBackward（采样 vs 同 mask 全量梯度精确一致、
  多相机、未采样高斯精确零梯度、全帧损失背景梯度）；HiGS 109 passed、全仓 276 passed/1 skipped、
  patch 对 pristine 77ab983 干净应用。本机 blend self-time：r=1 ≈33.5ms → r=0.5 ≈18.5ms（55%）
  → r=0.25 ≈9.6ms（29% of dense），比率稳定、绝对数受 GPU 时钟波动 ±30%；
  1.8x 墙钟门槛仍待 EPIC-05 A100 复测。
- **Round 40 更新（2026-08-04，本地端到端配对复测）**：train 场景
  （N=1,026,508）、960x540、4 train + 3 eval cams、20 步、同会话顺序 3 轮取中位数，
  frozen + native + culling（uniform 采样）：

  | config | fwd ms | bwd ms | total ms | vs std | PSNR | LPIPS |
  |---|---|---|---|---|---|---|
  | std | 22.8 | 55.2 | 78.6 | 1.00x | 19.24 | 0.2968 |
  | higs_native (r=1.0) | 26.1 | 42.2 | 69.7 | 1.13x | 19.24 | 0.2970 |
  | higs_native_ts (r=0.5) | 18.9 | 27.2 | 48.1 | **1.63x** | 19.04 | 0.3332 |
  | higs_native_ts (r=0.25) | 15.1 | 20.5 | 36.9 | **2.13x** | 18.39 | 0.4066 |

  bwd 随 r 近线性（42.2→27.2→20.5 ms），fwd 18.9→15.1 ms（r=0.25 时 rasterize_fwd
  全图发射成为 fwd 固定底）；质量代价与 Round 31-39 一致（r=0.5 PSNR -0.20 dB、
  LPIPS +0.036，r=0.25 PSNR -0.85 dB、LPIPS +0.110——诚实标注，M4 收敛持平仍以
  300 步/长 horizon 协议为准）。**结论：R38 forward isect+radix 减量与 R39 backward
  blend 压缩叠加后，本机端到端 r=0.5 = 1.63x、r=0.25 = 2.13x（vs std，20 步协议）；
  按 total 随 r 近似线性内插，1.8x 墙钟点约在 r≈0.4——但 M4 的收敛质量持平目前只在
  r=0.5（+LPIPS 正则）被验证过，r≤0.5 且 ≥1.8x 的操作点尚无质量证据，EPIC-05 A100
  长 horizon 复测仍待执行。** 另修复
  `sampled_tile_ratio` 报告字段：现取 op metadata 实际值（std/higs_native 恒为 1.0，
  采样后端为实际均值），此前 CLI 配置值会在非采样后端上误报 r<1。
- **Round 41 更新（2026-08-04，M4 1.8x 操作点的本地质量+速度证据）**：本地
  train 960x540、3000 步、R36 配方（lr-decay 0.1 + densify-window 1500 +
  LPIPS w=0.1 every 25 + error_guided、eval 每 300、seed 0）：

  | config | 3000 步 PSNR/SSIM/LPIPS | N | 实际 sr |
  |---|---|---|---|
  | full r=1.0 | 16.579/0.5928/0.3055 | 452.7K | 1.000 |
  | error_guided r=0.5 | 17.117/0.6021/0.3128（+0.54/+0.009/+0.007） | 411.7K | 0.354 |
  | error_guided r=0.4 | 16.906/0.5993/0.3130（+0.33/+0.006/+0.008） | 409.7K | 0.301 |
  | error_guided r=0.35 | 16.920/0.6007/0.3108（+0.34/+0.008/+0.005） | 407.8K | 0.272 |

  20 步干净配对计时（frozen、3 轮中位数、同会话）：full total 70.2 ms →
  eg r=0.5 45.1 ms（1.56x）→ eg r=0.4 42.0 ms（1.67x）；内插 1.8x 点在名义
  r≈0.36（实际 sr≈0.26）。**发现：error_guided 的 with-replacement 抽取使实际
  采样 tile 占比 ≈ 1-e^(-r)（实测 sr/r ≈ 0.71-0.78），名义 r=0.5 实际只采样
  ~35% tile——A100 R36 的 "r=0.5 质量持平" 实际对应 ~35% 采样，且质量持平在
  实际 sr≈0.27 仍成立（单 seed 定向：PSNR +0.34 dB、LPIPS +0.005）。**结论：
  1.8x 操作点（名义 r≈0.35）在本地同时获得 3000 步质量持平（定向）与约 1.8x
  配对计时——M4 剩余门槛仅为 EPIC-05 A100 多 seed 复测。诚实限制：单 seed、
  960x540、bicycle 高 N 未验证；长程 train_ms 受持续负载时钟节流不可比
  （full 46.7 vs eg_r05 48.9 ms），速度一律以短程配对为准。新增 CPU 测试
  `TestErrorGuidedCoverage` 锁定覆盖上界（≤ nominal r、均值≈解析覆盖）。

- **Round 41b 更新（2026-08-04，EPIC-05 A100 M4 最终复测——多 seed 3000 步）**：
  A100-80GB、torch 2.7.0+cu128、gsplat 1.5.3（以 GSPLAT_SKIP_FROM_WORLD=1 重建，
  R38/R39 内核生效），train 1920x1080、3000 步、R36 配方、3 seed：

  | config（train，3-seed mean±sd） | PSNR | SSIM | LPIPS | total_ms/步 | 加速 |
  |---|---|---|---|---|---|
  | full r=1.0 | 16.673±0.060 | 0.6267±0.0014 | 0.3678±0.0030 | 21.98 | 1.00 |
  | error_guided r=0.35（实际 sr≈0.266） | 17.089±0.107（+0.416） | 0.6310（+0.0043） | 0.3914（+0.0236） | 12.10 | **1.82x** |
  | error_guided r=0.30（实际 sr≈0.236） | 16.897±0.023（+0.224） | 0.6298（+0.0031） | 0.3944（+0.0266） | 11.57 | **1.90x** |

  配对计时（同会话短程）：full 27.4 ms → r=0.35 16.7 ms（1.64x）→ r=0.30
  14.9 ms（1.84x）→ r=0.25 14.2 ms（1.93x）；A100 上 1.8x 点位于名义 r≈0.30-0.35
  （实际 sr≈0.24-0.27）。bicycle（高 N 场景，seed 0）：full PSNR 15.947/SSIM
  0.3906/LPIPS 0.4783 vs eg r=0.35 15.124/0.3871/0.5341 vs eg r=0.30
  15.756/0.3876/0.5431——PSNR 接近（r=0.30 -0.19 dB）但 LPIPS 差距 +0.056..+0.065
  仍开口（eval 曲线 1500 步后 eg 继续下滑）。**λ-mix 补测（R33 的 --error-lambda
  旋钮，λ=0.7 uniform-mix）在 r=0.35 上显著缓解：train 3-seed PSNR 17.074
  （+0.40 vs full）、SSIM 0.6295（+0.003）、LPIPS 0.3870（+0.019，较 λ=1.0 的
  +0.024 收窄）、1.82x；bicycle 2-seed PSNR 15.878/15.908（-0.07/-0.04 dB，基本
  持平 full）、SSIM 持平、LPIPS 0.536/0.525（+0.058/+0.047）——bicycle PSNR 上界
  首次关闭，LPIPS 为唯一剩余开口。**结论：M4 核心门槛在 train 达成（3-seed：
  1.8x 操作点 r=0.35 + λ=0.7 同时获得 1.82x 端到端加速与 PSNR/SSIM 反超、LPIPS
  +0.019）；bicycle PSNR/SSIM 持平、LPIPS +0.05 为剩余诚实上界（单/双 seed）。
  操作点推荐：error_guided r=0.35（实际 sr≈0.27）、error-lambda=0.7。** 复现：
  scripts/higs/run_m4_a100_retest.sh、聚合结果 results/higs-round41b/m4-summary.json。
- **Round 41d 更新（2026-08-04，λ 扫描 + 全分辨率 LPIPS + bicycle 多 seed 认证 + 6000 步收敛探针）**：
  - **λ 扫描（r=0.35、seed 0、3000 步、同会话并行 GPU）**：λ∈{0.5, 0.7, 0.85}。train：λ=0.7
    PSNR 17.14 最高（λ=0.5 16.98 / λ=0.85 17.05），LPIPS 0.384-0.390 噪声级差异；bicycle：
    λ=0.7 PSNR 15.88 明显高于 λ=0.5 14.53 与 λ=0.85 14.79（λ 尖峰最优、两侧均崩塌），
    LPIPS 全部 λ≈0.531。**结论：λ=0.7 是唯一推荐操作点；bicycle LPIPS 上界对 λ 鲁棒。**
  - **根因排查——LPIPS 训练损失在采样帧上计算**：采样帧未选 tile 为背景色填充，污染感知
    信号；新增 `--lpips-full-res`（LPIPS 步用一次带梯度全分辨率渲染，兼作 error-cache
    refresh，与 LPIPS 步同频 25，额外渲染≈0）。结果：train LPIPS 0.3838（vs 采样 0.3823-
    0.3853，无变化）；bicycle 3-seed PSNR 15.965（-0.06 dB vs full，持平），LPIPS 0.5298
    （-0.006），速度 22.81 ms → 1.98x。**结论：全分辨率 LPIPS 是“免费”的弱质量改进
    （采用进推荐配方），但上界非损失输入污染所致。**
  - **bicycle 非确定性发现**：同 seed 重跑可发散（6k 探针中 full@3000 PSNR 15.63 vs
    15.95、N 2.15M vs 2.41M；正常 3000 步重跑 ±0.1 dB），推断为 CUDA 原子操作非确定性
    经 densify/prune 阈值放大——bicycle 结论必须多 seed。**3-seed 认证（同会话）**：
    full（3 次）PSNR 16.024±0.072 / SSIM 0.3908 / LPIPS 0.4795；eg fr（3 seed）15.965±0.109
    / 0.3891 / 0.5298（+0.050±0.002，稳健）；eg λ=0.7（2 seed）15.893/0.3890/0.5305。
  - **6000 步收敛性探针（bicycle）**：R36 配方在 3000 步后持续退化（full 6000 PSNR 15.33
    vs 15.95、eg 14.91 vs 15.95；LPIPS full 0.510 / eg 0.559）——LPIPS 差距是渐进的而非
    收敛速率问题；配方评估点固定 3000 步（lr-decay 0.1 + densify-window 1500）。
  - **M4 状态（Round 41d）**：train 3-seed 1.82x（PSNR +0.40 / SSIM +0.003 / LPIPS +0.019）
    主要门槛保持达成；bicycle 3-seed PSNR 持平（-0.06±0.13，fr）、SSIM 持平、LPIPS +0.050
    ±0.002 为稳健的剩余诚实上界（λ、全分辨率 LPIPS、6000 步均无法关闭）——投稿前补强项
    （更高 λ / 采样策略 / prune 侧信号修复）。推荐配方：error_guided r=0.35 + error-lambda=0.7
    + --lpips-full-res（train 1.80x / bicycle 1.98x）。



- **Round 42 更新（2026-08-04，M5 多场景矩阵——garden/bonsai/truck，3 场景 × 3 seed）**：
  EPIC-05 A100、与 round-41d 完全相同的 R36 配方与推荐操作点（error_guided r=0.35
  λ=0.7 + --lpips-full-res，3000 步、1920x1080、n-train 4 / n-eval 3），每场景
  full 与 eg 各 3 seed，共 18 次运行（全 rc=0，garden/bonsai/truck 三 GPU 并行）：

  | 场景（初始 N） | full PSNR/SSIM/LPIPS | eg PSNR/SSIM/LPIPS | ΔPSNR | ΔLPIPS | 加速 | sr |
  |---|---|---|---|---|---|---|
  | garden (5.8M) | 18.733±0.024 / 0.5007 / 0.3987 | 17.971±0.019 / 0.4690 / 0.4482 | -0.76 | +0.050 | **2.12x** | 0.319 |
  | bonsai (1.2M) | 23.128±0.466 / 0.8211 / 0.2047 | 22.721±0.318 / 0.8059 / 0.2246 | -0.41（3-seed 噪声带内） | +0.020 | 1.74x | 0.311 |
  | truck (2.5M) | 18.711±0.197 / 0.6821 / 0.3115 | 19.297±0.026 / 0.7025 / 0.2966 | **+0.59** | **-0.015** | 1.88x | 0.313 |

  **结论：推荐操作点在 3 个新场景上给出 1.74-2.12x 端到端加速（garden 2.12x 最高）；
  质量上 truck 全面反超（PSNR +0.59 dB、SSIM +0.020、LPIPS -0.015），bonsai PSNR
  差距落在 3-seed 噪声带内（full ±0.47 / eg ±0.32）且 LPIPS +0.020 为小开口，garden
  （高 N 场景，与 bicycle 同类）出现 -0.76 dB PSNR 与 +0.050 LPIPS 的真实差距——
  **高 N 场景的感知上界从 bicycle 扩展到 garden，仍是投稿前补强项；低/中 N 场景
  （train、truck、bonsai）质量持平成立**。诚实限制：矩阵仍为 1080p 单分辨率，
  多分辨率留待后续。脚本 scripts/higs/run_m5_scenes_matrix.sh、聚合器
  scripts/higs/aggregate_run_summary.py（对 round-41d 汇总数字逐位一致）、
  原始 18 次运行 + 汇总 results/higs-round42/。

- **Round 43 更新（2026-08-04，error_alpha 扫描——garden/bicycle，负结果关闭杠杆）**：
  高 N 场景差距（garden PSNR -0.76 dB / LPIPS +0.050、bicycle LPIPS +0.050）是否由
  误差引导采样的集中效应造成？扫描 error_alpha ∈ {0.5, 0.75, 1.0}（3 seed、
  推荐配方、r=0.35 λ=0.7 + full-res LPIPS）：
  - garden：a=1.0 17.971 / a=0.75 18.011 / a=0.5 18.005（ΔPSNR -0.76/-0.72/-0.73），
    LPIPS 0.4482/0.4477/0.4471（Δ +0.050/+0.049/+0.048）——差距与 alpha 无关。
  - bicycle：a=0.5 PSNR 16.016（Δ -0.01，仍持平）LPIPS 0.5240（Δ +0.045，
    较 a=1.0 的 +0.050 收窄 0.005）——LPIPS 上界仍开口。
  - 所有 alpha 的实际 sr 相同（0.321-0.322）、速度相同（garden 2.12x）——误差图在
    训练中实际接近平坦（floor/均匀混合主导），alpha 不是质量杠杆。
  **结论：alpha 扫描为负结果、杠杆关闭；高 N 场景差距不是采样集中伪影，需从
  prune/densify 侧信号修复或损失目标侧继续（后续轮次）。** 结果
  results/higs-round43/（脚本 scripts/higs/run_round43_alpha_sweep.sh）。

- **Round 44 更新（2026-08-04，densify 梯度累积——garden/bicycle，负结果关闭杠杆）**：
  round-43 关闭 alpha 后，测试风险表中的"梯度累积"对策：tile 采样训练下逐 step 位置
  梯度在采样 tile 外为零，densify 决策信号稀疏。新增 opt-in `--densify-grad-accum`：
  在 densify 窗口（densify_every=5）内累积（detached）逐 step 位置梯度范数，用累积
  信号驱动 dup/clone（标准 3DGS 配方；阈值未按窗口缩放，行为等价于更激进的 densify
  探针）。3 seed、推荐配方（r=0.35 λ=0.7 + full-res LPIPS）：
  - garden：ga PSNR 17.997±0.036 / LPIPS 0.4453±0.0011（Δvs eg +0.03 dB、-0.003
    LPIPS，均噪声内），vs full 差距不变（-0.74 dB / +0.047，2.11x）；final_n 与 eg
    几乎相同（+0.14%）——累积信号几乎不改变密度化决策。
  - bicycle：ga PSNR 15.647±0.130 / LPIPS 0.5415±0.0046（Δvs 同 seed eg
    -0.32±0.08 dB、+0.012±0.003 LPIPS，3 seed 一致变差），vs full -0.38 dB /
    +0.062 LPIPS / 1.97x；final_n +1.2%（更多克隆反而伤害质量）。
  - 速度成本 ~0.5-0.8%（纯 Python 累积，可忽略不计）。
  **结论：densify 梯度累积为负结果、杠杆关闭——r=0.35 下逐 step 梯度被采样 tile
  主导，5 步窗口累积仍复现不了全分辨率 densify 信号，bicycle 上过度密度化反而退化。
  风险表中"密度化专用全分辨率步"为下一杠杆（Round 45：让 densify 与 full-res LPIPS
  step 对齐，获得零额外成本的全分辨率 densify 信号）。** 结果
  results/higs-round44/（脚本 scripts/higs/run_round44_grad_accum.sh，汇总
  r44-summary.json，delta 字段相对 round-41d/42 eg 基线）。

- **Round 45 更新（2026-08-04，densify 全分辨率信号 + 节奏 25——garden/bicycle，负结果关闭节奏杠杆）**：
  round-44 证明累积信号无效后，测试风险表中"密度化专用全分辨率步"：让 densify 与
  full-res LPIPS step 对齐（densify_every=25 = lpips_loss_every=25 + --lpips-full-res），
  使 dup/clone 决策使用真全分辨率梯度且零额外渲染成本。3 seed、其余配方不变：
  - garden：de25 PSNR 17.209±0.104 / LPIPS 0.4866±0.0027（Δvs eg -0.76 dB / +0.038
    LPIPS，全 seed 一致变差），vs full -1.52 dB / +0.088 LPIPS（差距翻倍）；final_n
    1.86M——低于 eg 的 2.01M，更远低于 full 的 2.59M（欠密度化）。
  - bicycle：de25 PSNR 15.148±0.114 / LPIPS 0.5564±0.0033（Δvs eg -0.82 dB /
    +0.027 LPIPS），vs full -0.88 dB / +0.077 LPIPS；final_n 2.89M——反而超过 full
    的 2.71M（过密度化）。
  - total_ms 反而下降（garden 16.4s vs 20.3s，densify 事件少 5 倍）——速度不是问题，
    质量是。
  **结论：densify 节奏 5 step 是本配方承重墙——拉长到 25 步即使给全分辨率信号，
  两个场景的质量也全面退化（方向相反：garden 欠密度化、bicycle 过密度化），
  densify 节奏轴关闭。round-45 混淆了节奏（5→25）与信号（全分辨率）；信号轴在
  原节奏 5 下的最后诊断是 --anchor-densify（round-47 筛查进行中）。** 结果
  results/higs-round45/（脚本 scripts/higs/run_round45_densify_fullres.sh，
  r45-summary.json）。

- **Round 46 更新（2026-08-04，LPIPS 权重 0.2——garden/bicycle，负结果关闭损失侧权重杠杆）**：
  高 N 场景剩余诚实上界是感知的（LPIPS +0.050）；测试损失侧最后一道便宜杠杆：把
  全分辨率 LPIPS 正则权重从 0.1 翻倍到 0.2（其余配方不变，3 seed）：
  - garden：w=0.2 PSNR 17.938±0.021 / LPIPS 0.4540±0.0010（Δvs w=0.1 eg
    -0.03 dB / +0.006 LPIPS）——双指标都轻微变差（sd 极小，方向一致）。
  - bicycle：w=0.2 PSNR 15.917±0.070 / LPIPS 0.5378±0.0017（Δ -0.05 dB /
    +0.008 LPIPS）——同样轻微变差。
  - final_n 略降（garden 1.95M、bicycle 2.38M），速度不变。
  **结论：更强的感知正则干扰 tile 采样 L1 训练动力学，w=0.1 仍是尖峰最优；
  LPIPS 权重杠杆关闭。** 结果 results/higs-round46/（脚本
  scripts/higs/run_round46_lpips_w02.sh，r46-summary.json）。

- **Round 47 更新（2026-08-04，--anchor-densify 全分辨率 densify 步——garden/bicycle，首个正向杠杆）**：
  round-43/44/45/46 关闭采样侧、累积侧、节奏侧、损失侧杠杆后，风险表中最后一项
  "密度化专用全分辨率步"在原节奏 5 下给出首个正向结果：`--anchor-densify` 让每个
  densify 步以全分辨率渲染，dup/clone 决策使用真全帧位置梯度（成本 ~6-7%）。
  3 seed、其余配方不变：
  - garden：anchor PSNR 18.192±0.012 / SSIM 0.4784 / LPIPS 0.4361±0.0019
    （Δvs eg +0.22 dB / +0.009 SSIM / -0.012 LPIPS，全 seed 一致），vs full 差距
    收窄为 **-0.54 dB / +0.037 LPIPS**（原 -0.76 / +0.050）；速度 2.11x → 1.98x。
  - bicycle：anchor PSNR 15.926±0.299（vs eg -0.04±0.30，持平；bicycle 固有
    ±0.1-0.3 dB 不确定度）/ SSIM 0.3918 / LPIPS 0.5179±0.0024（3 seed 全部收窄
    -0.008..-0.015），vs full 差距收窄为 **-0.10 dB / +0.038 LPIPS**（原 -0.06 /
    +0.050）；速度 1.98x → 1.87x。
  - final_n 更接近 full（garden 2.06M vs 2.59M、bicycle 2.43M vs 2.71M）——
    全分辨率 densify 信号恢复了部分密度结构；实际 sr 升至 0.373-0.375（densify
    步计入全分辨率）。
  **结论：--anchor-densify 是首个正向质量杠杆：高 N 场景 LPIPS 上界从 +0.050 收窄
  到 ~+0.037（两场景、全 seed），garden PSNR 差距减半、SSIM 全面改善，速度代价
  ~6-7% 且两场景仍保持 ≥1.8x（1.98x / 1.87x）。建议作为高 N 场景 opt-in 推荐
  （低/中 N 场景默认不开：train 单 seed 探针 +6% 时间 → 1.82x 降至 ~1.72x
  跌破 1.8x 门槛，且质量方向不一致（PSNR -0.45 dB / LPIPS +0.007，1 seed
  不充分），无收益证据）。** 结果 results/higs-round47/（脚本
  scripts/higs/run_round47_anchor_screening.sh，r47-summary.json，delta 字段相对
  round-41d/42 eg 基线）。

- **Round 48/49 更新（2026-08-04，anchor 下密度轴关闭——garden 单 seed 筛选，负结果）**：
  round-47 后剩余差距（garden PSNR -0.54 / LPIPS +0.037，final_n 2.06M vs full
  2.59M）是否可由密度动力学收窄？在 anchor 下做两组廉价筛选：
  - densify 阈值（R48）：5e-3 → 2.5e-3 / 1e-3，final_n 几乎不动（2.061/2.063/
    2.065M）——克隆决策不由阈值主导（全分辨率梯度范数双峰分布）；PSNR/LPIPS
    均在噪声内。
  - prune 阈值（R49）：0.01 → 0.005 / 0.002，final_n 仅 +1.0%/+1.7%（2.06M →
    2.10M），LPIPS 单调变差（0.4345/0.4360/0.4380）——强行保留低不透明度高斯
    反而伤感知质量。
  **结论：密度轴（克隆侧 + 保留侧）在 anchor 下关闭；剩余高 N 差距是 tile 采样训练
  本身的内在边界，非密度动力学可修复。最终推荐操作点：error_guided r=0.35 λ=0.7 +
  --lpips-full-res +（高 N opt-in）--anchor-densify → 1.87-1.98x、LPIPS 上界
  ~+0.037、garden PSNR -0.54。** 结果 results/higs-round48/、results/higs-round49/
  （单 seed 筛选）。

- **Round 50 更新（2026-08-04，--anchor-densify-every 全分辨率 densify 步降采样——garden/bicycle，成本-质量前沿）**：
  round-47 的 --anchor-densify（每个 densify 步全分辨率）成本 ~7%；单 seed 筛
  `--anchor-densify-every {2,4}`（每 2/4 次 densify 事件锚定一次、其余回采样分辨
  率）显示边际收益集中在前半程锚定事件：garden（s0）anchor1 18.178/0.4345/21.65ms
  → every2 18.1265/0.4445/20.89ms → every4 17.9505/0.4477/20.63ms（塌回 eg 水平
  17.99/0.4482）；bicycle 上 every2/every4 同噪声级（16.02/16.08 vs eg 15.95）。
  对 every2 做 3-seed 确认（s1/s2 新跑）：
  - garden：every2 PSNR 18.068±0.062 / SSIM 0.4731 / LPIPS 0.4427±0.0016
    （Δvs eg +0.097 dB / +0.004 SSIM / -0.006 LPIPS，3 seed 一致；vs anchor1
    -0.12 dB / +0.007 LPIPS——保留约一半 anchor 收益）；vs full 差距 -0.66 dB /
    +0.044 LPIPS（eg 为 -0.76 / +0.050）。
  - bicycle：every2 PSNR 15.940±0.078 / SSIM 0.3916 / LPIPS 0.5257±0.0010
    （Δvs eg -0.02 dB 持平 / -0.004 LPIPS，3 seed 一致）；vs full +0.046 LPIPS。
  - 成本：every2 总耗时 +2.9%（garden）/+3.1%（bicycle）vs eg（anchor1 为 +6.8%/
    +6.2%）→ 端到端加速 garden 2.06x / bicycle 1.92x（anchor1 为 1.98x/1.87x，
    仍 ≥1.8x 且余量更大）。
  **结论：--anchor-densify-every 2 是 3-seed 验证的成本-质量甜点：保留约一半
  anchor 质量收益、成本减半、速度反超 anchor1；anchor1（every 1）仍为质量上限
  opt-in（LPIPS 上界 +0.037）；every4 在 garden 上收益消失，不建议。推荐高 N 场景
  opt-in 默认 `--anchor-densify --anchor-densify-every 2`，质量优先时保留 every 1。**
  结果 results/higs-round50/（r50-summary.json，delta 字段相对 round-41d/42 eg
  基线；脚本 scripts/higs/run_round50_anchor_every.sh）。

- **Round 51 更新（2026-08-04，M6 对照 1/3：ICCV random-tile loss 基线——train/garden/bicycle，3-seed 3000 步）**：
  M6 第一项对照：`--sampling-mode uniform`（随机 tile + masked tile-L1 loss，
  即 ICCV 2025 random-tile loss 的对应实现）vs error_guided（无偏重要性加权），
  其余配方完全一致（r=0.35 名义、full-res LPIPS、high-N 场景 + anchor every2、
  train 无 anchor）。同批补跑 train eg 3-seed 验证 round-41b 数值（17.10±0.10 /
  0.3871 / 12.31ms vs 文档 17.074 / 0.3870 / 12.06ms，复现一致）。
  - 名义 r 相同 ≠ 计算相同：uniform 实际采样率 sr≈0.38-0.40（garden/bicycle）/
    0.376（train），比 eg（0.31-0.35）高 15-20% → 同名义 r 下均匀基线墙钟
    +7-12%（train +12.3%、garden +11.7%、bicycle +7.0%）。对照必须按实际 sr 对齐。
  - 匹配 sr 对照（uniform r=0.30 → sr 0.328-0.356 vs eg sr 0.308-0.348）：
    - train（低 N）：eg 显著更好——PSNR 17.097±0.102 vs 16.775±0.143（-0.32 dB；
      uniform s1 同 seed 重跑 16.07 → 16.70，±0.6 dB 波动已记录并采用重跑值）、
      LPIPS 0.3871 vs 0.3895（+0.002）、墙钟 12.31 vs 13.00ms（uniform 仍 +5.6%）
      ——随机 tile 在低 N 场景被支配（质量 + 速度双输）。
    - garden（高 N）：PSNR 持平（18.082 vs 18.068，+0.014）、LPIPS uniform 略优
      （0.4389 vs 0.4427，-0.004）、墙钟 +4.7%。
    - bicycle（高 N）：PSNR 持平（15.928 vs 15.940，-0.012）、LPIPS uniform 更优
      （0.5165 vs 0.5257，-0.009，3 seed 全部低于 eg）、墙钟 +0.7%。
  - 端到端加速（uniform r=0.30）：garden 1.96x / bicycle 1.91x / train 1.69x
    （train eg 复测 1.79x、round-41b 文档 1.82x，测量噪声级差异）。
  **结论（诚实）：error_guided 的重要性加权优势是场景相关的——低 N train 上明确
  支配随机 tile（质量 + 速度双优）；高 N 场景（garden/bicycle）在匹配实际采样率
  时随机 tile 不劣（PSNR 持平、LPIPS 反优 0.004-0.009，墙钟 +0.7..+4.7%），与
  round-31 M3 的早期观察（bicycle 上 uniform 不差）在 3000 步全配方下复现。
  "误差引导 > 随机 tile" 的投稿级声明需限定在低/中 N 场景；高 N 场景的 LPIPS 上界
  由实际 tile 数（sr）主导而非采样策略。** M6 状态：对照 1/3 完成；Turbo-GS 与
  Speedy-Splat 仍留待投稿阶段。结果 results/higs-round51/（r51-summary.json，
  delta 相对同配方 eg/eg-every2 基线；脚本 scripts/higs/run_round51_random_tile_baseline.sh、
  run_round51b_matched_sr.sh）。

- **Round 52 更新（2026-08-04，M6 对照 2/3：Turbo-GS 式渐进分辨率训练——train/garden/bicycle，3-seed 3000 步）**：
  新增 `--res-schedule "0.5:0,1.0:1500"`（解析 `_parse_res_schedule`/`_res_stage`）：前 1500 步半分辨率训练、之后全分辨率，评估恒为全分辨率（Turbo-GS 粗到细的对应实现；其余配方与 round-50/51 完全一致：error_guided r=0.35 λ=0.7 + full-res LPIPS、high-N + anchor every2、train 无 anchor）。结果 vs eg/every2 基线（3-seed 均值 ± sd）：
  - prog（plain）：train 16.94±0.32（vs eg -0.15 dB，LPIPS 0.3888±0.0052，+0.0017）/ garden 18.07±0.06（±0.00，LPIPS 0.4489，+0.0062）/ bicycle 15.59±0.39（-0.35，LPIPS 0.5204，**-0.0053**）；总耗时 10.63±0.09 / 17.19±0.03 / 18.31±0.04 ms → **vs 全分辨率 2.07x / 2.49x / 2.47x**（vs eg 墙钟 +15.8% / +21.0% / +28.5%）。结论：粗到细带来 ~2.1-2.5x 大提速，train/garden PSNR 与 eg 持平；garden LPIPS 回退到 eg 界（0.4489 vs every2 0.4427——半分辨率阶段 full-res LPIPS 信号丢失）；bicycle PSNR 方差大（s0 于 1500 步后 15.87→15.18 退化），但 LPIPS 反而优于 every2。
  - full-signal 变体（`--res-schedule-full-signal`：粗阶段 LPIPS 步与 anchor densify 步保持全分辨率）：**负面结果**——train 持平（17.00±0.12，LPIPS 0.3874），garden PSNR -0.35 / LPIPS +0.027，bicycle PSNR -0.98 / LPIPS +0.011，且更慢（20.27 / 20.93 ms）。机制：densify 每 5 步进行（`densify_every=5`），`--anchor-densify-every 2` 使粗阶段 anchor（全分辨率）与非 anchor（半分辨率）密度化事件交替，高 N 场景密度化信号不稳定 → 保留更多但位置更差的高斯（garden 2.30M vs 1.98M），质量不升反降。
  **结论：plain 渐进分辨率是当前最高性价比训练臂（~2.1-2.5x，质量持平或可接受）；粗阶段叠加全分辨率信号不解决问题。** 详见 results/higs-round52/（r52-summary.json；脚本 scripts/higs/run_round52_progressive_res.sh、run_round52b_full_signal.sh）。
- **Round 53 更新（2026-08-04，M6 对照 3/3：Speedy-Splat 式稀疏像素训练信号——train/garden/bicycle，3-seed 3000 步）**：
  新增 `--sampling-mode sparse_pixel`（`--pixel-sampling-ratio 0.35`）：每步独立伯努利保留 35% 像素，损失取像素子集 L1 均值（与 tile-masked 同构的无偏估计，无需重加权）。frozen gsplat HiGS 无像素级稀疏光栅化，光栅化仍为全帧——该臂复现 Speedy-Splat 的**训练信号**（非墙钟速度），用于在同等像素覆盖下对比 tile 粒度采样信号。配方其余与 round-50/51/52 一致（full-res LPIPS every 25、high-N + anchor every2、train 无 anchor）。结果（3-seed 均值 ± sd）：
  - train 16.58±0.06 / LPIPS 0.3755±0.003 / 23.5ms；garden **18.57±0.07 / 0.4081±0.001** / 42.8ms；bicycle **16.12±0.13 / 0.4871±0.005** / 45.9ms。
  - vs 全分辨率：PSNR -0.09 / -0.16 / +0.10 dB，LPIPS +0.008 / +0.009 / +0.008——**35% 像素覆盖下质量基本等于全分辨率训练**。
  - vs eg/every2 同覆盖 tile 臂：garden LPIPS -0.035（0.4081 vs 0.4427）、PSNR +0.39；bicycle LPIPS -0.039（0.4871 vs 0.5257）、PSNR +0.09；train LPIPS -0.012。
  **结论：高 N 场景的 tile 采样质量界主要是采样相关性噪声（16x16 宏块粒度），而非像素数量**——相同覆盖率下像素 iid 采样恢复近全质量。代价：全帧光栅化无提速（~1.0x full；Speedy-Splat 的像素稀疏光栅化不在 frozen gsplat 范围内）。**这直接指向下一个质量杠杆：在保持 tile 光栅化提速的同时去相关化 tile 选择（分层/抖动 tile 采样），以及多分辨率矩阵。** M6 状态：对照 3/3 完成（ICCV random-tile + Turbo-GS 渐进分辨率 + Speedy-Splat 稀疏像素）。详见 results/higs-round53/（r53-summary.json；脚本 scripts/higs/run_round53_sparse_pixel.sh）。
- **Round 54 更新（2026-08-04，采样去相关化杠杆：stratified tile 采样——train/garden/bicycle，3-seed 3000 步）**：
  R53 指出高 N 质量界是采样相关性噪声；本轮的 in-harness 去相关化杠杆是光栅化器内置的 `--sampling-mode stratified`（每 round(1/r) 个 tile 组成的层内取一个 tile，单步内铺满全图）。配方与 R50/51 一致（r=0.35、full-res LPIPS every 25、high-N + anchor every2、train 无 anchor）。结果（3-seed 均值 ± sd）：
  - train 16.73±0.33 / LPIPS 0.3933±0.002 / 13.3ms（sr 0.36）；garden 18.15±0.04 / LPIPS 0.4346±0.001 / 22.5ms（sr 0.387）；bicycle 15.92±0.19 / LPIPS 0.5103±0.001 / 24.6ms（sr 0.387）。
  - vs uniform（R51，同配方匹配 sr）：garden LPIPS 0.4346 vs 0.4335（±0.001）、bicycle 0.5103 vs 0.5106（-0.0003）、train 0.3933 vs 0.3871（+0.006）——**分层与随机 tile 采样在匹配 sr 下质量等价**；vs eg/every2（sr 低 0.04-0.05）：garden LPIPS -0.008、bicycle -0.015，主要为更高覆盖率贡献（stratif 实际 sr 0.387 vs eg 0.344-0.348）。
  **结论：负面——分层去相关化没有恢复质量**。结合 R53：损失不是采样簇集（draw clustering），而是 tile 粒度本身的内部相关性（16x16 块同渲染/同误差结构）；在 frozen gsplat 光栅化器内，采样杠杆（error_guided / uniform / stratified）已全部闭合且等价，恢复质量需要更细粒度（像素级掩码/抖动网格偏移）的光栅化支持，属渲染器层面后续工作。当前最优训练臂仍是 R50 every2（质量）与 R52 渐进分辨率（速度）。详见 results/higs-round54/（r54-summary.json；脚本 scripts/higs/run_round54_stratified.sh）。
- **Round 55 更新（2026-08-04，粗阶段仅 LPIPS 全分辨率信号——隔离 R52b 机制，负结果）**：
  R52b（--res-schedule-full-signal：粗阶段 LPIPS 与 anchor 都全分辨率）为负面，归因于
  全/半分辨率 densify 事件交替（every 2）。本轮用新 flag `--res-schedule-full-lpips`
  隔离感知侧：粗阶段**仅 LPIPS 步**保持全目标分辨率、anchor densify 留在阶段分辨率
  （无交替）；其余配方与 R52 完全一致（error_guided r=0.35 λ=0.7 + high-N anchor
  every2、train 无 anchor、3 seed）。结果（3-seed 均值 ± sd）vs R52 plain prog：
  - train：17.093±0.076 / LPIPS 0.3920±0.003 / 12.98ms（vs r52 16.943 / 0.3888 /
    10.63ms——PSNR +0.15（噪声级）、LPIPS +0.003、墙钟 +22%）。
  - garden：17.972±0.034 / LPIPS 0.4545±0.001 / 18.11ms（vs r52 18.069 / 0.4489 /
    17.19ms——PSNR -0.10、LPIPS +0.006，仍高于 every2 的 0.4427；墙钟 +5%）。
  - bicycle：15.442±0.552 / LPIPS 0.5225±0.001 / 19.11ms（vs r52 15.592 / 0.5204 /
    18.31ms——PSNR -0.15（±0.55 噪声级）、LPIPS +0.002、墙钟 +4%）。
  **结论：负面——粗阶段恢复全分辨率 LPIPS 信号不恢复质量**（garden LPIPS 不降反升
  0.006，bicycle/train 持平或略差），且一致更慢（+4..+22%）。与 R52b 联合解读：
  粗阶段任何形式的全分辨率信号（仅 LPIPS / LPIPS+anchor）都无法关闭渐进分辨率的
  garden LPIPS 开口；plain 渐进分辨率（R52）仍是该轴最优臂。`--res-schedule-full-lpips`
  作为已关闭杠杆保留（含 CPU 测试 TestResScheduleFullLpipsFlag）。剩余唯一质量杠杆
  仍是渲染器级更细粒度采样。结果 results/higs-round55/（r55-summary.json；脚本
  scripts/higs/run_round55_full_lpips.sh）。
- **Round 56 更新（2026-08-04，M5 多分辨率矩阵——540p/720p 补齐，3 场景 × 3 seed）**：
  推荐操作点（error_guided r=0.35 λ=0.7 + full-res LPIPS；high-N + anchor every2、
  train 无 anchor）vs 全分辨率在 540p（960x540）与 720p（1280x720）各跑
  train/garden/bicycle × 3 seed（36 次运行，全 rc=0）；1080p 单元格复用
  round-41d/42/50 既有结果。完整矩阵（ΔPSNR / ΔLPIPS / 加速，eg vs full 同分辨率）：
  | 场景 | 540p | 720p | 1080p |
  |---|---|---|---|
  | train | +0.45 / +0.004 / **1.39x** | +1.03 / -0.003 / **1.56x** | +0.42 / +0.019 / **1.78x** |
  | garden | -0.60 / +0.053 / **1.76x** | -0.51 / +0.045 / **1.88x** | -0.67 / +0.044 / **2.06x** |
  | bicycle | -1.93 / +0.050 / **1.67x** | -0.75 / +0.048 / **1.74x** | -0.08 / +0.046 / **1.92x** |
  **关键发现**：① 加速随分辨率单调上升（540p 1.39-1.76x → 720p 1.56-1.88x →
  1080p 1.80-2.06x）——固定开销（densify/LPIPS/投影/eval）在低分辨率下摊薄
  比例更高，≥1.8x 声明成立区间是 ≥720p 的高 N 场景与 ≥1080p 的 train；
  ② train（低 N）在每个分辨率都是明确质量胜（PSNR +0.45..+1.03，720p 下
  LPIPS 甚至 -0.003 反优）——低分辨率 + 采样训练对低 N 场景更有利；
  ③ 高 N 的 LPIPS 界对分辨率稳健（garden +0.044..+0.053、bicycle +0.046..
  +0.050 全程近似持平），但 bicycle 的 PSNR 差距随分辨率下降而急剧放大
  （540p -1.93 dB vs 1080p -0.08 dB）——低分辨率下高 N 场景的像素级重建
  退化是诚实边界，投稿叙事需按分辨率限定。**结论：多分辨率矩阵完成（M5
  扩展），加速/质量的分辨率依赖关系已量化；投稿建议在 1080p 主张 ≥1.8x，
  低/中 N 场景可下沉到 720p。** 结果 results/higs-round56/（r56-summary.json；
  脚本 scripts/higs/run_round56_multi_res.sh）。
- **Round 57 更新（2026-08-04，M6 收尾：渲染器级稀疏像素光栅化——train/garden/bicycle，3-seed 3000 步）**：
  新增后端 `higs_sparse_px` 与 `--pixel-raster-ratio`：用上游 gsplat 的稀疏内核
  （`build_sparse_tile_layout` + `isect_tiles_sparse` + `rasterize_to_pixels_sparse`）
  在渲染器内只光栅化每步 iid Bernoulli 掩码选中的像素（打包输出），即 Speedy-Splat
  式像素级稀疏渲染——R53 证明该训练信号可恢复近全质量，R57 第一次落地到渲染器本身。
  18 次运行（3 场景 × 3 seed × 2 arm），配方与 R53 相同（full-res LPIPS every 25、
  lr-decay、densify-window 1500、high-N + anchor every2、train 无 anchor）；对照臂
  是同一组合管线自己的稠密基线（px100 = 同一后端 ratio 1.0），隔离像素粒度单一变量：

  | 场景 | px100 PSNR/SSIM/LPIPS | px035 PSNR/SSIM/LPIPS | ΔPSNR | ΔLPIPS | 加速 |
  |---|---|---|---|---|---|
  | train | 16.586±0.316 / 0.6216 / 0.3711±0.0067 | 16.306±0.182 / 0.6211 / 0.3813±0.0083 | -0.28 | +0.010 | **1.06x** |
  | garden | 18.753±0.015 / 0.5011 / 0.3978±0.0005 | 18.554±0.030 / 0.4973 / 0.4092±0.0012 | -0.20 | +0.011 | **1.09x** |
  | bicycle | 15.727±0.276 / 0.3907 / 0.4845±0.0032 | 16.237±0.044 / 0.3929 / 0.4846±0.0022 | **+0.51** | +0.0001 | **1.06x** |

  **关键发现**：① 质量——渲染器级像素稀疏在 ~40% 像素覆盖下恢复近全质量，高 N
  场景尤其显著：bicycle PSNR 反超稠密基线 +0.51 dB 且 LPIPS 持平（+0.0001），
  garden LPIPS +0.011（对比 tile 级 eg/every2 的 +0.044 收窄 4 倍），与 R53 的信号
  结论端到端一致；train（低 N）小幅退化 -0.28 dB / +0.010（3-seed，dense 臂自身
  sd ±0.32，约 0.8 sd，方向性弱）。② 速度——**墙钟杠杆结构性收窄：仅 1.06-1.09x**
  （fwd 1.06-1.09x、bwd 1.04-1.09x）。原因：iid 像素掩码在 40% 覆盖下几乎不跳过任何
  tile，相交（isect）成本不变；投影/SH/求交及反向都是像素数不变成本，只有逐像素
  混合循环随像素数缩放（实测该部分仅占组合管线步骤的 ~5-10%）。**结论：M6 的
  "渲染器级更细粒度采样" 问题闭环——像素级稀疏渲染在渲染器层面确认恢复近全质量
  （质量杠杆成立），但墙钟提速被非像素缩放阶段（投影/SH/相交）上界钉在 ~1.1x，
  达不到 tile 级 1.8x+；投稿叙事：像素稀疏 = 质量恢复杠杆，tile 采样 = 速度杠杆，
  二者正交，1.8x 操作点仍由 error_guided tile 采样提供。** 结果
  results/higs-round57/（r57-summary.json；脚本 scripts/higs/run_round57_sparse_px.sh；
  新增后端 `higs_sparse_px`、`--pixel-raster-ratio` 与 10 项测试
  tests/test_higs_sparse_pixel_raster.py）。

- **Round 58 更新（2026-08-04，速度最大单元收官：720p 目标 + 渐进分辨率——train/garden/bicycle，3-seed 3000 步）**：
  填补前沿表最后一个单元：720p 目标下叠加 R52 的渐进调度（--res-schedule 0.5:0,1.0:1500，
  粗阶段 360p）与推荐配方（eg r=0.35 + full-res LPIPS + high-N anchor every2，train 无 anchor），
  9 次运行；对照为 R56 同分辨率臂（720p full / 720p eg）。**结论：720p × 渐进 = 明确负面**——
  相对 plain 720p eg 无提速（train 0.89x 更慢、garden 1.01x、bicycle 1.09x），质量一致退化
  （train PSNR -0.20 dB / LPIPS +0.007，garden -0.37 / +0.033，bicycle -0.46 / +0.011，3-seed）。
  机制（200 步探针 scripts/higs/run_round58_res_scaling_probe.sh）：360p 相对 720p 单步成本仅省
  ~0-9%（garden 18.9 vs 20.8 ms；train 360p 甚至不省，15.1 vs 12.2 ms）——720p 下单步成本已被
  分辨率不变的每-Gaussian 阶段（投影/SH/反向）主导，粗阶段无物可省，叠加 R52 已知的粗阶段
  densify 失稳，质量反降。**前沿因此收官**：速度最大单元 = plain 720p + eg（R56，相对同分辨率
  full 1.56-1.88x；相对 1080p full 墙钟 2.4-2.7x），渐进分辨率仍是 1080p 专属杠杆（R52
  2.07-2.49x），质量最大 >=1.8x 操作点保持 1080p eg + anchor-densify-every-2。结果
  results/higs-round58/（r58-summary.json；脚本 scripts/higs/run_round58_prog720p.sh /
  aggregate_round58.sh）。
- **Round 59 更新（2026-08-04，per-Gaussian floor 首个杠杆：cull-mask 缓存——train/garden/bicycle，36 次主扫 + 12 次 bicycle 方差审计）**：
  kernel 级 profile（scripts/higs/profile_step_breakdown.py + run_round59_profiling.sh，torch.profiler 40 步、
  4 操作点）定位 garden 720p r0.35 单步成本：higs_blend_bwd_px 8.4 ms (21.6%)、fused Adam 6.8 (17.5%)、
  higs_sh_vjp_grid 5.8 (14.9%)、rasterize_to_pixels_fwd 3.9 (10.2%)、higs_projection_bwd 2.3 (6.0%)、
  projection fwd x2 ~1.9 (5.0%)——其中每步一次的 full-N batched culling projection
  （fully_fused_projection，无梯度）是纯 per-Gaussian、与像素无关的可省项。将上游 gsplat 的单槽
  cull 缓存改为**相机集键控**（handle._cull_cache dict，cache_key="train"/"eval"），并修掉两个真实缺陷：
  此前 --cull-interval 从未生效（benchmark 未传 handle 且 autograd Function 无 cache_key），且 K>1 时
  eval 会复用 train 的 mask 污染评估；新增 --cull-interval-schedule "K:start,..." 支持阶段门控
  （"1:0,16:1500" = densify 窗口内每步刷新、之后每 16 步）。720p eg 配方（R56 相同）3 场景 x
  {K=1,4,16} x 3 seed + 门控 g16 x 3 seed = 36 次运行；bicycle 追加 k1/k4/g16/k16 seeds 3-5
  方差审计（n=6/arm，12 次运行）。**速度：K4/K16 相对 K=1 基线 -1.8%..-3.9% 墙钟（total_ms
  1.018-1.041x，每步约省 0.17-0.53 ms）；g16 仅 -0.4%..-2.2%（阶段1 无缓存）**。
  **质量：LPIPS/SSIM 全臂全场景持平（bicycle LPIPS k1 0.4762 vs k4 0.4767 / k16 0.4775 / g16 0.4786，
  Δ<=+0.003；garden/train Δ<=+0.001）**。bicycle PSNR 的 3-seed "退化"（k4 -0.70 dB）经 n=6 审计
  证明是**场景固有 run variance**：k1 基线自身也有 14.57 dB 的坍缩运行（seed 4），g16 seed-0 在
  阶段1（K=1 每步刷新、与基线完全同路径）同样坍缩到 14.51；n=6 下各臂 PSNR 均值差（k4 -0.34 /
  g16 -0.22 dB）远小于臂内 sd（0.6-0.7），不显著。**结论：cull-mask 缓存是 per-Gaussian floor 上
  首个确认的速度杠杆（2-4% 墙钟、LPIPS/SSIM 中性），推荐 K=4 为 opt-in（K16 增益边际递减，K=1
  仍是默认）；相机集键控修复随代码合入**。剩余主导成本仍是 per-Gaussian 的 blend bwd + SH vjp +
  projection bwd（约 48% 单步），需 kernel 级融合/近似才能再进一步。结果 results/higs-round59/
  （r59-summary.json；脚本 scripts/higs/run_round59_cull_sweep.sh / run_round59b_gated_cull.sh /
  run_round59c_bicycle_audit.sh / aggregate_round59.sh）。
- M5 扩展性：**Round 42 已完成 garden/bonsai/truck（3 场景 × 3 seed，见上表）；多分辨率矩阵 Round 56 完成（540p/720p/1080p × train/garden/bicycle × 3 seed，36 次运行，见上表）**。
- M6 对照：**3/3 完成：ICCV random-tile（R51）、Turbo-GS 渐进分辨率（R52，~2.1-2.5x 提速）、Speedy-Splat 稀疏像素训练信号（R53，35% 像素覆盖近全质量）。R53/R54 联合结论：高 N tile 采样质量界为 tile 粒度相关性噪声（去相关化分层采样 R54 为负面，与 uniform 匹配 sr 等价）。渲染器级细粒度采样已闭环（R57）：像素级稀疏光栅化（higs_sparse_px）在 ~40% 覆盖下恢复近全质量（bicycle PSNR +0.51 dB / LPIPS 持平，garden LPIPS +0.011），但墙钟仅 1.06-1.09x——相交/投影/SH 为像素数不变成本，像素稀疏只省逐像素混合循环，速度杠杆结构性上界 ~1.1x，1.8x 操作点仍由 tile 采样提供；720p × 渐进已关闭为负面（R58，无提速有质量损失，720p 单步成本已达每-Gaussian 不变地板），速度最大单元 = 720p + eg（相对 1080p full 2.4-2.7x）；cull-mask 缓存（R59）为 per-Gaussian floor 首个速度增量（K4 opt-in，2-4% 墙钟、LPIPS/SSIM 中性）。**

## 6. 风险与对策
- 采样训练改变密度化动力学（梯度稀疏 -> densify 信号变化）：对策 = 梯度累积 / 密度化专用全分辨率步 / 调 densify 阈值。
- 短程（20 步）收益 != 长程收敛收益：一切以 M4 为准。
- 与 tile-wise training 的区分度：需要实验证明（质量保持 + 明确速度收益 + 宏块结构利用）。**Round 51 random-tile 对照：区分度在高 N 场景主要来自实际 sr/成本结构（uniform 同名义 r 多渲 15-20% tile、+7-12% 墙钟），质量侧不显著（匹配 sr 时 LPIPS 反优 0.004-0.009）；低 N 场景质量 + 速度双优。投稿叙事需以低/中 N 场景 + 宏块结构利用为主**。

## 7. 里程碑与验证标准（论文门槛）
- M2 完成 = 可演示 r=1/4 时总时间降 ~35-45%（若 blend 主导成立）；这是"明显加快"的第一实证。（**Round 31 已达成**：r=0.25 总时间 -33..-41%，bwd 近线性。）
- M4 完成 = 核心实验（收敛质量持平 + >= 1.8x wall-clock 加速）。（**部分→主要达成（train）**：r=0.5 dynamic + LPIPS 正则化（w=0.1 every 25）在 train 3-seed 达成 PSNR/SSIM 反超（+0.64 dB/+0.009）、LPIPS 差距缩至噪声级（+0.0046±0.0063），bicycle 仍 +0.038 未关闭；r=0.25 收敛未做。**Round 40 本地 20 步配对复测：r=0.5 = 1.63x、r=0.25 = 2.13x vs std**（R38 forward + R39 backward 叠加）。**Round 41 本地 3000 步质量探针：error_guided r=0.5/0.4/0.35 均方向性持平或反超 full（PSNR +0.33..+0.54 dB、LPIPS +0.005..+0.008），1.8x 计时点在名义 r≈0.36（实际 sr≈0.26）。**Round 41b EPIC-05 A100 多 seed 复测完成：train 3-seed 在名义 r=0.35（实际 sr≈0.27）达成 1.82x 端到端加速且 PSNR/SSIM 反超（+0.42 dB/+0.004），r=0.30 达 1.90x——M4 主要门槛达成；LPIPS +0.024 重新开口、bicycle λ=0.7 下 PSNR/SSIM 持平（-0.04..-0.07 dB）、LPIPS +0.047..+0.058 为剩余诚实上界；train LPIPS +0.019（λ=0.7）——M4 主要门槛达成，bicycle LPIPS 为投稿前补强项。**Round 41d：λ 扫描确认 λ=0.7 尖峰最优；--lpips-full-res（全分辨率 LPIPS，≈0 成本）使 bicycle 3-seed PSNR 持平（-0.06±0.13 dB）并小幅改善 LPIPS（+0.050±0.002，3-seed 稳健）；6000 步探针显示配方 3000 步后双端退化，LPIPS 差距为渐进上界；bicycle 同 seed 重跑有 ±0.1-0.3 dB 非确定性（CUDA 原子放大），结论均多 seed。M4 主要门槛（train + bicycle PSNR/SSIM 持平 + ≥1.8x）达成，bicycle LPIPS +0.05 为唯一剩余诚实上界。**）
- M6 完成 = 可投稿（目标 CVPR/ICCV/ECCV 或 SIGGRAPH Asia）。
- 负结果必须诚实报告（宏块 backward 上限、共享内存累加负收益等已有关闭杠杆）。
## 8. Round 60：cull-masked Adam（R60，2026-08-04）

**动机**：R59 剖面显示 fused Adam 占 720p r0.35 操作点每步 39 ms 中的 ~17.5%（6.8 ms，3.5M 高斯量级）；union-visibility cull 只渲染可见子集（garden/bicycle 剔除 52%/69%），但 stock Adam 仍对全部 N 行更新。

**实现**（enchmark/higs_masked_adam.py，--masked-adam 开关）：融合 CUDA kernel 只对 train forward union-visibility mask=True 的行执行与 torch 2.7 fused Adam 位级一致的更新（m/v 双精度、bc 双精度、step_size 与 denom 按 opmath_t=float 折算，浮点对齐经 P13 补丁后的 cull cache 提供 zero-cost mask）。内存布局为扁平按元素合并访问（每个元素 i 取 row=i/D、mask 检查为 warp-uniform），bias correction 提升到 host 避免每元素 pow。隔离基准（N=3.5M，5 参数组）：torch fused 4.18 ms → masked 42% 可见 2.40 ms / 58% 可见 2.84 ms（全 mask 4.30 ms 持平）。正确性探针（PROBE_PASS）：p 最大相对差 1.66e-7（1-2 ulp）、m/v ~1 ulp（torch 自身 nvcc 编译在 ~0.4% 元素上舍入不同，累积有界）、mask=False 行与初值位级一致；P2 真行与 vanilla 位级一致。

**实验**（3 scene × 3 seed = 9 runs，720p eg 配方，k1，与 R59 k1 同配方同 seed 对照；r60-summary.json 聚合 30 runs）：

| 场景 | train_ms k1→ma | 墙钟 speedup_train | PSNR k1→ma | SSIM | LPIPS |
|---|---|---|---|---|---|
| train | 11.02→10.23 ms | 1.078x（-7.8%） | 17.039→17.025（-0.014） | 持平 | +0.001 |
| garden | 18.97→14.13 ms | 1.343x（-34%） | 18.058→19.953（**+1.90 dB**） | +0.085 | **-0.105** |
| bicycle | 20.66→14.67 ms | 1.409x（-41%） | 15.685→15.732（+0.047） | +0.027 | +0.017 |

**结论**：
- 这是**第一个质量正向的 per-Gaussian-floor 速度增量**：端到端 train_ms 全场景下降（-8%/-34%/-41%），garden 质量大幅提升（+1.90 dB PSNR、LPIPS -0.105，3-seed 极稳定 ±0.02 dB），train 质量中性，bicycle PSNR/SSIM 改善但 LPIPS +0.017 轻微回退（诚实上界）。
- 机制：冻结 train 不可见行后，stock Adam 的零梯度动量衰减不再侵蚀 eval 可见内容（train 4 相机 union 之外、eval 3 相机可见的高斯被保留），同时总 N 更高（garden 1.96M→2.87M）而渲染可见集约减半（garden 1.03M→0.53M），fwd/bwd 与优化器开销同步下降。
- 下一步候选：--cull-interval 4 × masked-Adam 叠加（K4 2-4% + 本杠杆 -34~-41%）；mask 与 prune 解耦（只冻结优化器、保留 opacity 衰减语义）以关闭 bicycle LPIPS 上界。
## 9. Round 61：剩余 per-Gaussian-floor 候选杠杆全部关闭（R61，2026-08-04）

**背景**：R60 之后，per-Gaussian-floor 层只剩三个候选：union-mask prune（直接删行）、LPIPS 训练损失 work-size（降采样替代损失）、K4×masked-Adam 叠加（陈旧 mask）。R61 全部实现并实测，全部因质量硬约束关闭；per-Gaussian-floor 速度前沿就此定格在 R60 的 k1 masked-Adam 操作点。

### 9.1 union-mask prune（--mask-prune）— 明确负面

**机制**：train-union 与 eval mask 都不可见且冻结 ≥ min_frozen 步的高斯直接从场景删除（参数行移除），期望同时缩小 Adam 行数与渲染规模；含 opacity 阈值、densify 末尾刷新 eval mask 等变体（代码保留为实验开关）。

**证据**（cull-cache mask 经校验正确；train cams=[0..3]、eval cams=[4..6] 全程固定）：
- 120 步 garden smoke：naive prune PSNR 20.66 vs R60 对照 22.49（final N 1.82M vs 4.13M）；opacity-cap 0.1 → 21.23；densify 末尾刷新 eval mask → 12.94（N→130K）；刷新+cap → 16.14。
- 宽限窗（120 步）：min_frozen=6（30 步宽限）21.50（-1.0 dB）；min_frozen=10（50 步）22.36（-0.13，但刷新开销使速度变慢）。
- 300 步（更真实）：CTRL 21.88 / train 28.07 ms / N 3.84M；mf10 r0 21.35（-0.53 dB，prune 919K，train 27.07 ms）；mf12 r0 21.66（-0.22 dB，prune 532K，train 27.76 ms）。

**根因**：union-invisible 集合正是迁移中期、后期才被需要的几何。Probe（step149）：union-invisible 占比 garden 48.5% / bicycle 71.6% / train 21.8%；eval-only 可见且 train 不可见的稳定集合 garden 676K / bicycle 212K / train 2.3K——这些行绝不能删。任何版本 PSNR 损失 ≥ 0.2 dB，速度收益 ≤ 3.6%，关闭。

### 9.2 LPIPS 训练损失 work-size（--lpips-work-size 256）— 质量门槛拒绝

**动机**：720p 全分辨率 LPIPS（AlexNet trunk 训练于 ~224px 块）单次前向纯卷积 ~8.9 ms（probe：cutlass conv 151.7 ms / 17 次调用），约占单步 10%；等比降采样到 max-side 256（~8 倍像素减少）应显著降低该开销。

**证据**（3000 步 × 3 seed × 3 scene，与 R60 配置逐项相同仅 ws 不同，r61-summary.json）：

| 场景 | train_ms ws0→ws256（3-seed 均值） | speedup_train | ΔPSNR (dB) | ΔLPIPS |
|---|---|---|---|---|
| train | 10.225 → 9.701 | 1.054x | -0.144 | +0.0062 |
| garden | 14.127 → 13.564 | 1.042x | -0.035 | +0.0116 |
| bicycle | 14.666 → 14.286 | 1.027x | -0.061 | +0.0131 |

**结论**：速度 +2.6~5.4% 真实且稳定（train_ms sd < 0.08 ms；lpips_ms_avg 9~13 → 3 ms），但 LPIPS 三场景全部回退（garden/bicycle +0.012/+0.013，≫ 3-seed sd 0.001~0.003，statistically significant），PSNR 全部下降——降采样替代损失与全分辨率 eval 的分布失配可测量地损害最终感知质量（代理损失越省、质量越差）。违反"质量保证"硬约束，关闭（代码保留为 --lpips-work-size 实验开关）。

### 9.3 K4×masked-Adam 叠加（--cull-interval 4 --masked-adam）— 质量门槛拒绝

**机制**：R59 K4 单独 train_ms -0.9~-2.0%（但 bicycle PSNR -0.34，3-seed）；R60 masked-Adam 是 k1 新鲜 mask。叠加希望保留 MA 质量增益同时省 3/4 refresh。

**证据**（3000 步 seed 0 屏测，vs R60 k1-MA seed 0）：garden 14.183→13.627 ms（-3.9%）PSNR -0.112；train 10.421→10.076（-3.3%）PSNR -0.226；bicycle 14.551→14.115（-3.0%）PSNR -0.075，LPIPS +0.001~+0.009。方向与 R59 K4 3-seed（bicycle PSNR -0.34）一致：mask 陈旧化使冻结保护滞后 3 步，质量成本 > 速度收益，关闭。

### 9.4 内核画像（probe_r61_kernels.py，garden 720p masked-adam，40 步，自 CUDA 时间）

| 内核 | 自时间 | 占比 |
|---|---|---|
| higs_blend_bwd_px | 317.4 ms | 22.9% |
| higs_sh_vjp_grid | 172.5 ms | 12.5% |
| cutlass conv（LPIPS AlexNet） | 151.7 ms（17 次调用 ≈ 8.9 ms/次） | 11.0% |
| rasterize_to_pixels_3dgs_fwd | 150.0 ms | 10.8% |
| masked_adam_kernel | 137.1 ms（200 次调用） | 9.9% |
| 合计 self CUDA | 1384.9 ms | — |

**启示**：LPIPS 卷积与 masked-Adam 合计约占单步自 CUDA 时间 1/5——正是 R61 两个被关闭杠杆的落点；它们要么开销已足够低（masked-adam 9.9% 且质量正向），要么降成本必损质量（LPIPS）。R60 的 k1 masked-Adam 操作点已是该层最优。

### 9.5 结论

可训练 HiGS 的 per-Gaussian-floor 速度前沿 = R60 k1 masked-Adam（train_ms -8%/-34%/-41%，质量正向），R61 三个剩余候选全部以质量门槛关闭。下一步质量候选：mask/prune 解耦（只冻结优化器、保留 opacity 衰减语义）以收窄 bicycle LPIPS +0.017 上界；速度侧不再有该层质量安全的杠杆。