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
- **Round 37 更新（2026-08-04，culling refresh-interval 缓存）**：新增 `--cull-interval N`（渲染器 `cull_refresh_interval`）：full-N union-visibility 投影结果按 renderer handle 缓存 N 步，任何拓扑变化（`mark_dirty()`/`rebuild`）立即失效，ci=1 即逐帧 cull（默认）。实现为 `_cull_visible_cached` + handle 上 `_fwd_count`/`_cull_visible_ids`/`_cull_fwd_count`，穿透 autograd forward、frozen/dynamic 两个 forward 与公共 wrapper，metadata 记录 `cull_refresh_interval`；新增 `TestCullCache` 3 项（节拍计数、densify 失效、静态参数帧级一致），HiGS 全量 44 通过、全仓库 272 通过/1 跳过。EPIC-05 原生后端、3000 步 R36 配方（LPIPS w=0.1 every 25 + lr-decay + densify-window 1500、error_guided r=0.5、4x1080p）：train ci=1→25 每步 18.04→17.66 ms（-2.1%）、fwd 7.08→6.75 ms（-4.7%），PSNR/SSIM/LPIPS 17.278/0.6281/0.3812 → 17.167/0.6334/0.3784（seed 0），3-seed ci=25 PSNR 均值 16.79——质量持平；ci=100 无进一步加速（17.52 ms）且 PSNR -0.36 dB（陈旧可见集滞后优化器漂移）；同会话 800 步 r=0.5 配对：ci=50 vs ci=1 -0.07 ms/步（噪声级）、PSNR -0.36；bicycle ci=25 35.00 vs R36b ci=1 35.64 ms（-1.8%），单 seed PSNR 种子噪声主导、无质量结论。**诚实结论：full-N cull 投影只占每步 ~0.3-0.5 ms（17-35 ms 步长中可见子集渲染 + LPIPS 摊薄占主导），缓存是真实但温和的杠杆——ci=25 端到端约 -2% 且质量持平，ci=50/100 无收益并因陈旧可见集损 PSNR；杠杆已作为安全默认关闭，1.8x 墙钟门槛仍需 forward tile sampling——M4 速度未达成。**
- M5 扩展性：未做。
- M6 对照：未做（ICCV 2025 random-tile loss、Turbo-GS、Speedy-Splat 对照留待投稿阶段）。

## 6. 风险与对策
- 采样训练改变密度化动力学（梯度稀疏 -> densify 信号变化）：对策 = 梯度累积 / 密度化专用全分辨率步 / 调 densify 阈值。
- 短程（20 步）收益 != 长程收敛收益：一切以 M4 为准。
- 与 tile-wise training 的区分度：需要实验证明（质量保持 + 明确速度收益 + 宏块结构利用）。

## 7. 里程碑与验证标准（论文门槛）
- M2 完成 = 可演示 r=1/4 时总时间降 ~35-45%（若 blend 主导成立）；这是"明显加快"的第一实证。（**Round 31 已达成**：r=0.25 总时间 -33..-41%，bwd 近线性。）
- M4 完成 = 核心实验（收敛质量持平 + >= 1.8x wall-clock 加速）。（**部分**：r=0.5 dynamic + LPIPS 正则化（w=0.1 every 25）在 train 3-seed 达成 PSNR/SSIM 反超（+0.64 dB/+0.009）、LPIPS 差距缩至噪声级（+0.0046±0.0063），bicycle 仍 +0.038 未关闭；r=0.25 未做；1.8x wall-clock 未达到——采样 forward 仍渲染全部 tile，r=0.5 每步仅 -27..-29%。）
- M6 完成 = 可投稿（目标 CVPR/ICCV/ECCV 或 SIGGRAPH Asia）。
- 负结果必须诚实报告（宏块 backward 上限、共享内存累加负收益等已有关闭杠杆）。