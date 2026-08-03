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
## 5. 实验矩阵与状态（2026-08-03 Round 32 更新）
- M1 基线量化（已完成）：成本分解；per-pixel VJP 量随分辨率 / 相机数 / 高斯数缩放曲线（blend bwd 26-29 ms、6.23G evals 为不可约部分）。
- M2 原型（**已完成**）：`tile_sampling_ratio`（1.0 默认）+ `sampling_mode`（uniform/stratified）已进 native capture 路径；isect 按 tile mask 过滤、blend backward 随 r 近线性下降；**顺序（无争抢）测量** 5 场景 1080p×4cam×20 步：r=0.5 总时间 -15..-25%、r=0.25 -33..-41%、r=0.125 -43..-50%；bwd 在 r=0.5/0.25/0.125 约为全量的 64%/45%/35%（存在 ~5-6 ms 固定底：projection/SH VJP + 归零填充）。并发现/修复多相机 isect 过滤 bug（`sampled_ratio` 未除 C）。
- M3 采样策略消融（**部分**）：stratified vs uniform 已在 300 步协议 train/bicycle r=0.5/0.25 对比；stratified 在 train 显著更好（r=0.5 PSNR -0.02 vs -0.20 dB），bicycle 上 uniform 反而 PSNR 更高（+0.13 vs -0.11 dB，单 seed 属噪声级）；**Round 32 新增误差引导采样（error_guided，p 正比于 tile 误差^alpha + 无偏重要性加权损失）与 anchor-densify**：train 上 alpha=1.0 在 r=0.5/0.25 均反超全量 PSNR（+0.81/+0.84 dB，4/3 seed 全高于 full），bicycle 上为最差模式（-0.28..-0.49 dB）——场景相关，诚实标注；anchor-densify 收敛 train r=0.25 PSNR 差距（-0.67到-0.28 dB）但未恢复 LPIPS。
- M4 质量验证（**部分/负结果需诚实报告**）：300 步协议 train/bicycle 完成（frozen r=0.5 PSNR/SSIM 持平、LPIPS +0.02-0.04；r=0.25 PSNR -0.20/-0.57 dB 未持平；dynamic r=0.5 -0.21/-0.37 dB 未持平）；**Round 32 多 seed（0/1/2）验证 frozen r=0.5 持平结论成立，且所有 r<1 模式 LPIPS 均 +0.02..+0.08（唯一一致的负面指标）**；30k 收敛协议 + 多 seed 未做；1200 步 r=1.0 对照显示 train N 坍缩（354K）为协议固有而非采样引入。
- M5 扩展性：未做。
- M6 对照：未做（ICCV 2025 random-tile loss、Turbo-GS、Speedy-Splat 对照留待投稿阶段）。

## 6. 风险与对策
- 采样训练改变密度化动力学（梯度稀疏 -> densify 信号变化）：对策 = 梯度累积 / 密度化专用全分辨率步 / 调 densify 阈值。
- 短程（20 步）收益 != 长程收敛收益：一切以 M4 为准。
- 与 tile-wise training 的区分度：需要实验证明（质量保持 + 明确速度收益 + 宏块结构利用）。

## 7. 里程碑与验证标准（论文门槛）
- M2 完成 = 可演示 r=1/4 时总时间降 ~35-45%（若 blend 主导成立）；这是"明显加快"的第一实证。（**Round 31 已达成**：r=0.25 总时间 -33..-41%，bwd 近线性。）
- M4 完成 = 核心实验（收敛质量持平 + >= 1.8x wall-clock 加速）。（**未达成**：r=0.25 与 dynamic 均未持平，r=0.5 frozen 仅 PSNR/SSIM 持平；1.8x wall-clock 也未达到，因 fwd/culling 固定成本占比高。）
- M6 完成 = 可投稿（目标 CVPR/ICCV/ECCV 或 SIGGRAPH Asia）。
- 负结果必须诚实报告（宏块 backward 上限、共享内存累加负收益等已有关闭杠杆）。