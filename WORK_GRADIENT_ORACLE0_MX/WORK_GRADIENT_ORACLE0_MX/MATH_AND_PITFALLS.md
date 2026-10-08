# 数学问题、估计器和重要陷阱

## 1. 严格可证部分

在**同一个真实冻结状态、同一 train camera、同一可加 L1 loss**条件下，将每个 tile 的梯度贡献写作 `g_j = d L_j / d theta`，其中 `L_j` 已按照总像素数（包括边界实际有效像素数）做全图归一化。

独立 Bernoulli 采样：`z_j~Bernoulli(p_j)`，`p_j>0`。估计器 `g_hat = sum_j(z_j / p_j)*g_j` 无偏：`E[g_hat]=sum_j g_j`。

不同 tile 抽样独立时，按固定预处理线性映射 P 的梯度误差：

`E || P (g_hat-g) ||^2 = sum_j (1-p_j)/p_j * ||P g_j||^2`。

给定**仅为分析所用的可加代理成本** `c_j>0`, `sum_j p_j c_j <= B`，无其它约束时内点：

`p_j = clip( sqrt(v_j / (lambda*c_j)), p_min, 1)`, 其中 `v_j = ||P g_j||^2`，用 bisection 求 `lambda` 满足预算。

上述公式 **不是原创理论**，也不代表整个 GPU renderer 具有 `T = sum p_j*c_j` 成本结构。整个固定开销、intersection/sort 调度、warp occupancy 和相邻 tile 共享代价均非加性。

## 2. 梯度组与可比性

- 主指标：冻结定义的 geometry 梯度（means3D + quats/scales），写明每组维度、范数规范化、权重的精确定义；不得事后重调 group weights。
- 辅助指标：opacity、SH、densification-control 统计量；不能以 geometry 获胜掩盖 SH/opacity 严重退化。
- 统计应分别按 scene/state/camera 形成 cluster，不能把每个 tile 或 RNG mask 当作独立场景 seed。
- 解析 oracle 需要 `||g_j||` 的真实信息，收集它的巨大额外开销**必须计入数据采集**，不得以它作为实际可部署方法的 overhead。
- 如果只测了随机 64-tile candidate pool，则 oracle 只关于该 pool，不能外推 full frame，更不能称整个训练过程已加速。

## 3. 损失校正

对 `L = (1/(3*H*W*C)) * sum_{pixel/channel} abs(render-ref)`，由 train-view tiled masked rendering，`L_hat = (1/(3*H*W*C)) * sum_j(z_j/p_j) * sum_{valid pixel/channel in tile j} abs(render-ref)`。

- 边界 tile 权重遵从**实际像素数**。
- **不同 p_j 时直接求 selected pixel mean 不是无偏估计**。
- A/B 原有固定数量且非均匀 border 的 sampled mean 也可能有轻微偏差；必须实证和/或采用相同 HT 估计器重新测对照。
- 原仓库 `error_guided` 有放回抽样 `m/(k*p)`，不是 Bernoulli `1/p`，禁止混同。并且这种有放回方案的 distinct active tiles 可能少于名义 k，必须记录实际 unique active tile 数。
- DSSIM/SSIM/LPIPS 并非纯像素可加项；以上无偏理论不能用来保证它们的稀疏估计无偏。除非定义并审计明确的 sparse local objective，否则先将 L1 梯度机制 oracle 与 full recipe quality 分开验证。

## 4. 成本语义

`c_j` 可以是 per-tile Gaussian intersection/blend interaction proxy 或在专门固定测试中推测的 marginal cost；必须明确其定义、精度和计算开销。

- 需要测量实际 sampled-mask **集合**下的 F、B、F+B CUDA events 和同步 wall；至少 30 warmup + 100 measured × 3 paired repetitions；比较 U、E、P 的实际 selected-pixel 比例和 active intersections。
- 全图可共享的 projection/Jacobian/SH/optimizer 成本属于不可按 tile 加法分配的固定/交互部分。
- 重要性采样收益以相同**实际训练 wall/CUDA cost**和可加 L1 同状态梯度误差衡量；报告 MSE/NMSE/cosine；同时报告预期成本代理与实测时间排序的相关性。
- oracle/proxy 若对 F+B 无收益或 online policy 的调度成本 >5%，不得宣称提速。

## 5. 可能的失败原因

1. 真正重要的 tile 梯度相似，oracle 无优势。
2. 梯度差异很大，但廉价的历史代理无法预测真实梯度信息。
3. 重要 tile 恰好是成本最高的 tile，成本-信息收益不可优化。
4. 在线预测每个 tile 都需要跑完整 projection，管理代价吞掉 renderer savings。
5. 室内有效，garden/bicycle 高-N 无效，不能作为通用方法。
6. 低 NMSE 不自动意味着 30K PSNR/LPIPS 提升。

## 6. Prior-art novelty guard

- TurboGS (ICML 2026): https://proceedings.mlr.press/v306/dong26k.html
- RT-Loss (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Tile-wise_vs._Image-wise_Random-Tile_Loss_and_Training_Paradigm_for_Gaussian_ICCV_2025_paper.html
- 传统 cost-sensitive importance sampling 与优化器 variance reduction 是已知技术，不得宣称“首次推导 p∝||g||/sqrt(c)”。真正需要原创新意是 3DGS 可部署的 cheap signal + 非加性 renderer cost + 训练收益。
