# P13-X — Variable-Projection Gaussian Splatting：对抗性审查

审查截止：2026-09-26。结论：**DROP**。

此结论针对目前提出的研究定位：以“固定几何下 RGB/SH 线性消元 + renderer-native matrix-free solver + 多速率缓存”作为新颖的 ≥2× 完整训练方向。它不等于证明任何完整 variable-projection 训练器都不可能有效。最直接的内层求解器已有论文和公开代码；剩余完整训练收益尚未验证。不能把“尚无论文同时使用这几个名称”视为组合创新。

## 0. 证据边界

| 类别 | 本次建立的内容 |
|---|---|
| **MEASURED** | 本机 CPU float64 合成矩阵反例与成本网格；公开源码逐项核对。没有运行 3DGS GPU 训练，没有场景条件数或真实 E2E speedup 测量。源码存在属于可核对事实，不等于结果复现。 |
| **LITERATURE** | 固定几何 SH 的 PCG refit、解析颜色拟合与全局残差修正、联合 LM、matrix-free 二阶求解、不同参数更新频率、训练块缓存均已有公开工作。下文速度均为作者报告，比较对象各不相同。 |
| **DERIVED** | 未截断 appearance 的仿射性；Gaussian 间正规矩阵耦合；精确内层最优下的 envelope gradient；缓存有效条件；成本阈值。 |
| **HYPOTHESIS** | 本文三种成本场景、warm-start 可减少求解步数、局部重叠预条件器是否有效、冻结窗口是否不伤收敛。不能标成现实预测。 |
| **UNKNOWN** | 当前 fused-SSIM 强基线的可替代耗时比例、SH3 全场景求解迭代数、可接受的 K、等质量完整训练加速、剩余系统设计是否有独立贡献。 |

检索覆盖 CVF 的 CVPR 2024/2025/2026、ICCV 2025 目录，ECVA ECCV 2024，以及 2024–2026 arXiv、作者项目页和公开 GitHub；对 SIGGRAPH / SIGGRAPH Asia、3DV / ECCV 2026 的相关候选核对论文或项目标注。关键词覆盖 analytical/closed-form color/SH、least squares、variable projection、alternating/BCD、LM/GN、matrix-free、CG/LSQR、freeze/multi-rate/cache、colorization、固定几何反演。并非所有 venue 的所有论文全文穷尽；尚未到来的 2026 年末论文不能据此排除。无关键词命中不证明没有先行工作。

检索记录、HTTP 状态和下载 SHA256 在本目录 `*_batch_results.json`；原网页/原始代码保存在 `sources/*.html`。自动提取的 `.txt` 仅便于读论文，代码以原始下载内容为准。早期尝试的不存在版本（如 3DGS-LM v3）不用于结论，使用其有效当前版本。

## 1. Prior-art audit

### 1.1 最直接碰撞：Observation-Gram Compaction

**Only What Was Seen: Observation-Gram Compaction of View-Dependent Appearance in 3D Gaussian Splatting**，arXiv，2026-09-24。

[论文](https://arxiv.org/abs/2609.28997) · [全文 §3.6](https://arxiv.org/html/2609.28997) · [固定版本 refit.py](https://github.com/moholo-founder/ogc-3dgs/blob/49ccae72e75eec9877354ed72074827531f7fd79/train/refit.py)

| 项目 | 审查结果 |
|---|---|
| WHAT THEY SOLVE | 主任务是 SH 压缩；§3.6 专门用训练图像修复剪枝后的 appearance。不能因标题是 compression 而漏掉。 |
| VARIABLES | 几何、opacity 固定，联合优化全模型每个 Gaussian 的 SH，degree 3 为 16×3 系数。 |
| LOSS | 多视图 photometric squared error；Levenberg 阻尼用于增量；预条件器另有稳定项。不是 L1+DSSIM。 |
| EXACT / APPROXIMATE | 固定活动集的算子保留全局耦合，但有限 PCG、阻尼与近似 clamp mask 不给出原 clamped objective 的精确最小值。所谓“closed-form repair”实际运行迭代求解。 |
| TRAINING / POST-HOC | POST-HOC refit；未建立交替几何更新的完整 variable-projection 训练结果。 |
| SPEEDUP | 作者报告中位 refit 时间 72 s；这不是完整训练 speedup，也不能单独推导相对当前强基线的倍数。 |
| IMPLEMENTATION | 有代码。每次 PCG 一轮训练视图 forward + backward；每 Gaussian 16×16 Gram block preconditioner；默认 8 次迭代、2 次 restart。 |
| OVERLAP | **固定几何 + 联合 SH + renderer-native matrix-free PCG + Gram 预条件 + clamp restarts，直接覆盖 P13-X 的内层核心组合。** |

源码审查补充：

- `op()` 用 signed SH 颜色、没有 SH 的 +0.5 offset，通过裸 rasterizer 和反向实现 JᵀJv。主调用使用黑背景；非零背景若直接进入该算子会破坏线性。
- `render_colors()` 绕过官方 renderer 最后的 [0,1] clamp；最终评估重新 clamp。因此求解与最终评估的前向语义并不完全一致。
- mask 是约 16 个采样相机上“某 Gaussian 的某通道是否全部为正”的乘积，冻结的是整组 Gaussian/channel；不是完整的 per-view/channel 精确活动集合。函数注释不能代替实现。
- 压缩用的 S¹ observation Gram 是图像误差上界/无相关误差近似；真实正规矩阵对角块需要 S²=Σp w²。refit 代码确实使用 `LAST_ACCUM[1]` 的 S²，不能把论文中两种 Gram 混为一谈。
- `LAST_ACCUM` 来自修改过的 rasterizer。此实现的存在不证明 stock renderer 不改 CUDA 即可直接得到真实 S² Gram。
- Adam 对照提供 `l1ssim` 和 `mse` 选项；只能取匹配损失和匹配 clamp 的对照解释求解收益。

### 1.2 Instant Colorization of Gaussian Splats

arXiv 2026-04；仓库标注 **CVPR’26 Findings**，不是据此认定 CVPR 主会论文。

[论文](https://arxiv.org/abs/2604.17155) · [代码](https://github.com/dlieber01/Instant-Colorization-of-Gaussian-Splats)

| 项目 | 审查结果 |
|---|---|
| WHAT THEY SOLVE | 把新 RGB 图像、特征或分割重新映射到已有 splats，用于 relighting/colorization/feature enrichment。 |
| VARIABLES | 固定位置、尺度、旋转、opacity，求 RGB 或 SH0–3/特征。 |
| LOSS | 初始求每 Gaussian visibility-weighted LS；颜色代码的 Adam 对照优化 MSE。多轮 rendered-residual refinement 处理 alpha 混色。 |
| EXACT / APPROXIMATE | 初始独立解不是全局 image-space LS。后续残差迭代不能忽略：在无 clamp、固定算子的条件下是保留全局残差的预条件迭代，收敛时可到相应正则 L2 解。有限轮只近似。 |
| TRAINING / POST-HOC | POST-HOC。 |
| SPEEDUP | 作者报告最高约一个数量级；特定 relighting 设置初始拟合约 1 s、梯度基线约 40–50 s。不能把初始拟合耗时、最终 refinement 耗时、完整训练耗时互换。 |
| IMPLEMENTATION | 公开 Python/rasterizer 实现；复用 visibility/SH 统计与局部矩阵；默认代码可跑 50 轮全相机 refinement。不是所有 raster pipeline 都已缓存。 |
| OVERLAP | **解析 SH/颜色、几何解耦、可复用统计、全局残差修正已存在。** 不能仅以“我们叫 CG，他们叫 recoloring”避开比较。 |

其局部矩阵为 Mᵢ=Σvp wᵥₚᵢYᵥᵢYᵥᵢᵀ，而真实 Hᵢᵢ=Σvp wᵥₚᵢ²YᵥᵢYᵥᵢᵀ。这是本质区别，但不构成否定其全局 refinement 的理由。其更新在相应假设下可写作：

\[
\beta^{t+1}=\beta^t+(M+\Lambda)^{-1}
\{A^T(y-A\beta^t)-\Lambda\beta^t\}.
\]

若 w≥0 且每像素 Σw≤1，令 zᵢ=Yᵢᵀδβᵢ，则 (Σwᵢzᵢ)²≤Σwᵢzᵢ²。因此 **M≽AᵀA**。这是一个可用的 block majorizer；在正则化/可观测子空间等条件下，迭代能求全局正则 L2，而非永远局限于独立 Gaussian。此推导属于本次 DERIVED，论文的有限步结果仍是 APPROXIMATE。

### 1.3 其他最接近工作

下表中的 EXACT 是所列目标/子问题意义的精确性，不是“作者找到全局最优”的通用保证。

| 工作与来源 | WHAT / VARIABLES / LOSS | EXACTNESS；阶段 | SPEEDUP 与 IMPLEMENTATION | 与 P13-X 的交集及边界 |
|---|---|---|---|---|
| [3DGS-LM](https://arxiv.org/abs/2409.12892)，arXiv 2024、ICCV 2025；[代码](https://github.com/lukasHoel/3DGS-LM) | 联合 xyz/scale/rotation/opacity/SH 的 LM；将 L1+SSIM 写成平方根 residual。 | 目标值可保持；GN/LM 曲率近似，§3.3 忽略 SSIM 相邻像素导数，视图子集更新平均也近似。先 Adam 增密，再二阶阶段。 | 作者表中 Mip360 1271→972 s、T&T 736→663 s、DeepBlending 1222→951 s，约 20% 时间缩减；平均内存约 53 GB，对照 6–11 GB。自定义 CUDA `buildCache/applyJ/applyJT/diagJTJ`，PCG/Jacobi。 | 二阶、Jacobian 因子缓存、全局矩阵作用已有；不是 appearance 精确消元，也不是 2×。 |
| [Matrix-free Second-order Optimization… / LM-RS](https://arxiv.org/abs/2504.12905)，[3DV 2026 项目](https://vcai.mpi-inf.mpg.de/projects/LM-RS/)；[代码](https://github.com/hamzapehlivan/lm-rs) | 联合几何/opacity/RGB；主要 MSE，补充实验 MSE+0.2SSIM；**SH0、不开增密**。 | LM+PCG、dual-number/JVP、残差与视图采样，近似全批问题；SSIM 对角近似。迭代重建。 | 小场景约 5×，中等场景总体约 1.3× 对 Adam；中等 Mip360 indoor 252.5→190.25 s。约 2× 对 3DGS-LM、3.5× 更省内存；无采样朴素版比最终版慢约 4×。公开自定义 CUDA。 | **matrix-free CG 与系统瓶颈已研究；§4.2 明说 JᵀJ 不具对角/块对角占优。** SH3 全训练收益不能从该文外推。 |
| [3DGS²: Near Second-order Converging 3D Gaussian Splatting](https://arxiv.org/abs/2501.13975)，arXiv 2025，venue 未独立核实 | 逐 Gaussian/逐属性局部 Newton，先几何后颜色，附加邻视图目标；**L2+SSIM**。 | 显式指出位置固定后 SH 线性、颜色二阶渲染导数为零；局部 Hessian/Jacobi 式并行忽略跨 Gaussian 耦合，非全局精确 LS。训练方法。 | 作者报告 5.1/5.3/6.4× 对 vanilla，迭代减少 >10×；文中 C++/CUDA executable，未验证官方完整源码。第三方 reproduction 仓库仍写 in development。 | 颜色线性与分块交替二阶已明确出现；不能用 L2+SSIM 的结果证明 L1+SSIM 同语义收益。 |
| [TurboGS](https://arxiv.org/abs/2606.15924)，arXiv 2026-06；[正文及补充](https://arxiv.org/html/2606.15924v1) | 稀疏像素/视图采样、增密剪枝、每Gaussian局部Moment-LM；高阶SH单独Adam。损失为Charbonnier L1与tile-level sparse NCC，非原fusedSSIM。 | 局部Jacobian/曲率、忽略跨SH参数相关性、独立Cholesky系统，是APPROXIMATE hybrid FULL TRAINING。 | 作者报告约100 s、最高约10×对vanilla，含采样/容量/损失等系统改变；未核实公开可运行源码。 | **补充材料明确说SH用Adam是为了避免联合LM增大Jacobian与内存；直接提供工程取舍理由。** 非appearance消元，不能拿整个系统倍数证明P13-X。 |
| [FlashSplat](https://arxiv.org/abs/2409.08270)，ECCV 2024；[代码](https://github.com/florinshen/FlashSplat) | 固定 splats 的二元/分割标签，以 alpha 合成后标签拟合 0/1 mask，L1。 | 在目标为 0/1、输出在 [0,1] 的特殊条件下 L1 化为线性目标，权重投票可给相应精确解；POST-HOC。 | 作者报告 <30 s、约 50× 分割阶段。公开实现。 | **2024 已利用 alpha 线性；特殊二元 L1 不等价于任意 RGB L1/SH LS。** |
| [Splat Feature Solver](https://arxiv.org/abs/2508.12216)，arXiv 2025；[代码](https://github.com/saliteta/splat-distiller) | 固定 splats 上的 AX=B 特征提升，visibility 加权、正则及后处理。 | 实现是 row-sum/Jensen surrogate，非精确全局正规矩阵解；某些 guidance 改变 opacity 激活以增强主导性。POST-HOC。 | 作者报告分钟量级特征阶段，训练几何/提特征耗时不包含；未核实可用于完整训练的相对倍数。有代码。 | 大规模线性 feature lifting 已有；高维特征不自动等同 view-dependent SH；“closed-form”最优性的范围需审查。 |
| [Predictive Photometric Uncertainty in Gaussian Splatting for NVS](https://arxiv.org/abs/2603.22786)，2026，论文标注 ECCV26；[项目](https://chumsy0725.github.io/3DGS-Uncertainty/) | 固定几何的线性 uncertainty channel；SH 可吸收进 A，Bayesian 正则 LS。 | 显式推导线性系统，但实际仍用 SGD，理由是巨大正规系统难以直接处理。POST-HOC uncertainty；不是 RGB 重建。 | 非训练加速论文，speedup N/A；本次核对全文/项目，未验证可运行代码。 | **直接反例：知道线性仍选梯度优化，并非不知道线性。** |
| [Reparametrizing 3DGS for Real-Time Palette-based Color and Luminance Editing](https://arxiv.org/abs/2609.03897)，2026-09，论文标注 SIGGRAPH Asia 2026；[仓库](https://github.com/tedchao/ReparamGS-Palette) | palette/权重 SH 重参数化与微调；交互时求少量 palette/tone-curve 变量，几何与权重固定。 | IRLS、阻尼 BCD，有限迭代近似；POST-HOC editing。 | 作者报告 5–15 outer iterations、编辑几十 ms；仓库存在，未核实完整可执行实现。 | 固定几何、解析/交替 appearance 操作已有，但小 palette solve 不等于全体 Gaussian SH。会议尚未举行，勿写成已发表会议实录。 |
| [ABCD: Alpha-Composited Block Coordinate Descent](https://arxiv.org/abs/2608.27735)，arXiv 2026 | 空间块训练：inactive 块几何与颜色都冻结，预渲染 foreground/background RGBA；active 块梯度更新。 | 正文称 reconstruction loss，具体损失形式/权重未核实；不交错深度/适当分割下组合可精确，任意交错不可无条件压成两层。FULL TRAINING，内存导向。 | 作者小场景示例 83.5→1061.2 s，显著减速；源码未核实。 | **训练缓存已有；省内存/少活动参数不等于加速。** 不等于只更新 appearance 的全场景线性求解。 |
| [FastGS](https://arxiv.org/abs/2511.04283)，CVPR 2026；[调度源码](https://github.com/fastgs/FastGS/blob/44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f/scene/gaussian_model.py#L225) | 常规 Adam，几何/opacity/DC 与高阶 SH 分组；L1+fused SSIM，含增密/剪枝/compact box。 | 前 15k：几何/opacity/DC 每步，高阶 SH 每16步；15k–20k 两组每32步，之后每64步；累积梯度，仍渲染/反向。FULL TRAINING。 | README 报 3.32× 对 DashGaussian/Mip360、15.45× 对 vanilla/DeepBlending；为整个系统收益，不可归因于调度。公开源码已核对。 | **multi-rate 参数更新不是新概念**；其更新方向与“appearance 快、geometry 慢”不完全相同，无证据实现 P13-X 的 LS 消元/缓存。 |

另外筛查了 [GSta](https://arxiv.org/abs/2504.06716)（基于位置/颜色梯度选择性冻结；项目标 ICASSP2026，code coming soon；组合 TrickGS 的最高 5× 不等于冻结独立收益）与 [Group Training](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Faster_and_Better_3D_Splatting_via_Group_Training_ICCV_2025_paper.html)（ICCV2025，摘要报告最高约 30% 收敛加速，非 LS 消元）。Analytic-Splatting 的 analytic 指抗锯齿积分，相关 split-algorithm 指 primitive splitting；没有仅凭名称归类。

**未核实线索**：OGC 引用 Zhou/Liu 关于 structured spherical-harmonic optimization 的 SPIE 工作，本次未得到可靠主文；`rogerhh/analytical-gaussian-splatting` 仓库标题不足以验证算法。列为 UNKNOWN，不计入决定性证据。

### 1.4 对特别命题的裁决

> “传统 3DGS 仍然用 Adam 优化 appearance，是因为社区还没有充分利用固定 geometry 下 appearance 子问题的线性结构。”

**这个因果叙述缺乏证据；其中‘尚未利用线性结构’已被公开工作反驳。** OGC 的 matrix-free PCG、Instant Colorization 的解析 SH 与全局残差迭代是直接证据；uncertainty 工作还明确在推导线性系统后选择 SGD。

TurboGS补充材料进一步给出一个可引用的具体选择理由：高阶SH继续用Adam，以避免扩大全局/联合参数求解时的局部LM系统、Jacobian与内存开销；作者把忽略这些交叉相关性描述为效率和精度的取舍。它并未证明appearance-only PCG一定不划算，但足以反驳“保留Adam只能解释为社区忽视了结构”。该线索来自现有共享项目记录，本次已独立下载原论文并核对相关原文；共享任务的其他CPU实验未作为本报告独立复现实测。

更有证据约束的描述是：线性结构已被用于固定场景重着色、特征提升、压缩修复和二阶优化；把它转为含 clamp、L1+SSIM、增密、持续变化几何的低成本完整训练器仍是另一问题。工程成本、条件数、目标语义和随机小批次可能解释 Adam 的实用性，但不能未经对照实验声称这些是整个社区选择 Adam 的唯一原因。

## 2. Mathematical audit：严格线性究竟在哪里

设视图 v、像素 p、高斯 i、SH 基数 q=(l+1)²，通道 c。固定相机与所有影响 alpha 的参数，包含 opacity；记实际执行 renderer 的有效 contributor 集合与顺序为 Sᵥₚ：

\[
T_{vpi}=\prod_{j\prec_{vp}i}(1-\alpha_{vpj}),\quad
w_{vpi}=T_{vpi}\alpha_{vpi},\qquad
C_{vpc}=\sum_{i\in S_{vp}}w_{vpi}c_{vic}+T_{vp,bg}C_{vp,bg,c}.
\]

无 appearance clamp 时，cᵥᵢc=ΣkYk(dᵥᵢ)βᵢkc+aᵥᵢc（例如 SH 路径的固定 +0.5）。定义

\[
A_{(vp),(ik)}=w_{vpi}Y_k(d_{vi}),\qquad
b_{vpc}=\sum_iw_{vpi}a_{vic}+T_{vp,bg}C_{vp,bg,c}.
\]

于是 **C=Aβ+b**，这里 β 有 qG 行、3 列，A 可共享于 RGB 三个右端项。向量化时是相应的 Kronecker 三通道算子。b 只在 fixed-geometry 子问题中是常量；做几何梯度时必须保留 ∂b/∂θ。

| 检查项 | 结论 |
|---|---|
| alpha / opacity | α=oᵢexp(−½δᵀQδ) 经 cap/cutoff，仍独立于颜色。oᵢ 必须冻结；只冻结 xyz/scale/rotation 不够。固定 opacity logit 同样固定 sigmoid(o)。 |
| view-dependent SH | dᵥᵢ依赖相机和位置，冻结后 Y 固定；view dependence 不破坏线性。若采用逐像素方向，改为 Y(dᵥₚᵢ)，仍线性但缓存更大。 |
| 逐高斯 clamp | Graphdeco/gsplat 的 max(SH+0.5,0) 使系数→图像全局分段仿射。几何固定不代表颜色活动集固定。 |
| 最终 image clamp | Graphdeco 当前 Python renderer 还有 clamp(0,1)；跨边界时不仿射。局部固定 mask 后可以表示 Jβ+b′，不能以同一个 A 覆盖所有 β。 |
| background | 固定背景进入 b；随机背景每次变则 b 变。可学习线性背景可增广未知量，但如果与其它可学习项相乘就要重新分析。 |
| early termination | 标准按 T 阈值停止，与颜色无关。固定 geometry/opacity 时 stop index 固定；影响的是 A 的定义，不使它对 β 非线性。若其它 renderer 使用颜色贡献阈值，则要另审。 |
| AA / filtering | 固定几何的 covariance filter、opacity compensation、固定 shutter/采样均可吸收到 w。随机采样/动态 LOD/参数更新导致权重变化，不能继续用旧 A。 |
| packed representation | packed IDs/gather 只是固定索引与存储组织，不会凭空造成非线性。动态 mask、LOD、prune 后索引失效。 |
| depth sorting / visibility | 排序与剔除由 geometry/opacity/camera 决定，固定时可包含在 A。geometry 更新跨过排序/剔除阈值时 θ→A 分段不光滑。 |
| multiple RGB channels | 线性 RGB 下共用 A、可同时求3个 RHS；固定 channel-mixing exposure 仍是更大的线性算子。常见逐通道 SSIM 本身也可分通道，但空间像素不独立。 |
| exposure / embeddings | 固定仿射 exposure EᵥC+tᵥ 仍仿射；联合求 E 与 β 为双线性。非线性 tone map/MLP appearance embedding/BRDF-visibility-lighting 联合反演通常非线性；固定 embedding 也不保证 decoder 对 β 线性。 |
| distortion / depth loss | 若只依赖 depth、alpha/T，冻结内层时为常数，仍影响外层几何；不能从完整训练成本删除。若 loss 的权重依赖 RGB，则要计入 appearance 目标。 |
| regularizers | 固定 L2/SH-band 正则给增广 LS；L1/group/TV 正则仍可定义凸非二次问题；非线性 appearance 正则不再是 LS。geometry-only 正则不影响 β*，但影响外层。新加 ridge 是改目标，除非 baseline 也使用相同项。 |

核对来源：[Graphdeco forward](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu)、[Graphdeco renderer](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py)、[gsplat v1.5.3](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py)。原始代码快照已保存；上游 main 链接可能以后变化。Graphdeco 的 alpha cap 0.99、低 alpha cutoff 1/255、T cutoff 1e-4 都是颜色无关的分支。

**因此应把核心假设改写为严格限定的事实，而不是待发明的新理论：标准固定几何/opacity/camera 的未截断线性 RGB/SH renderer 是仿射映射；现有常用 clamp 使其分段仿射。**

## 3. Global coupling、条件数与最坏场景

令 y=I−b，L2 目标为 ½‖Aβ−y‖²，则

\[
H_{ij}=(A^TA)_{ij}=
\sum_{v,p} w_{vpi}w_{vpj}Y(d_{vi})Y(d_{vj})^T.
\]

只要 i、j 对同一像素都有非零贡献，就有潜在非零 off-diagonal block。没有一般性的独立 Gaussian 解。opaque front surfaces 有时降低重叠，但这不是 block diagonal dominance 定理，尤其不能从排序顺序推出。

- **overlap graph**：每个像素的 contributor 集合诱导 clique，多视图取并集。它不是简单的 3D kNN 图；沿深度较远的 splats 也能共现，多个视角还能连接不同区域。
- **visibility graph**：camera/pixel 与 Gaussian 是二部观测图。重叠少、每个 Gaussian 被多方向独立观察时较有利；隐藏 splats 是零列。只在少量视图中看到的 SH block 可能低秩。
- 同一相机中标准 SH 的 Y(dᵥᵢ) 对一个 Gaussian 的所有像素相同，所以其 Hᵢᵢ 的 rank 至多为有效观察方向数，不能靠一个相机的百万像素自动恢复 SH3 的16个方向维度。
- SH 在完整球面的正交性不等于在窄相机轨迹上的正交性。高阶 SH、低 opacity、弱可见 splats、重复/近重合 splats会产生弱方向及近共线列。
- 更多视图可能改善秩，也增加每次 A/Aᵀ 的成本；同方向重复图像未必增加独立信息。

**block Jacobi 有两种完全不同的用途：**一次丢掉 Hᵢⱼ 并独立求解，是近似问题；把 block diagonal 当全局 CG 的预条件器、每次仍算全局 AᵀAv，则保留全局耦合。后者正确，不代表迭代快。加大屏幕重叠块可改善跨 Gaussian 方向，却增加建块、存储、重建与分解成本。

| 场景 | 不利机制 |
|---|---|
| dense foliage | 大量半透明细枝/叶片重叠，近似可替代贡献，视角变化形成复杂联合图；可能只约束总颜色而不约束逐 Gaussian 分配。 |
| large transparent overlap | 一条 ray 很多弱贡献，列高度相关、尺度差异大，跨 Gaussian 预条件需求最强。 |
| indoor | 窄轨迹/窄视角、遮挡和重复平面纹理；高阶方向信息不足，隐变量可过拟合训练视图。 |
| high-order SH | q=1/4/16，未知量与 Gram O(q²) 增长；窄锥观察导致局部 Gram 近奇异。 |
| many views | 对 rank 可能有帮助，但全视图 sweep 吞噬加速；更大 union overlap 图、缓存随相机数增长。 |

CG 需要 SPD；未观测方向存在时，ridge、固定 nullspace 或适合秩亏的求解方法需明确。对 SPD，标准误差界含 (√κ−1)/(√κ+1) 的 m 次幂。LSQR 不显式存 AᵀA，数值上通常更适合 LS，但不会把谱病态、不可辨识性和全视图访问成本自动消除。正常方程 κ(AᵀA)=κ(A)²；不要把这句话误解为任何两列问题都必然要几百次 CG。

**本次 CPU 反例（非物理场景性能实测）：**

1. A=0.98[[0.8,0.2],[0.2,0.8]]、y=A[1,0]ᵀ。全局解 [1,0] 残差0；visibility row-sum 一次拟合得 [0.6664,0.3136]、残差范数0.26945。非负、每行和<1，符合 alpha-compositing 的代数权重条件；未声称构造了实际多视图3D Gaussian几何。
2. A=0.98[[0.501,0.499],[0.499,0.501]]，κ(A)=500，κ(AᵀA)=250000；Jacobi 迭代谱半径约0.999992。此2×2问题 CG 在精确算术至多2步，说明**条件数本身不能预测实际步数**。
3. 128组不同近共线2×2块组成的256维算例：对角PCG在32步的 normal relative residual 为1.21e−4，image relative residual 为2.14e−3。更严苛1e−10 normal tolerance 在浮点算例中用839步，不作为真实训练预测；图像2.14e−3是否已经足够应由质量门决定。

脚本：`oracles.py`；原始结果：`oracle_results.json`。它们否定“任何 block solve 都足够”的一般保证，不能证明真实场景 CG 必慢。

## 4. Loss semantics：不能把线性前向偷换成二次目标

当前审查对象保持项目强基线的

\[
F(\theta,\beta)=0.8\,\mathrm{mean}|R-I|+0.2(1-\operatorname{SSIM}(R,I)),
\]

以及相同 clamp、11×11/sigma1.5 窗口、padding/reduction、色彩域与附加正则。fused 只改变实现，SSIM 仍依赖局部均值、方差和协方差，仍为非二次且通常非凸。像素邻域重叠使其导数/曲率在像素之间耦合。各仓库 DSSIM 的因子约定可能不同，必须核对权重后比较。

| 方案 | EXACT / APPROXIMATE / OBJECTIVE-CHANGING |
|---|---|
| 无 clamp、匹配 L2、完整 A，LSQR/CG 收敛到容差 | **EXACT formulation**；数值解是给定容差近似。加上 baseline 同样拥有的 quadratic regularizer 仍为 LS。 |
| L1 用 r=Aβ−y 分裂，ADMM/primal-dual、soft-threshold | **EXACT formulation**，在凸仿射条件与收敛假设下可解原L1；有限迭代是 APPROXIMATE optimizer。内层LS不是整个L1问题的一步精确解。 |
| IRLS，权重1/√(r²+ε²) | 每轮 WLS 是 **APPROXIMATE**；固定ε通常优化平滑L1，严格讲对原L1 **OBJECTIVE-CHANGING**。ε→0及充分收敛需要另外论证。 |
| L1+SSIM 的真实梯度 + PSD local quadratic / proximal / trust region | **APPROXIMATE optimizer**，若每步用原loss校验/line search，仍针对原目标；没有内层全局最优保证。 |
| 将非负L1/SSIM损失项开方作为 residual，再GN/LM | 在定义有效的地方目标值可 **EXACT** 表示；GN忽略二阶残差项，仍是近似曲率。零残差/绝对值处要处理非光滑/数值奇异。完整窗口Jacobian不可任意删除。 |
| 冻结 SSIM statistics / 只保留中心像素或对角 Hessian | **APPROXIMATE** 局部模型；是否保持训练目标取决于是否用真实loss的梯度与接受准则。3DGS-LM/LM-RS 中的近似不能写成精确SSIM求解。 |
| L2 refit 与geometry的L1+SSIM更新交替 | 若直接替换β并不校验原loss，**OBJECTIVE-CHANGING / mixed-objective algorithm**；不是该原目标的β*。 |
| 新加 SH damping / ridge /颜色约束 | 若成为永久惩罚而baseline没有，**OBJECTIVE-CHANGING**；若只限制增量且最终收敛到原驻点，则是优化器设计，需区分。 |

保留原 objective 不代表每步必须求精确β*；但若只是能使原loss下降的几轮近似迭代，名称应是 inexact alternating optimization，不能凭 envelope theorem 把它描述成完成了 appearance 消元。

## 5. 真正的 variable-projection derivative

令内外共享同一标量目标 F，β*(θ) 为局部唯一、光滑且满足一阶条件的内层极小点：

\[
f(\theta)=F(\theta,\beta^*(\theta)),\qquad
\nabla_\beta F(\theta,\beta^*)=0.
\]

链式法则给

\[
\frac{df}{d\theta}=F_\theta+F_\beta\frac{d\beta^*}{d\theta}
=\boxed{F_\theta(\theta,\beta^*)}.
\]

**对这个最优值的一阶梯度，不需要显式计算 dβ*/dθ。** 可以在精确内解处 detach β* 后，对真实 renderer 和所有几何项做偏导。不能顺便 detach projection/alpha/b，后者会删掉所需的 Fθ。

若 Hββ 可逆，隐函数导数与约化 Hessian 为：

\[
\beta^*_\theta=-F_{\beta\beta}^{-1}F_{\beta\theta},\qquad
\nabla^2f=F_{\theta\theta}-F_{\theta\beta}F_{\beta\beta}^{-1}F_{\beta\theta}.
\]

因此 **约化 residual Jacobian、精确二阶几何更新、或别的 outer loss** 仍可能需要内解敏感度，不能把一阶 envelope 的结论推广为所有导数都免费。

最直观的L2证明：r=A(θ)β*+b(θ)−I、Aᵀr=0，则

\[
df=r^T[(dA)\beta^*+db]+r^TA\,d\beta^*
=r^T[(dA)\beta^*+db].
\]

有正则时需把正则也放进同一个F，使用完整最优性条件消项，不能只用Aᵀr=0。

成立条件及失败点：

1. **同一目标、数据和正则**。L2内层+L1SSIM外层、训练子集内层+全视图外层、额外内层ridge，一般 Fβ≠0；不能消去链式项后称为准确的外层梯度。
2. **内层足够最优**。只跑1–8轮CG/IRLS通常只得 β̃；Fθ(θ,β̃) 可作为交替优化方向，但不是已证明的 ∇f。梯度误差可能被小特征值放大；“normal residual 小”也不自动等于参数/外层梯度误差小。
3. **局部光滑与稳定分支**。rank改变、clamp活动集切换、visibility/sort/cutoff切换、非唯一解时经典光滑定理不直接适用。固定rank伪逆和非唯一情形有更细条件；不能一概说 envelope 必失效，也不能一概声称唯一普通梯度存在。
4. **L1非光滑**。需采用 Danskin/方向导数/广义梯度及相应最优性条件；不能把 Fβ=0 的光滑证明直接套到任意 kink。
5. **约束依赖θ**。固定可行域、稳定KKT条件下可用约束 envelope；若 g(θ,β)≤0，正确式包含乘子项 λᵀgθ。由视角决定的颜色非负约束就是可能的θ依赖约束。忽略它会错。
6. **非凸SSIM内层**。局部极小分支可做局部 envelope 推导，不等于得到了全局 argmin 的最优值函数。

这一理论属于经典 separable nonlinear least squares / variable projection；正确应用是必要条件，不构成3DGS里的新定理。

## 6. Geometry freeze / cache audit

表中 **SAFE TO CACHE** 均以同一相机/分辨率/renderer设置、xyz/scale/rotation/**opacity**/拓扑均固定、无随机采样变化为前提。换相机必须使用对应相机缓存；同一几何不意味着所有相机共用屏幕缓存。

| 状态 | 标记 | 理由/失效条件 |
|---|---|---|
| projection / projected mean | SAFE TO CACHE | 仅几何、相机、投影模型决定。 |
| conic / projected covariance | SAFE TO CACHE | 包括固定AA filter；scale/rotation/相机变化失效。 |
| radius | SAFE TO CACHE | 相同裁剪与radius计算规则。 |
| tile overlap | SAFE TO CACHE | 支持域、分辨率、tile规格与opacity cutoff固定。 |
| duplication / packed IDs | SAFE TO CACHE | 同一可见集、Gaussian索引和tile成员关系。 |
| depth key | SAFE TO CACHE | 深度和排序键规则固定。 |
| sorting | SAFE TO CACHE | color/SH变化不会改变标准geometry-derived顺序。 |
| tile range / offsets | SAFE TO CACHE | 沿用完全相同的排序结果。 |
| visibility / culling | SAFE TO CACHE | 几何可见性与alpha cutoff；不要把“颜色clamp mask”也叫visibility后混用。 |
| per-pixel alpha | CONDITIONALLY CACHEABLE | 数值对appearance不变；但需要O(pixel–splat interactions)存储，stock API未必输出，完整缓存可能比重算昂贵。 |
| transmittance / weights | CONDITIONALLY CACHEABLE | 固定alpha和顺序即固定；同样面临每交互内存/带宽问题，不能只计省掉的乘法。 |
| early termination / last contributor | SAFE TO CACHE | 对标准T阈值停止是固定；若按颜色误差自适应则重审。stock backward是否支持外部重用另论。 |
| SH basis | SAFE TO CACHE | 同一相机与xyz。每视图每Gaussian需q值，很多相机可能不值得常驻；重新算也可能更便宜。 |
| per-Gaussian/view/channel clamp mask | MUST RECOMPUTE | appearance变化可跨过0；临时固定仅是一次local active-set模型。 |
| colors / rendered RGB / final clamp mask | MUST RECOMPUTE | 正是未知量改变的部分。 |
| residual / L1 weights / SSIM stats & derivatives | MUST RECOMPUTE | 依赖当前RGB；冻结它们等于局部近似。 |
| exposure-transformed target/bias | CONDITIONALLY CACHEABLE | exposure和背景固定才可复用；若曝光也优化需更新。 |
| geometry backward graph | MUST RECOMPUTE（返回geometry更新时） | detached数值缓存不携带几何导数。可用定制解析导数替代重建，但本次原型不新增CUDA。 |

拓扑增密/剪枝、opacity reset、相机姿态训练、resolution schedule 会使对应缓存失效。冻结K步也改变densification梯度的采样与统计；必须明确几何梯度是每窗口算一次、还是累计多个相机。后者不能按一次step计成本。

内存数量级例子（DERIVED）：百万Gaussian、SH3的一个float32 16×16矩阵约1.024 GB；同时保存Gram和逆约2.048 GB，尚无PCG向量。百万Gaussian的一个RGB-SH向量约192 MB，5个工作向量约960 MB。单相机2MP、平均每像素64贡献，仅一个float32权重数组约512 MB；8个相机约4.096 GB，不含索引。缓存可行性不能只看projection/tile数组。

## 7. Amdahl oracle：108种组合，全部是假设模型

用户公式中的各成本应当**相加**：

\[
T_{new}=T_{untouched}+N_gT_g+M T_{solve}+T_{refresh}.
\]

令基线 N 个完整单视图step、每step时间归一化为1，Tstandard=N。设 u 是未被替代的基线总耗时比例（如早期增密/必需几何阶段，必须实测），其余(1−u)N步由冻结窗口替代。K 是试图替代的基线step数；η是达到同等质量需要的窗口膨胀因子。令

\[
N_g=M=\eta(1-u)N/K,
\quad T_{solve}=B(s_0+mp),\quad
T_{refresh}=MBC,
\]

则

\[
S(K,m)=\frac{T_{standard}}{T_{new}}
=\left[u+\frac{\eta(1-u)}K\{G+B(C+s_0+mp)\}\right]^{-1}.
\]

B为一次solve的相机数；G为geometry step（此处与cache refresh分开计）；C为每相机建缓存成本；s₀为每相机分摊的RHS/预条件/真实loss验证成本；p为每相机一次forward+adjoint pair；m为全批solver迭代数。所有成本与**当前强基线的完整单视图step**同单位。多个IRLS/active-set/SSIM outer rounds、重复线搜索需追加成本，不能只数最后一轮CG。

| 场景 | u | η | B | G | C | s₀ | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| pessimistic | .65 | 2 | 32 | .90 | .15 | .30 | .70 |
| realistic（待校准的中间假设） | .50 | 1.5 | 8 | .75 | .10 | .20 | .40 |
| optimistic | .10 | 1 | 4 | .55 | .04 | .10 | .12 |

**没有测量能证明中间行就是真实场景。** 这些是假设输入，不是把旧renderer profile套到fused-SSIM强基线上。尤其 u=.5 已决定任何非零成本方案的上限小于2×，不能把人为设定u后的失败当成物理不可能证明。下面给出完整敏感性网格，使这一点可复算。

Pessimistic E2E speedup：

| K / m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | .072 | .046 | .027 | .015 | .008 | .004 |
| 4 | .138 | .090 | .053 | .029 | .015 | .008 |
| 8 | .253 | .169 | .102 | .057 | .030 | .015 |
| 16 | .435 | .305 | .191 | .109 | .059 | .031 |
| 32 | .678 | .509 | .340 | .204 | .113 | .060 |
| 64 | .941 | .765 | .556 | .360 | .211 | .116 |

Realistic-assumption E2E speedup：

| K / m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | .347 | .245 | .154 | .089 | .048 | .025 |
| 4 | .591 | .437 | .286 | .170 | .094 | .049 |
| 8 | .913 | .717 | .501 | .313 | .179 | .096 |
| 16 | 1.254 | 1.055 | .802 | .541 | .328 | .184 |
| 32 | 1.541 | 1.382 | 1.144 | .852 | .564 | .336 |
| 64 | 1.741 | 1.634 | 1.456 | 1.195 | .880 | .576 |

Optimistic E2E speedup：

| K / m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1.226 | .969 | .683 | .430 | .247 | .133 |
| 4 | 2.185 | 1.768 | 1.279 | .824 | .481 | .263 |
| 8 | 3.586 | 3.004 | 2.268 | 1.522 | .918 | .512 |
| 16 | 5.279 | 4.620 | 3.698 | 2.642 | 1.682 | .974 |
| 32 | 6.910 | 6.320 | 5.399 | 4.180 | 2.880 | 1.776 |
| 64 | 8.173 | 7.745 | 7.012 | 5.896 | 4.472 | 3.016 |

在此网格内三者最高值分别0.941/1.741/8.173×，最高值都假定m=1已经达到所需质量，**尚无证据支持**。较具体的K=64,m=8是0.360/1.195/5.896×；中间假设m=16/32降至0.880/0.576×。

不依赖任一参数选择的2×必要条件：

\[
u<\tfrac12,\qquad
G+B(C+s_0+mp)\leq
\frac{K(0.5-u)}{\eta(1-u)}.
\]

例如较有利的u=.25、η=1.5、B=8、G=.75、C=.10、s₀=.20、p=.40，K=64要2×必须m≤3.46；m=8要求K≥129.4，超过本次64上限。若u=0、η=1并保持其余成本，K=64,m=8则可约2.23×。**是否成立取决于实测的u、η、m、B，而不是名称“variable projection”。**

三个必须避免的计账错误：

1. B=4/8/32是采样目标，不是所有训练视图上的精确β*(θ)。若V=200个相机且η/u等不变，B必须改成200，内层成本按视图数增长。warm-start、随机优化可能减少访问，但那是inexact算法而非免费全数据消元。
2. 若为维持几何收敛，Ng其实与原始geometry step数相同，则不能用Ng=N/K。η可能随K显著增大；K64不是天然64倍节省。
3. 旧room profile的88.1 ms/step与cuDNN SSIM约40.7%不是当前fused-SSIM强基线的成本分解；本报告没有用它校准模型。应重新测量appearance可替代成本以及cache成本。

**本次没有建立 realistic≥2× 的证据，因此不满足用户设定的moonshot门槛。** 模型提示哪些测量可能推翻此判断，不声称测量前已证明性能必然失败。

## 8. 不修改CUDA的最小prototype设计

这是实验设计；本次没有实现/运行GPU原型。先核验算子和目标，再决定是否值得写生产CUDA。

### 8.1 对照矩阵与checkpoint

- 选已有相同训练协议checkpoint，固定xyz/scale/rotation/opacity、相机、分辨率与topology；分早期、增密结束附近、收敛末期checkpoint，不能只选收敛后最好情况。
- SH degree分别0、1、3，q分别1、4、16。对每一级同时做**灰色重置**与**保留原appearance的小扰动/warm-start**。重置衡量从零修复能力；周期性training实际更接近warm-start。
- 场景最小集合覆盖室内、foliage、多层半透明/高重叠；固定一个小ROI/少量Gaussian构造显式A以检查数学，不把toy结果当全场景性能。
- A0：原renderer的Adam appearance-only；A1：**同样使用projection/tile缓存的Adam**；B：全局matrix-free LSQR/PCG；C：Instant Colorization式majorizer残差迭代；可选OGC相同PCG策略。只与A0比较会混淆缓存收益和solver收益。
- Adam至少调合理learning-rate范围；所有方法同初始化/正则/损失/视图集合。报告同一batch目标及跨view泛化；不能让CG全数据一次pass与Adam单视图一步都称为一次iteration。

### 8.2 stock gsplat 的算子

用现有gsplat拆分API执行 `fully_fused_projection → isect_tiles → isect_offset_encode`，保存geometry-derived张量。appearance阶段在Python/Torch计算SH颜色，并调用现有 `rasterize_to_pixels`；无CUDA源码编辑。

构造线性oracle时：

```python
# conceptual pseudocode; geometry cache is fixed per camera
def A_cam(v):
    colors = sh_basis_cam @ v_per_gaussian  # signed values allowed
    return rasterize_to_pixels(cache, colors, background=0)

def AT_cam(u):
    z = zeros_like(beta, requires_grad=True)
    z_image = A_cam(z)
    return autograd.grad(z_image, z, grad_outputs=u)[0]

def normal_product(v):
    # Sum over a FIXED view set, same weights/normalization every call.
    return sum(AT_cam(A_cam(v).detach()) for cam in views) + ridge * v
```

实际张量维度/packed Gaussian IDs需按API映射。**不要把signed搜索方向送入普通SH-clamped render；不要保留+0.5或非零背景。** 真实残差用b并保留原renderer语义，线性乘法只用A。VJP只需一次反向，不依赖自定义CUDA的double backward，也不必要求JVP支持。

在小oracle检查线性叠加、adjoint恒等式〈Av,u〉=〈v,Aᵀu〉、显式A与matrix-free结果、geometry缓存前后输出；FP32建议相对误差≤1e−4或经baseline数值噪声标定，FP64小矩阵做高精度参考。活动clamp另立测试，不能用无clamp结果假冒默认renderer。

### 8.3 无CUDA预条件器

对每视图、每Gaussian的预计算颜色做一次 `grad(sum(render), colors)`，得到S¹ᵥᵢ=Σp wᵥₚᵢ，再在Torch累积Mᵢ=ΣvS¹ᵥᵢYᵥᵢYᵥᵢᵀ。它是上文majorizer，**不是精确diag(AᵀA)**。可分块Cholesky、加入匹配目标的ridge或仅用于预条件的数值稳定项。

真实S² Gram若API不暴露，不假装已经有：小模型显式A可以得到；随机VJP探针可估计对角/块但增加passes；更大模型可先用S¹ majorizer。后者足以证伪solver经济性，而不需为预条件器先投入CUDA开发。

### 8.4 三阶段loss与停止判据

1. **L2 oracle**：无clamp affine map、匹配ridge；CG/LSQR对固定view集合，监测objective gap与true normal residual。另设真实clamp active-set版本，并标明APPROXIMATE。不得每次CG matvec换随机batch而仍使用标准CG递推。
2. **L1**：ADMM/primal-dual exact formulation；IRLS作为有平滑参数说明的approximate对照。全部记录outer与inner工作量，用真实L1检验。
3. **原L1+fusedSSIM**：通过现有fusedSSIM backward得到真实图像梯度；使用预条件梯度/proximal局部步，或由实际loss检验的阻尼quadratic/line-search。GN若需要完整SSIM窗口Jacobian，可在Python/Torch重写匹配窗口公式用于oracle并核对值/梯度，不能要求fused kernel已有二阶支持。所得是APPROXIMATE optimizer，目标可保持；是否达到β*另测。

### 8.5 必须输出的metrics

| 类别 | 记录 |
|---|---|
| renderer work | 每相机forward、backward、cache rebuild、RHS/preconditioner/validation passes；全视图sweeps；solver outer/inner迭代。 |
| wall time | GPU同步后的time-to-target、总setup、loss、solver、geometry/cache、evaluation分别列出；包含内存分配/传输，初始化warm-up单列。 |
| image quality | 训练/验证objective；held-out PSNR、SSIM、LPIPS，固定LPIPS backbone与SSIM定义；不让测试图像参加求解或调参。 |
| numerical | normal residual ‖Aᵀ(Aβ−y)+Λβ‖的明确归一化；数据residual；小oracle解误差；L1 primal/dual residual；真实SSIM目标与局部预测差。 |
| memory | GPU peak、Gram/solver向量/cache分项、CPU offload与传输。 |
| practical comparison | 到共同目标的时间曲线，质量–时间Pareto，而非只比较相同迭代数。warm-start是完整训练收益的主要筛选项。 |

预先登记质量容差建议：相对充分优化的同loss参考，PSNR下降≤0.10 dB，SSIM下降≤0.002，LPIPS上升≤0.005；真实训练objective不劣于参考超过1%（近零目标用绝对阈值）。这些是**建议门槛，不是实验已通过**。完整训练还需匹配Gaussian预算/增密策略并比较等质量time-to-target。

## 9. Kill criteria

以下是审查门，任何对应必要条件失败即可DROP当前moonshot定位；阈值应先登记再看结果，不事后移动。

| DROP条件 | 可检查定义 |
|---|---|
| **prior art覆盖核心** | 若贡献仅为固定geometry SH线性、PCG/LSQR、block Gram、残差迭代或缓存统计，本次文献已满足此DROP条件。必须提出并验证与OGC/Instant Colorization不同的完整训练机制，才是一个新的候选。 |
| solver passes接近Adam | 计入全部outer/restart/setup/validation，以同view单位到同质量；若solver工作量/墙钟≥cached-Adam的80%，且cache本身无法令完整模型过2×门，则DROP。若局部阶段仅2×，任何未加速部分都会使E2E低于2×。 |
| 条件数使CG慢 | SH3 warm-start在代表场景需>32次全view pair仍达不到objective/图像门，并且代入成本模型后<2×；不单凭κ数值杀死方向。 |
| block近似掉质量 | 一次独立block solve相对全局oracle超出0.10dB/0.002SSIM/0.005LPIPS或objective门；补全局迭代后又超成本预算，则DROP，不能只呈现有利SH0。 |
| full objective无法保持 | 只有去掉clamp、换L2或改变SSIM窗口/梯度才有速度；恢复原L1+fusedSSIM后达不到等质量加速门，则DROP“同语义训练加速”主张。 |
| geometry必须频繁更新 | K≥8就持续超过质量容差/需η大幅增加；只剩K≤4且实际窗口成本违反2×不等式，则DROP。缓存安全不等于冻结schedule合理。 |
| appearance不是瓶颈 | 理想地把可替代appearance阶段成本置0的反事实上限仍<2×，或实际profile未被替代耗时u≥.5，则DROP。不要把Adam参数写入时间当全部appearance收敛成本。 |
| 内层解得更好却外层更差 | warm-start/早期checkpoint中更低training appearance loss掩盖geometry误差，held-out质量变差或geometry到目标耗时更久；计入η后不足2×则DROP。 |
| memory / cache代价抵消 | 所需Gram/跨视图权重cache超目标显存或offload后wall-time失去2×，则DROP该实现路线。 |
| predicted E2E<2× | 用当前强基线profile和真实m/K/η校准，中间/现实假设speedup<2×，即不属于本任务moonshot。仅optimistic≥2×不够。 |

## 10. Final verdict

**DROP**

- **MEASURED**：仅完成CPU代数检查与模型网格；没有真实3DGS solver或完整训练加速测量。独立拟合的反例成立；CG真实场景速度未知。
- **LITERATURE**：OGC已实现最直接的固定几何SH matrix-free PCG；Instant Colorization已有解析SH与全局残差迭代；3DGS-LM/LM-RS、FastGS及ABCD分别覆盖二阶/矩阵作用、不同更新率及训练缓存。TurboGS明确给出高阶SH保留Adam的Jacobian/内存理由。宽泛的新颖性叙述不成立。
- **DERIVED**：未截断前向仿射，但完整loss不等于LS；高斯全局耦合不可随意删除；同目标精确内层的value gradient可用envelope，不需显式dβ*/dθ；geometry缓存有严格失效条件。
- **HYPOTHESIS**：当前中间成本假设最优仅1.741×，K64/m8约1.195×；optimistic存在>2×区域，需实测证实其假设。上述数字不称为实测或校准预测。
- **UNKNOWN**：仍没有“原L1+fusedSSIM + 默认clamp + SH3 + 增密/几何训练 + 当前强基线”组合的realistic≥2×证据，也没有证明这种组合绝对不可能。

若继续做本文无CUDA实验，应把它定位为**验证剩余训练收益的证伪实验**，而不是带着“社区忽略线性”的novelty narrative投入生产实现。本报告的DROP是reviewer建议，不冒充用户已采纳的项目决策。
