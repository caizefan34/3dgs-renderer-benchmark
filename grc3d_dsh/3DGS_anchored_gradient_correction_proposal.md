---
tags: [3DGS, Research, CUDA, gradient-estimation, control-variate]
created: 2026-09-27
updated: 2026-09-27
type: research-proposal
status: hypothesis-with-cpu-oracle
---

# 锚定双分辨率梯度校正 3DGS

## 结论与目标

这是针对用户“原创、超越现有强 3DGS”的要求提出的具体研究候选，不是已有算法的实测胜出结果。目标限定为：保持标准单一 Gaussian 表示及目标 L1+SSIM，在当前快速训练骨架上降低等质量完整训练时间，争取约 2×。

本轮完成：数学估计器设计、有限范围先行工作检索、CPU float64 的 2D alpha-composited Gaussian 机制实验、成本门槛推导。未完成：3D CUDA 原型、真实场景训练、完整新颖性审查、SOTA 胜出验证。

核心假设：高分辨率完整梯度的大部分，可以由一个廉价、当前参数可微的代理解释；只对解释不了的梯度差付出高分辨率计算成本。需要实验证明这一假设在真实场景、参数组和训练阶段成立。

## 1. 为什么要用“锚定代理”

直接以低分辨率图像取代高分辨率图像，会带来持久的细节缺失。即使精细模型接近目标，粗图仍然模糊，粗梯度可能持续指向错误方向，令控制变量的方差很大。

为每个训练相机 v 保存一次精细渲染和对应的粗渲染：

- A_v = R_f(theta_anchor, v)：历史精细图像；
- B_v = R_c(theta_anchor, v)：同一历史状态的粗图像。

缓存均视为常量，不保留历史计算图。当前步骤重新计算 R_c(theta, v)，构造：

$$
\widetilde I_v(\theta)
=A_v+U\left[R_c(\theta,v)-B_v\right].
$$

U 是固定线性上采样，初版用 bilinear。必要的颜色 clamp 在代理中明确定义并参与求导；精细目标保持原实现的 clamp。

这个代理保存了历史精细图像，使用当前粗渲染解释其变化。所有 Gaussian 参数仍可更新，没有冻结历史 Gaussian，也不求解 SH 正规方程。

如果缓存未压缩且 theta=theta_anchor，代理图像值恰好等于历史精细图像；**代理 Jacobian 通常仍不等于精细 Jacobian**。保留图像细节不等于免费得到细节梯度，后面的校正必不可少。

缓存可以陈旧甚至近似，但会损害方差和效率。只要它是条件固定的、两处使用的是同一个代理，后续估计器的条件无偏性并不要求代理本身精确。

## 2. 从完整损失构造校正

固定当前相机、参数和 topology。以基线的完整分辨率渲染 R_f 定义真实目标：

$$
L_f(\theta)
=\frac1{HW}\sum_{p}
\left[0.8\,\ell_1(R_f(\theta)_p,I_p)
+0.2\bigl(1-\mathrm{SSIM}_p(R_f(\theta),I)\bigr)\right].
$$

RGB reduction、窗口、padding、clamp 等以实际基线为准。上式仅示意该局部损失结构。

将**损失中心像素**分成互不重叠的块 b，令 L_b^f 为每块对完整图像损失的贡献，已经包含全图归一化，所以 L_f=sum_b L_b^f。用代理图像定义相同分块的 L_b^p 和 L_p。

各块以已知包含概率 pi_b>0 独立 Bernoulli 抽样，指示变量为 Z_b。构造：

$$
\widehat L
=L_p+\sum_b \frac{Z_b}{\pi_b}
\left(L_b^f-L_b^p\right).
$$

对当前参数求导：

$$
\widehat g
=g_p+\sum_b \frac{Z_b}{\pi_b}
\left(g_b^f-g_b^p\right).
$$

条件期望为：

$$
\mathbb E[\widehat g\mid\theta,\text{history}]
=g_p+\sum_b(g_b^f-g_b^p)
=g_f.
$$

这里的历史包括代理缓存、采样概率和调度策略状态。必须对它们 stop-gradient，且在本轮新抽样之前确定。若使用其他抽样方案，需要使用该方案的真实包含概率；不能把没有放回的权重简单当作类别抽样概率。

这是经典 control-variate 思想的应用，不把该恒等式本身当作原创理论。

### 为什么没有低分辨率目标偏置

本方法求的是当前精细目标的随机梯度。粗代理提供覆盖，精细减代理的项消掉代理偏差。精细校正区域使用当前真实参数渲染，不以缓存图像替代。

需要避免两个错误：

1. 不可先构造一个无偏图像估计，再把它放进非线性 L1/SSIM，就声称损失梯度无偏；这里直接对完整、确定的局部损失进行差值抽样。
2. 不可省掉减去 L_b^p 的项；否则得到的是低分辨率辅助损失加稀疏高分辨率损失，目标已经改变。

估计损失的某次取值可以为负，代理项系数也可能为负。这本身不否定估计器。对估计损失做非线性截断会改变估计器。

## 3. 计算路径：一次投影，汇合两路反向

建议共享当前参数下与像素采样分辨率无关的投影和 SH 工作：

1. 计算一次相机空间几何、规范坐标中的 2D 均值/协方差、深度和 SH 颜色。
2. 从共享状态生成粗网格，以及精细选中区域的 footprint。
3. 粗网格做全图 rasterization；精细网格只渲染抽中的中心块及必要邻域。
4. 构造代理图像，计算整张代理损失及精细局部损失差。
5. 汇合两路对共享投影状态、颜色的 adjoint。
6. 执行一次共享的 3D 参数投影反向与 SH 反向，再执行基线 optimizer。

如果 z=P(theta)，且两个分支均通过 z 求导，核心为：

$$
\nabla_\theta\widehat L
=J_P(\theta)^\top
\left(\frac{\partial \widehat L}{\partial z}\bigg|_{\rm coarse}
+\frac{\partial \widehat L}{\partial z}\bigg|_{\rm fine}\right).
$$

此处是在计算图中共享公共子表达式，不是宣称已有所有 CUDA 实现天然暴露该接口。初版可用两个分支验证机制，生产版再合并 adjoint；不能把尚未实现的共享收益提前计入实测。

分辨率相关的 eps2d、抗混叠滤波、opacity compensation、radii、culling 和 tile 列表需要分别处理；不能直接重用错误 footprint。两个分辨率的 tile lists 也不能假定完全相同或免费共享。

在损失层可把代理块系数写成 1-Z_b/pi_b，只对代理 renderer 做一次合并后的 backward，而不重复运行一次整图代理 backward 和一遍逐块代理 backward。

### SSIM 支持域

对于 11×11 SSIM：

- 一个 B×B 的损失中心块需要四周 5 像素的真实渲染邻域；
- B=64 时，是最多 74×74 的区域，面积系数约 1.337；
- 邻近块应去重渲染其公共邻域；
- 只计入选中中心的损失，保留 halo 像素对这些中心的反向贡献；
- 只有真实图像边界使用基线 padding，不能把 patch 边界误当成图像边界；
- 使用原相机和全图像素位置，crop 的坐标变化必须正确处理。

名义抽样 1/8 不意味着精细 renderer 耗时为 1/8，halo、depth complexity 和不规则调度均需计量。

## 4. 决定是否能省计算的是梯度差

令 d_b=g_b^f-g_b^p。独立 Bernoulli 抽样下：

$$
\operatorname{tr}\operatorname{Cov}(\widehat g)
=\sum_b\frac{1-\pi_b}{\pi_b}\|d_b\|_2^2.
$$

普通稀疏梯度在同样抽样方案下对应 g_b^f。本候选希望让 d_b 的二阶能量远小于 g_b^f，而不是只让图像残差更小。

定义筛选指标：

$$
\kappa=\frac{\sum_b\|g_b^f-g_b^p\|_2^2}
{\sum_b\|g_b^f\|_2^2}.
$$

若普通稀疏方案用 pi=1/4，而本候选用 pi=1/8，则条件方差比为 7*kappa/3。因此要在减半精细抽样量时不增加该方差，需要 kappa<=3/7。**这比较的是两种稀疏估计器，不是说明其方差低于确定的完整单视图梯度。**

真实参数组单位不同，还应分别分析 xyz、scale、rotation、opacity、SH，并用固定的历史 optimizer 缩放度量复核。CPU 测试中的原始坐标范数不能直接当作真实 Adam 收敛的度量。

### 调度

最小原型用均匀 pi=1/2、1/4、1/8。后续才用历史梯度差与实测成本分配预算。

在“块成本可加、独立抽样、梯度差已知”的简化模型中，最小化方差可导出：

$$
\pi_b=\operatorname{clip}
\left(\lambda\sqrt{\frac{V_b}{C_b}},\pi_{\min},1\right),
\qquad V_b=\|d_b\|^2.
$$

这也是已有重要性抽样原理，不宣称新颖。实际 halo 共享令成本非可加；V_b 也不能每一步免费得到。先用少数离线块测量真实参数梯度差，在线仅用经验证的代理统计或稀疏探针，全部开销计入。

基于本轮样本发现问题后再挑选性丢弃该梯度，可能引入选择偏差。刷新/切换决策应基于历史信息，或使用独立 pilot 后重新抽取用于估计的样本。

## 5. 增密和 Adam 的边界

条件无偏梯度并不等于：

- 与 dense Adam 更新相同；
- Adam 二阶矩无偏；
- absgrad / 梯度范数增密统计无偏；
- 完整训练轨迹相同；
- 最终测试质量一定保持。

初版保留基线 optimizer，全参数状态与动量成本仍然存在，不使用“没有采样就不更新”的等价性主张。

增密阶段采用单独的 dense 校准步收集**真实精细**统计，同时刷新当前相机的 A_v/B_v。只有这些步更新增密统计的累积量与配套计数，稀疏随机梯度不直接进入原 absgrad 阈值链。

这改变了统计采集频率，不能称为原增密规则的精确等价实现。验证分两层：

1. 机制消融中，dense 对照和候选使用相同的统计采样日程。
2. 最终必须再与未经这种改动的官方完整方法比较，防止通过削弱对照制造收益。

如果要恢复相同质量而不得不频繁运行 dense 校准、改变阈值或延长训练，必须完整计入时间。增密期是完整训练 2× 的主要风险之一。

新 Gaussian、opacity reset、明显几何变化或遮挡变化会破坏代理关联性。代理在任意固定缓存下仍有上述期望校正，但效率可能崩溃；不能依赖“无偏”忽略方差。

## 6. 本轮 CPU 机制实验

环境：NumPy/SciPy；float64；6 个二维 Gaussian，固定深度顺序、24×32 RGB 图像；粗网格为宽高各 1/4；原图 11×11 SSIM，8×8 中心块。使用有限差分计算位置、log-scale、opacity logit、RGB logit 的梯度。

它不是 3DGS 场景实验：没有 3D projection、SH、动态排序、CUDA、增密或完整训练。场景为人工构造，固定一个 seed，不代表自然场景分布。

| 人工场景 | 直接粗图 kappa | 锚定代理 kappa | 候选 pi=1/8 / 普通 pi=1/4 的条件方差比 |
|---|---:|---:|---:|
| 宽 Gaussian，近期锚点 | 0.5260 | 0.0136 | 0.0317 |
| 细小 Gaussian，近期锚点 | 1.3597 | 0.8339 | 1.9459 |
| 宽 Gaussian，较大锚点漂移 | 0.5260 | 0.2890 | 0.6744 |

结果：

- 锚定机制在第一种测试中显著降低梯度差能量。
- 第二种测试**失败**：减少精细抽样反而增大方差；必须提高精细比例、提升粗分辨率或回退 dense。
- 锚点陈旧会削弱收益。
- 分块损失加 halo 与整图对应损失一致，最大差为 0。
- 根据数值块梯度计算的条件期望与精细数值梯度的相对误差约 1e-16；这是代数一致性，不是自动微分准确性或真实训练验证。
- 有限差分步长由 1e-5 改为 1e-6 后，三种 kappa 最大绝对变化约 1.76e-4。

不能将上述 0.0317 的方差比写成“真实场景快 31×”，也不能据此认为真实 scene 的 kappa=0.0136。

## 7. 2× 的成本条件

以当前强基线完整单步成本归一化为 1，定义：

- u：无法随精细区域缩小的固定部分，包括共享投影、optimizer 等；
- c：额外全图代理分支成本，包括粗 rasterization、代理全图 L1/SSIM 与其 backward；
- r：精细分支实际工作比例，已经计入 halo 和调度影响；
- h：正常步骤的额外缓存读写、采样、临时缓冲等开销；
- delta：dense 校准步占比；
- d：一次 dense 校准步相对基线成本，包括同时生成粗锚点；
- eta：到达同质量所需步骤数相对基线的比例；
- zeta：未计入上述步骤的额外设置/传输成本，相对基线完整训练总时间。

$$
S=
\left[
\eta\left((1-\delta)\bigl[u+c+(1-u)r+h\bigr]+\delta d\right)
+\zeta
\right]^{-1}.
$$

当各阶段差异大时应分阶段按实际基线耗时加权，不能直接平均迭代占比。

以下仅为**未校准预算假设**：

| 情况 | u | c | r | h | delta | d | eta | zeta | 推导 S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 能碰到 2× 的预算例子 | .15 | .08 | .18 | .03 | .05 | 1.05 | 1.10 | .01 | 2.003× |
| 固定成本与刷新较重 | .30 | .12 | .30 | .06 | .15 | 1.08 | 1.30 | .02 | 1.007× |

第一行并不是现实预测。它说明 2× 需要同时满足：代理便宜、精细工作量低、刷新少、额外收敛代价小。缺少其中任何一项都可能失败。

若 FastGS 的默认阶段已经很低分辨率、Gaussian 很少，代理成本和固定成本可能占优。不得以大模型后期 kernel 的收益外推当前最快完整训练骨架。

### 缓存预算

float16 存储每相机一张精图和宽高各 q 倍的粗图，约需：

$$
M_{\rm cache}=2\cdot3VHW(1+q^2)\ {\rm bytes}.
$$

V=200、1920×1080、q=1/4 时约 2.64 GB（十进制），不包含 GT、梯度或模型。CPU offload、缓存压缩、LRU 都可以研究，但传输成本与代理失真必须计入；本提案不默认这些操作免费。

## 8. 与已知工作的关系及原创性边界

| 已知工作/原则 | 已覆盖的思想 | 本候选要验证的具体区别 |
|---|---|---|
| 普通稀疏 tile / Random-Tile / TurboGS | 只计算部分像素或 tile，结构损失与密度控制适配 | 精细预算用于“当前精细损失梯度减当前代理损失梯度”；代理覆盖全图，保留原 L1+SSIM 的条件期望 |
| LFH | 分频、冻结已有 Gaussian 场、图像域组合 | 只有一套持续更新的标准 Gaussians；历史图像用于梯度代理，不成为最终分层表示 |
| Eulerian GS | 对 Gaussian 位置采样分布使用控制变量降低 score-function 梯度方差 | 校正标准显式 Gaussian 参数的不同分辨率 pathwise 梯度，不替换为概率密度场 |
| Recursive Control Variates for Inverse Rendering | 借助历史信息减少可微渲染优化中的误差/方差 | 为确定性 splat renderer 构造锚定粗分辨率代理，按完整局部损失校正，并共用 projection/SH adjoint |
| 多层/多保真 SGD 与控制变量 | 粗近似加随机精细修正 | 通用数学基础；不是本候选可独占的理论贡献 |

来源：

1. [Eulerian Gaussian Splatting 作者项目](https://euleriansplatting.github.io/)
2. [Recursive Control Variates for Inverse Rendering，SIGGRAPH 2023](https://rgl.epfl.ch/publications/Nicolet2023Recursive)
3. [TurboGS，2026](https://arxiv.org/html/2606.15924v1)
4. [LFH，2026](https://arxiv.org/html/2609.03334v1)
5. [Random-Tile Loss，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Tile-wise_vs._Image-wise_Random-Tile_Loss_and_Training_Paradigm_for_Gaussian_ICCV_2025_paper.html)
6. [Multilevel Optimization for Inverse Problems，2022](https://proceedings.mlr.press/v178/weissmann22a.html)

本次检索覆盖 3DGS 与 multilevel/multifidelity/control-variate/gradient-correction/anchor/cache 等组合词，以及上述直接相关工作。没有命中完全相同设计不证明首创；本次没有穷尽论文、源码、专利或所有 2026 新工作。

可争取的论文贡献必须落在可测量的具体问题上：

- 锚定代理是否实质降低了真实 3DGS 参数梯度差，而直接粗图失败；
- 共享 projection/SH 反向是否让代理校正比两次独立 renderer 便宜；
- 如何在 SSIM 邻域、增密和遮挡变化下把方差与刷新成本控制在可用范围；
- 是否在现代强基线上改善完整等质量训练的 Pareto 曲线。

仅把 control variate 写入公式，或重新命名稀疏 tile 训练，不足以成立原创贡献。

## 9. 最小真实原型与判停

### Gate A：先测代理有没有用

room、bicycle、garden；早期、增密前后和后期状态；SH0/SH3。固定一个状态和一组相机，逐块取得真实精细与代理参数梯度。

对照：

- A0：完整精细梯度；
- A1：直接粗图 + 精细差校正；
- A2：锚定代理 + 精细差校正；
- A3：相同实际精细成本的普通 tile 抽样；
- A4：仅代理，无精细校正，用来显露偏差；
- A5：同样校正但不共享投影，用来分离算法与系统收益。

初版可使用两个现有 rasterizer 和 Torch 计算损失；不必先写生产 CUDA。若没有原生局部渲染，先渲染整图再取块仅用于梯度/方差检查，不能报告为加速实现。

预先登记筛选：关键阶段和关键参数组需要有足够小的 kappa，例如目标 <=0.2；这只是下一阶段门槛，不是理论必须值。若在花草、细边界、SH3 上长期接近 1，否决统一低采样配置；只有局部/阶段性收益时需重新算完整时间上界。

### Gate B：测真实成本

测 u/c/r/h/delta/d/eta 所需的直接数据，包含 proxy full-image SSIM、projection floor、dense optimizer、缓存传输、halo、tile emission、排序、临时缓冲和校准。

所有“已共享”“只渲染选中区域”的性能主张须由实际实现支持。若理想省掉可替代部分仍不能满足目标，则不投入长期训练。

### Gate C：完整训练

在相同硬件、数据划分、渲染分辨率与评估指标下，比较 FastGS、Faster-GS、ShorterSplatting、TurboGS（明确其不同 loss），以及用户当前最优质量合格配置。

建议质量门：PSNR >= baseline-0.10 dB、SSIM >= baseline-0.002、LPIPS <= baseline+0.005；在确认实验前登记，逐场景和三 seed 报告。测试集不参与缓存拟合、抽样权重或阈值选择。

只有在包含增密和刷新成本的完整训练中达到等质量 2×，才可写“2×”。如果只在后期、固定 topology、单一高分辨率场景或相对原始 3DGS 获益，则按实际范围表述。

## 10. 判定

**进入小规模真实梯度原型的候选；尚不进入“已获原创 SOTA”状态。**

机制实验显示锚定值得验证，同时明确暴露了细小基元和陈旧缓存的失败风险。下一项决定性测量是：在真实 SH3 场景、当前强基线上，梯度差能量与额外代理成本是否同时足够小。


## 附录：本轮 CPU oracle 完整代码

该代码复现上文人工场景的机制测试，不提供生产 CUDA 或训练加速实现。

```python
"""CPU float64 2D alpha-composited Gaussian oracle; NOT a 3DGS speed benchmark."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve1d, map_coordinates

H,W,B,R=24,32,8,5
N=6
blocks=[(y,min(y+B,H),x,min(x+B,W)) for y in range(0,H,B) for x in range(0,W,B)]
k=np.exp(-np.arange(-R,R+1,dtype=float)**2/(2*1.5**2)); k/=k.sum()
def sigmoid(z): return 1/(1+np.exp(-z))
def render(t,h,w):
    p=t.reshape(N,8)
    yy,xx=np.meshgrid((np.arange(h)+.5)/h,(np.arange(w)+.5)/w,indexing="ij")
    image=np.zeros((h,w,3)); trans=np.ones((h,w))
    for v in p: # fixed depth order
        e=((xx-v[0])/np.exp(v[2]))**2+((yy-v[1])/np.exp(v[3]))**2
        alpha=np.minimum(.99,sigmoid(v[4])*np.exp(-.5*e))
        image+=(trans*alpha)[...,None]*sigmoid(v[5:])[None,None,:]
        trans*=1-alpha
    return image
def up(a):
    h,w=a.shape[:2]
    yy,xx=np.meshgrid((np.arange(H)+.5)*h/H-.5,(np.arange(W)+.5)*w/W-.5,indexing="ij")
    return np.stack([map_coordinates(a[:,:,c],[yy,xx],order=1,mode="nearest") for c in range(3)],axis=-1)
def filt(x):
    return convolve1d(convolve1d(x,k,axis=0,mode="constant",cval=0),k,axis=1,mode="constant",cval=0)
def loss_map(a,b):
    ma,mb=filt(a),filt(b)
    va,vb=filt(a*a)-ma*ma,filt(b*b)-mb*mb
    cov=filt(a*b)-ma*mb
    ss=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return (.8*np.abs(a-b)+.2*(1-ss)).mean(axis=-1)
def parts(a,b):
    lm=loss_map(a,b)
    return np.array([lm[y:Y,x:X].sum()/(H*W) for y,Y,x,X in blocks])
def halo_error(a,b):
    full=parts(a,b); vals=[]
    for y,Y,x,X in blocks:
        y0,y1,x0,x1=max(0,y-R),min(H,Y+R),max(0,x-R),min(W,X+R)
        lm=loss_map(a[y0:y1,x0:x1],b[y0:y1,x0:x1])
        vals.append(lm[y-y0:Y-y0,x-x0:X-x0].sum()/(H*W))
    return float(np.max(np.abs(full-np.array(vals))))
def scenario(scale, drift,seed):
    rng=np.random.default_rng(seed)
    t=np.zeros((N,8)); t[:,:2]=rng.uniform(.15,.85,(N,2))
    t[:,2:4]=np.log(scale)+rng.normal(0,.1,(N,2))
    t[:,4]=rng.normal(.2,.3,N); t[:,5:]=rng.normal(0,.7,(N,3))
    truth=t.copy()
    truth[:,:2]+=rng.normal(0,.012,(N,2))
    truth[:,2:]+=rng.normal(0,.04,(N,6))
    anchor=t.copy(); anchor[:,:2]+=rng.normal(0,drift,(N,2))
    anchor[:,2:]+=rng.normal(0,drift*3,(N,6))
    t=t.ravel(); target=render(truth.ravel(),H,W)
    ref_f=render(anchor.ravel(),H,W); ref_c=render(anchor.ravel(),H//4,W//4)
    def images(z):
        fine=render(z,H,W); low=render(z,H//4,W//4)
        return fine,np.clip(up(low),0,1),np.clip(ref_f+up(low-ref_c),0,1)
    def evaluate(z):
        return np.stack([parts(a,target) for a in images(z)])
    eps=1e-5
    jac=np.empty((3,len(blocks),t.size))
    for j in range(t.size):
        step=np.zeros(t.size);step[j]=eps
        jac[:,:,j]=(evaluate(t+step)-evaluate(t-step))/(2*eps)
    gf,gr,ga=jac
    den=np.square(gf).sum()
    kraw=np.square(gf-gr).sum()/den
    kanch=np.square(gf-ga).sum()/den
    q=np.linspace(.08,.20,len(blocks))
    expected=ga.sum(axis=0)+np.sum(q[:,None]*(gf-ga)/q[:,None],axis=0)
    groups={"xy":[0,1],"log_scale":[2,3],"opacity_logit":[4],"rgb_logits":[5,6,7]}
    group_result={}
    for name,cols in groups.items():
        ix=[i*8+c for i in range(N) for c in cols]
        group_result[name]=float(np.square((gf-ga)[:,ix]).sum()/max(np.square(gf[:,ix]).sum(),1e-30))
    # Exact conditional variance, independent Bernoulli p=1/8 vs plain p=1/4.
    var_ratio=(7*np.square(gf-ga).sum())/(3*np.square(gf).sum())
    return {
      "scale":scale,"anchor_xy_drift_std":drift,"seed":seed,
      "raw_coarse_residual_energy_ratio":float(kraw),
      "anchored_residual_energy_ratio":float(kanch),
      "anchored_group_energy_ratios":group_result,
      "variance_ratio_anchored_p0125_to_plain_p025":float(var_ratio),
      "conditional_expectation_relative_error":float(np.linalg.norm(expected-gf.sum(0))/max(np.linalg.norm(gf.sum(0)),1e-30)),
      "ssim_halo_max_abs_loss_error":max(halo_error(a,target) for a in images(t)),
      "fine_loss":float(evaluate(t)[0].sum())
    }

results={"scope":"CPU float64, six 2D Gaussians, fixed depth, 24x32 RGB, 11x11 SSIM; finite-difference gradients. No GPU training, 3D projection, SH, sorting or densification.",
         "finite_difference_step":1e-5,
         "cases":[scenario(.20,.003,7),scenario(.04,.003,7),scenario(.20,.06,7)]}
print(json.dumps(results,indent=2))
Path(__file__).with_suffix(".json").write_text(json.dumps(results,indent=2))

```

