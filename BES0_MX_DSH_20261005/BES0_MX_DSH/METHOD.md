# BES-GS：有界事件高斯泼溅（研究假说，BES0）

本包是可证伪的研究执行规格，不是已验证的加速器。主目标：在相同质量门槛下，训练达到目标质量的总墙钟时间至少缩短 3 倍。新颖性仅是候选组合；没有证明首创、论文录用或颠覆性。所有科学实验只在 MX 执行。

## 1. 固定目标

保留投影、有效支持、alpha 上限/阈值、中心深度顺序、SH 与背景。数学目标是有限列表的完整 alpha 合成（关闭 transmittance early-stop）：

I = sum_i T_i alpha_i c_i + T_end c_bg，T_i = product_{j<i}(1-alpha_j)。

每个 Gaussian–tile occurrence 独立分类，像素所属 tile 只能出现一次该 primitive。同一高斯可以在不同 tile 属于不同集合。排序平局按冻结的 native primitive-id 规则处理。不得改变深度模型后还声称相同图像。

## 2. 有界事件

找 tile 内有效 alpha 的上下界 l_i <= alpha_i(p) <= u_i。l_i>0 且 u_i/l_i<=gamma 的 occurrence 进入 R，剩下进入 D。gamma=2。跨 cutoff/support 边界的 occurrence 必须 l=0、进入 D。alpha=0 的 occurrence 可以安全删除。

投影椭圆的 q=(p-mu)^T A(p-mu)，A 正定。q 最大值位于矩形角点；最小值取中心（若在矩形内）或四条边的约束最小值。边 x=x0 时 y*=clip(mu_y-b/c*(x0-mu_x))；边 y=y0 时 x*=clip(mu_x-b/a*(y0-mu_y))。先包围连续 tile 再考虑离散像素；不得用稀疏采样的最大值当上界。实际浮点认证见 NUMERICS.md。

对 R 建 lambda_i=-log1p(-u_i)，前缀 Lambda。当前位置 k 之后，抽 E~Exp(1)，二分找到第一个 Lambda_i > Lambda_k+E；无则背景。仅在候选计算 alpha_i(p)，以 alpha_i/u_i 接受；拒绝后跳过该 primitive，重新抽 E。不能使用 log-hazard 的比值接受，不能拒绝后留在同一个 primitive。

理由：跳跃等价于按顺序独立尝试概率 u_i 的候选；再用 alpha_i/u_i 稀释后，每一项的接受概率恰为 alpha_i。

因此第一接受 H 的概率为 alpha_H * product_{j in R,j<H}(1-alpha_j)。若 Q 为候选评估次数：E[Q]=sum_{i in R} u_i T_i^R <= gamma，且 P(Q>k)<=(1-1/gamma)^k。查询前缀仍需 O(log|R|)，边界、建表、排序和 D 的成本不在这个界内。

## 3. 给定事件，解析积分 D

F_H=E_D(<H)+T_D(<H)c_H；H=背景时使用全部 D 与背景。这里 E_D 是按原顺序只合成 D 的前缀。R 和 D 可以交错，不能把 D 移到最前面。E[F_H]=I。相对于全随机透明度事件，这是条件期望，因此颜色方差不增加；不推出梯度方差或每秒方差更好。

## 4. 可训练的三次事件查询

A 路独立抽 H_A，产生残差 r=stopgrad(F_A-y)。B 路抽 H_B，再抽它之后 R 的首接受 J_B（或背景）。A 与 B 的随机流必须独立。固定事件索引、R/D 分类与采样表，不反传采样路径。构造：

J_hat = partial_theta F_H + stopgrad(F_H-F_J) (grad_theta log alpha_H)^T。

背景时第二项为零。第一项包含 D 的 alpha、所有相关颜色与几何依赖；不能丢弃 SH 方向的几何导数。F_J 的导数只在它自己作为 H 的样本时出现，不能穿过本样本的差值系数。

L_sur = r dot F_H + stopgrad(r dot (F_H-F_J)) log alpha_H。

使用普通 autodiff 对这个标量反传即可，背景省略 log 项。E[grad L_sur] 等于完整确定性图像的 1/2 L2 梯度。直接优化随机 F 的平方误差会额外优化方差，不是同一个目标。L1/SSIM 没有这个无偏承诺。score 项来自已有 dual-hit 估计器，不是本方案单独首创。

每个训练样本三次 next-hit 查询，gamma=2 时 R 的候选 alpha 查询期望 <=6；D 可一次顺序扫描，取得 H_A、H_B、J_B 三个深度的前缀快照。每像素增加多个独立样本按比例增加成本。

## 5. 研究价值与失败条件

候选差异是“限制上下界比值的路由 + log-hazard 跳跃 + 随机/确定性集合交错的条件积分 + 三查询训练路径”。组成部件有明确先例。只有真实总耗时、相同质量和收敛结果能支持价值判断。A100 没有假定 RTX RT 功能。

可能失败：大量 occurrence 的 l=0；D 扫描占主导；前缀与认证比省下的查询更贵；梯度方差要求更多迭代；相同质量的渲染需过多 spp；固定 N 的训练收益不能推广至 densification。任何一项都应如实输出 NO_GO 或 INCONCLUSIVE，不能把“查询数减少”改名“训练 3x”。
