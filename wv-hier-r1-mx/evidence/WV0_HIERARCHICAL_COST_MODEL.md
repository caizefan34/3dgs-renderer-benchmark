# WV0 分层成本模型与绝对工作量审计

**VERDICT / evidence class：STRUCTURAL_ORACLE_ONLY；没有 G4/H16→4 kernel 或 F+B 实测。**

## OBSERVED — 成本计数器的范围与误差

原 1.4M / 4.0M / 13.2M / 180M 对应 room iter5000 的 camera 0，精确保存值为 1,424,660 / 4,021,594 / 13,162,206 / 180,296,285。它不是全部场景均值。每个 state 只保存 camera 0 的 bbox 交数，而 gradient 表是 8 cameras 平均，成本与误差轴不完全配对。

源码 `compute_work_counters` 用 floor 上界、闭区间、clip(tw−1)；真实 CUDA `IntersectTile.cu` 用 ceil 上界、半开区间、clip(tw)。在 64×64、G16、mean=(8,8)、radii=(8,8) 的公式例子，旧代理计 4，CUDA 区间规则计 1。这是边界公式反例，不是 synthetic 场景性能结果。原始 projected tensors / emitted IDs 没保留，不能量化真实场景偏差或称其为 physical isect_count。

## DERIVED — 绝对计数与 payload

对保存的 bbox 代理 I_G，均匀 tile Bernoulli 的预期选中代理为 p·I_G。每 G 自己归一化得到 p，并不代表 I_G 一样。G4/G16 的比值为 4.657154–9.238840；因此 p=.25 时 G4 相对 G16-dense 代理为 1.164289–2.309710，p=.4 时为 1.862862–3.695536。

CUDA 源码为每 entry 存 int64 isect_id 与 int32 flatten_id，即 12 bytes。`WV0_ABSOLUTE_COST.csv` 的 payload = 12pI_G 只是假定代理成立时的一套 key/value 大小，不含双缓冲、排序 workspace、扫描、读写流量或时间。实际 intersections、sort bytes、kernel_ms、F+B_ms 保持 NOT_MEASURED。

## RECOMPUTED — 两个 p 的 Pareto

`WV0_PARETO_RECHECK.csv` 给出每 state、每 p 四个候选的 NMSE–absolute-proxy 与 NMSE–payload frontier，另列允许两种 p 相互比较时的 frontier。12-byte payload 是同一绝对 proxy 的线性换算，所以 frontier 相同。

固定 p 下，降低 G 同时降低误差、提高代理成本，72/72 个点都是非支配的权衡点；“处于 frontier”不表示 G4 最优或免费改进。跨两种预算一起比较时，一部分点会被另一粒度/预算支配；详见 CSV 的 dominated_by 列。由于 cost=cam0、quality=8-camera mean，这仍只是代理 Pareto，不能包装成严格配对的真实 GPU 工作 frontier。NMSE–measured F+B frontier 不存在。

## DERIVED — 相同绝对成本的有限模型

令 a_t 为一个 tile 的全部像素梯度贡献，冻结 state/camera 时，独立 Bernoulli HT 的理论 NMSE 为 `(1−p)/p · K_G`，其中 `K_G=Σ_t ||a_t||²/||g||²`（每参数组分别定义，再做 macro）。从保存的有限样本均值估计 `K_G≈NMSE_G(p)·p/(1−p)`。

在同一代理成本 C 下，`p_G=C/I_G`，故模型 `NMSE_G(C)=K_G·(I_G/C−1)`。以 G16 的 .25 或 .4 成本为基准，推得 G4 NMSE/G16 为 **3.8391–6.3482**，G8 为 **1.8181–2.7605**，G1 为 **16.5749–31.3270**。这些外推点通常远低于已有采样 p，**不是观测结果**，假设还把 camera0 成本用于 camera-mean quality，且 I_G 本身是有边界缺陷的代理。

因此，现有数据不支持“同绝对交数成本下 G4 更好”；这个有限模型反而提示成本惩罚可能超过梯度收益。但其假设和证据缺口不允许据此直接 DROP 分层 B2，更不能换算为 CUDA 时间。

## DERIVED / HYPOTHESIS — 16→4 分层账本

| 成本项 | WV0 实际代码 | 原生 G4 的待测项 | 共享父列表 H16→4 的待测项 |
|---|---|---|---|
| Projection | 每次对全部输入 Gaussian 投影 | 通常仍需全部 N | 通常仍需全部 N；细 mask 不自动跳过 |
| Intersection/binning | 始终 G16 dense；G4 只有离线代理 | G4 发射/压缩真实 pair | 仅一套 G16 parent ranges；若父节点 active 才发射，仍几乎全 active |
| Sorting | dense G16 排序 | G4 key/value 排序 | 父列表排序一次；不得复制 16 份再排 |
| Fine mask / scheduling | CPU rand、像素 mask、CPU→GPU | tile list/packing/launch 成本 | fine child compact/index 与 parent-range 复用成本 |
| Rasterization | 全图渲染后 loss mask | 只有选中 G4 像素/tiles compositing | 只选中 child 的像素 compositing；遍历父候选可能多于原生 G4 |
| Backward VJP | dense forward 图上的 masked loss | 真实跳过未选中 child；精确有符号参数累积 | 同上；保持早停、depth-order、alpha/transmittance、SH/camera |
| Mask / IPW loss | 1/p 加权的 RGB SSE | 采样与固定分母 loss 必须计时 | 同上，不排除 host sync、分配等开销 |

对完整父块，P(active parent)=1−(1−p)^16：.25→0.989977404，.40→0.999717889。边界父块用实际有效 child 数 k：1−(1−p)^k。`raw/evidence/WV1_RECONSTRUCTED_MASKS.csv` 的 observed-now mask counts 包含边界。均匀 child 抽样不能声称父 bin/sort 节约 75%。按粗略结构模型，B2 相对同 p 的 G16 sparse 父 intersection 比约 q/p：3.9599 或 2.4993；它仍可能比 native G4 更少重复 intersections，但不保证总时间更快。

B2 是否有利取决于 blend/backward 节省能否覆盖父遍历、列表管理和调度开销。4×4=16 pixels，小于 warp32；packing、shared range reuse、早停、寄存器、occupancy 和 lane idle 均待测。本轮无 CUDA 修改，也无 profiler 数据。

## BLOCKED / HYPOTHESIS — 时间与训练

没有可用的 G16/G4/H16→4 同源码、同 state、同 camera、同 mask objective 的 kernel/F+B 配对计时。旧训练 phase_timing.json 属于其他方法、轨迹与 runtime，不纳入本比较。

假如将来 B2 每步更慢，time-to-quality 改善至少要求 `steps_B2/steps_B0 < time_B0/time_B2`。当前没有时间比，不能给数值 break-even，更不能把 NMSE 比代入迭代数比。冻结梯度误差不是收敛速度。
