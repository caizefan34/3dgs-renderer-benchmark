# WV0 统计与原始证据审计

**VERDICT：BLOCKED_RAW_EVIDENCE。** 本文区分直接读取的事实、从汇总复算的结果和数学推导。未复算不存在的原始梯度。

## OBSERVED — 数据恢复

读取 MX `/dev/shm/wv0_out/` 的完整输出，并校验原交付 manifest：94/94 匹配；9 个 checkpoint SHA 均匹配。指定 home/pool 深度 9 搜索只找到本轮已保存的副本，原始结果位于易失的 `/dev/shm`。归档 `wv0_deliverables.tgz` 含 95 个文件（含 manifest），其中 RAW 的 SHA 与解包版本相同；数据归档只含相机/图像。Windows 的 95 个交付文件及归档也编入清单。哈希相同的副本不重复计为样本。筛选 shell history 未找到 WV0 命令；保留的 launcher 和执行日志则可读。

关键证据链：`raw/evidence/wv0_live_scripts/_wv0_campaign_state.py:115` 将 8 个 camera 指标平均；`:142` 定义 `stats()`；`:166–173` 只保存统计字典；`:181` 保存 `.pt`。`_wv0_addraw.py` 将这些字典改名为 `*_per_mask`，没有还原逐 seed 数组。

## OBSERVED / DERIVED — 样本量与有效样本单位

9×8×4×2×32 = 18,432 是源码与日志维度所暗示的 camera-level 估计数。32 个 seed 的每个指标先平均 8 cameras，因此若保留，应有 2,304 个 camera-averaged seed 记录。实际仅保留 72 个 state-condition 汇总、360 个 group 汇总、1,080 个 metric-stat 字典；逐 camera/seed 行为 0，逐 seed 数组为 0，完整梯度张量为 0。

9,216 没有对应的去重或聚合规则。所有 9 个日志均有 cams=8、rc=0 和 SAVED，未发现减半相机的证据，但没有逐调用索引可独立确认缺失组合。`n=32` 是保存的汇总样本数，不能乘以 5 组或 8 cameras 冒充独立重复。

每个 camera 都重置同一 CPU seed；相同尺寸的 camera 使用同一空间 mask。跨 checkpoint 也复用种子。应以 seed 为配对簇，保留 camera 层级，不把同一 mask 的多视角响应当独立随机 mask。跨 G 或 p 的 RNG 序列也复用，配对协方差需要原始记录。`numpy.std` 默认 ddof=0。

## OBSERVED — 相机与参数语义

本机准备脚本从完整列表的 `floor(i*n/24)`（i=0..23）取 24 个视角；MX runner 使用其中前 8 个。相机 JSON 没有 train/test 字段，WV0 loader 不执行分割。当前通用 GTDataset 加载所有匹配视角，也不能据此证明历史 checkpoint 的独立 eval 约定。room 历史 provenance 的 trainer SHA 与当前 trainer 不同；garden/bicycle 对应目录没有 provenance.json。**这些视角不是已验证的 held-out 评估集**，本轮只解释冻结梯度 oracle。

| scene | WV0 camera IDs | [H,W] |
|---|---|---|
| room | 0, 12, 25, 38, 51, 64, 77, 90 | [540, 810] |
| garden | 0, 7, 15, 23, 30, 38, 46, 53 | [540, 833] |
| bicycle | 0, 8, 16, 24, 32, 40, 48, 56 | [540, 813] |

保留完整相机 JSON、原图路径/SHA、checkpoint SHA。加载器使用 c2w 的逆矩阵作为 w2c，K 主点设为 W/2,H/2，双线性 resize，FP32 RGB/255；tile_size=16、packed=False、SH=3、eps2d=.3、radius_clip=0。梯度对 activated means、normalized quats、exp(scales)、sigmoid(opacity)、SH colors 求导，并非原训练器所有未激活参数的链式梯度；不能把这五组统计直接当作原训练优化器更新效果。

## OBSERVED / DERIVED — 损失与 HT

`_wv0_grad.py:45`：L = Σ_pixels,RGB (r−y)²；`:63`：Lhat = Σ M_tile/p · (r−y)²。是平方误差总和，不是已除以 3HW 的 MSE。tile 由 `torch.rand < p` 独立 Bernoulli 产生，没有固定个数抽样，没有按实际选中像素数再归一化；边缘 tile 中的每个有效像素仍以 p 入选。

对冻结 pixel-gradient a_x，E[M_tile/p]=1，因此 E[Σ M_tile/p a_x]=Σa_x。若目标定义为 per-channel MSE，应在 dense 与 sparse 同除以固定 3HW。共同标量不改变同 camera 的 NMSE/cosine/relL2，但绝对方差乘以其平方。该线性可加证明不覆盖 DSSIM 或自适应选择，也不替代 CUDA backward 正确性测试。

NMSE = ||ghat−g||²/(||g||²+1e−12)；relative L2 = ||ghat−g||/(||g||+1e−12)；cosine = dot(g,ghat)/(||g||||ghat||+1e−12)。各组先单独计算，再对 geometry 三组等权 macro，不能改写成拼接向量指标。

## RECOMPUTED — 汇总所支持的结论

重新计算保存的均值：G4 在 18/18 个 state-budget 条件的 geometry NMSE 低于 G16，16/18 个达到至少 30% 降低；G4/G16 范围 0.414947–0.707107。cosine 在所有条件改善；原 gate 使用 Δ(1−cos)<0，符号正确。该结果是**汇总层面的描述性复算**，不是独立 raw 复现或新置信区间。

报告的 Bias 列错误引用 `geom_rel_l2_mean`（单 mask 误差）。真正保存的 per-group `bias_rel_l2` 是有限 32-seed 的 mean-gradient 对 camera-mean dense gradient 的相对误差；geometry macro 范围 0.0383417–0.2372444。两者均是相对范数单位，不能把 1.3 写成 1.3%。零均值估计器在有限样本下也会有非零 mean-gradient 误差。

保存的 variance = E_camera,mask ||ghat||² − ||E_camera,mask ghat||²，混入 camera 间梯度差异，不等于固定 camera 的 mask variance；加之缺少向量及每 camera 累积量，无法从这些标量恢复条件方差或独立评估 bias 的置信区间。

## RECOMPUTED — Mask 的独立可重现部分

在 MX 上用相同 torch 2.4.1 CPU generator、原源码和 seeds 重建 768 个唯一空间 mask，展开为 2,304 个 state-condition-seed 项，所有 72 个保存的 selected_pixels 均值完全匹配。packed mask bits 和 SHA 已保存。它们明确标为现在重建的 mask，不是原始运行记录。像素边界为 540×810/813/833，已包含非 16 整除尺寸。没有运行 GPU gradient HT/NaN/forward/backward 正确性实验。

## BLOCKED — 不能从现存结果回答的统计问题

不能重算原始梯度、逐 camera/seed 指标、配对差异分布、可靠配对 CI、camera 异质性、固定 camera mask 方差或原始 mask 文件一致性。文件名 RAW、32 的 n 字段以及 SHA 校验只能证明保存文件的身份与一致性，不能证明不存在的记录。
