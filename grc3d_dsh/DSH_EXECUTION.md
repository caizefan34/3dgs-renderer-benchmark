# DSH：GRC3D-M0/M1 真实场景机制实验

执行对象：DeepSeek Harness。目标仓库 `caizefan34/3dgs-renderer-benchmark`；首选现有 EPIC-05/epic A100 环境。实验包需先解压到独立目录，读取本文件与 `grc3d/README.md`，再执行。

任务目标：证伪或支持「锚定双分辨率梯度校正能降低真实 3DGS 的高分辨率梯度计算需求」。本轮只完成正确性、梯度方差、组件成本诊断，结尾提交 GO / DROP / INCONCLUSIVE / BLOCKED 之一。本轮不启动完整 30K campaign，不改为其他优化课题。

## 0. 冻结与环境

1. 读取实际 repo 的 AGENTS.md、protocol 和最近审计结果。在新工作树/独立实验目录工作，保留旧证据；不覆盖 P11/P12/P13 结果。
2. 优先复用已认证 gsplat `512d366b67073d77ca099ede742683c165dfc23b` 及现有 fused SSIM。先确认当前强基线的精确实现，记录差异；该 commit 只是既有环境标识，不能称全球 SOTA。
3. 记录 repo SHA/dirty diff、源文件和 .so SHA256、导入路径、Python/PyTorch/gsplat/CUDA/nvcc/driver、hostname、GPU 型号/UUID、显存、checkpoint SHA、camera sequence SHA、GT hash、分辨率和所有选项。
4. 使用已有 `ssh EPIC-05` 或 `ssh epic` 配置。若 TCP timeout/不可达，停止重试，报告 SERVER_BLOCKED；只保留离线准备，不使用 synthetic 补真实结果。mx 若是 A100，只称独立 A100 跨系统复现；与 EPIC 结果分表，不混合 geomean。
5. 不占用其他 campaign 的 GPU，不终止无关进程。共享 GPU 只作机制诊断，正式 timing 要独占；matched arms 固定同一物理 GPU，保留污染样本并单独 clean rerun。

## 1. 冻结算法

历史缓存：`A = R_f(theta_old, camera)`，`B = R_c(theta_old, camera)`，均 detach。

当前代理：`P(theta) = clamp(A + bilinear(R_c(theta)-B), 0, 1)`。

分块目标：全分辨率 0.8 L1 + 0.2 (1-SSIM)，RGB mean，11×11/sigma1.5，padding/clamp 以已认证目标为准。每块已经除以完整 H×W，不按抽中像素数重新归一化。

独立 Bernoulli 包含概率 pi 在抽样前确定、detach、严格大于零：

`L_hat = sum_b L_proxy,b + sum_b Z_b/pi_b * (L_fine,b - L_proxy,b)`。

不可把随机图像代入非线性 loss 后宣称无偏；不可删除负代理项；不可强制非空后仍用原 pi。当前实现的无偏梯度不意味着 Adam 更新、二阶矩或 ADC 统计与 dense 相同。

不得混入 SH-PCG/固定几何颜色解算、LFH、新 pruning/opacity reset、masked Adam、其他历史已 DROP 的方案。新颖性属于后续文献审查，DSH 本轮只检验机制。

## 2. M0 正确性，先于真实梯度结论

从解压根目录运行：

```bash
python -m unittest grc3d.test_core -v
```

本包已在 CPU float64 通过四组检查，但必须在实际环境重跑并保留输出。

随后恢复一个真实 bicycle checkpoint，通过现有 model.restore 路径导出 bundle，审计 `grc3d/export.py` 的 raw 参数映射。禁止猜 checkpoint 字段、把已激活 scale/opacity 再激活。

在相同完整相机、当前参数、SH degree、black background、eps2d=.3 下：

- 本包 fine renderer 与强基线 fine renderer 比较逐像素 forward 与六参数组 gradient；本包 absgrad=False 是诊断开关，不能接管 ADC。
- 本包 separable SSIM 对照实际 fused SSIM，比较 scalar loss 与图像梯度，再比较模型梯度。p=1 校正器与 dense 必须一致。
- halo: 只在真实图像边界 padding；覆盖边缘块、非 64 整除分辨率，比较全图/patch 中心 loss 和完整图像梯度。11×11 的 halo=5。
- CPU float64 默认 atol=1e-10/rtol=1e-8；CUDA fp32 预注册 loss abs<=1e-6、image maxabs<=1e-5、每组 gradient relative L2<=1e-4（分母 norm<1e-8 时报告 maxabs<=1e-7）。不得测试后放宽阈值；差异超限先解释数值来源并标记 M0_FAIL/INCONCLUSIVE。

若 baseline loss/window/padding 与本包不一致，在 source audit 后同步修改 fine 和 proxy 的 reference loss，并保留变更证据。若需算法性 CUDA 变更才能构建或复现 baseline，停止并报告 BLOCKED_BUILD_SEMANTICS。

## 3. M1 梯度差，先小规模 pilot

固定矩阵：bicycle、garden；iteration 5K、14K；seed42 的真实基线轨迹；每状态两张由已冻结 camera sequence 预选的训练相机；anchor lag=0、16、64 个 optimizer steps。相机先选定记录，不根据 kappa 挑选。相同相机分别渲染历史/当前模型；历史状态可 topology 不同，缓存无需 Gaussian index 对应。

必须来自真实历史状态/精确 replay，不用参数随机扰动模拟 lag。缺失 checkpoint、optimizer 或 camera 状态则标记 STATE_BLOCKED；不要拿30K状态冒充5K。SH degree 在每组历史/当前 proxy 诊断内固定为当前合法 degree，记录与真实缓存策略的差异。

主配置 q=1/4（高宽各1/4）、block中心64×64。首次 pilot：两场景各5K、各一相机、lag16、每图8个预先随机选择块，确认显存/耗时后再扩大到每图32块及整个矩阵。诊断累计 GPU 占用20分钟仍不能跑完 pilot 时停止扩大并报告成本；已有结果仍全部交付。

```bash
python -m grc3d.probe --backend gsplat --bundle inputs/bicycle_5k_cam0_lag16.pt --coarse-div 4 --block-size 64 --max-blocks 32 --timing-repeats 20 --output outputs/bicycle_5k_cam0_lag16.json
```

对照：dense fine、无锚点 coarse proxy、anchored proxy；用同一组中心块/状态测量。

输出每块、每参数组 xyz / log-scale / quaternion / opacity-logit / SH-DC / SH-rest 的：

- `sum ||g_f,b||²`，`sum ||g_f,b-g_p,b||²`，kappa，raw-coarse kappa；报告近零分母，不把 None 当0。
- p=1/2、1/4、1/8 的解析方差 `(1-p)/p * sum ||difference||²`。采样块只提供能量比估计；保存块坐标与seed，给按块 bootstrap 区间，不能叫全图精确 kappa。
- 若能读取历史 Adam exp_avg_sq，额外报告固定历史预条件度量下的组内 kappa；缓存状态和 epsilon 必须记录并 detach，禁止把它说成完整 Adam 收敛证明。

预注册判断针对「anchored p=1/8 对比 plain p=1/4」：方差比=7*kappa/3，所以 kappa<=3/7 才不更差。报告每个状态及组的比值，不只总 raw-unit 数值。

若 q=1/4 在两个场景 lag16 都至少有一个非退化组 kappa>=1，且该结果的抽样区间明确高于3/7，主配置判 DROP。只允许一个预注册补救：q=1/2，重复同一 pilot；仍失败则结束此候选。本轮不继续搜索新代理、训练超参或采样策略。

GO_TO_KERNEL_PROTOTYPE 需要：M0通过，两个场景 lag16 每个非退化参数组 kappa 的95%区间上界<=3/7，且下述成本预算可行。lag64失败只表明缓存寿命较短，必须把对应 refresh成本计入；lag0失败说明当前代理 Jacobian 本身不足。区间太宽或状态不完整报告 INCONCLUSIVE，不以有限 pilot 宣称最终成功。

## 4. 成本测量与下一步边界

本包 `reference_step` 是 fine全图渲染后取块，`probe` 是全图重新反向的诊断工具，不具备训练加速能力。不得用其总耗时声称 sparse 性能。

记录：强基线 forward/backward/loss/optimizer/ADC 占比，coarse分支，整图proxy SSIM，anchor生成/传输/存储，粗细两套投影与排序，selected中心+halo的唯一像素及tile比例、depth complexity。首次编译与steady state分开，warmup后原始timing逐次保留。

当前公开 gsplat wrapper 重复投影，参考SSIM未融合；若它过慢，报告未优化组件成本，不称理论性能上限。不允许把计划中的共享投影、weighted fused SSIM、sparse raster节省写成已测值。

成本模型以同机强基线完整step为1：

`T_ratio = eta * [(1-delta)*(u+c+(1-u)*r+h)+delta*d] + zeta`。

u=fine基线固定工作占比；c=完整proxy分支；r=精细可变工作剩余比例（含halo）；h=缓存/采样；delta=dense校准比例；d=dense校准含更新缓存成本；eta=等质量步数倍率；zeta=额外初始化分摊。未实测变量必须 HYPOTHESIS，给区间，不代入乐观单点作为结论。

M1阶段 eta 未知。只有在 eta=1.2 时仍有至少1.3×条件速度空间、且新增稀疏/共享内核所需开销有明确测量路径时，才建议 GO_TO_KERNEL_PROTOTYPE；否则 INCONCLUSIVE_COST 或 DROP_COST。这个1.3×门槛是本轮研发决策标准，不是理论保证。

本轮到此停止。生产内核必须先实现完整相机下的 sparse tile/halo；简单更改 crop K 与宽高可能改变 FOV covariance clamp、culling、tile边界和深度排序，不能默认与全图等价。必须逐像素、逐参数组验证后才可做训练。

下一阶段完整训练还需：原生 dense Adam、dense-only ADC校准（与相同ADC调度控制组及原始baseline分别比较）、真实cache policy、统一评估协议、配对多场景多seed、time-to-quality和全程wall time。此次不提前启动。

## 5. 强制交付

保留 `manifest.json`、source diff/hash、export映射审计、checkpoint/camera/GT身份、M0日志、逐块梯度能量JSON、每次timing原始CSV、环境/占用记录、所有失败与重跑、最终 `GRC3D_M1_VERDICT.json` 与报告。

报告按 `OBSERVED / EVIDENCE / HYPOTHESIS / TEST / CONCLUSION` 分段；每条实测结论链接原始文件。结论注明 CPU/GPU、synthetic/真实场景、oracle/强基线、完整块/抽样块、固定状态/完整训练。不把“估计器正确”“梯度方差降低”“训练变快”“击败最强3DGS”混为一项。
