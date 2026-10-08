# PRISM-GS v0.1 — DSH 执行指令

目标：在投入完整训练前验证 PRISM 的几何机制。现有 GRC3D DROP 不变，不追加 GRC3D 实验。
本包不能被表述为三倍加速实现。先检查身份与 M0，再运行小型 GPU 探针。

## 0. 隔离与预检查

解压到新的 `/root/epic05_prism/`，不改动原训练器、固定 gsplat 树或 GRC3D 原始证据。
核对 `MANIFEST.json`。使用 EPIC-05 已固定的 Python/CUDA/gsplat/fused-SSIM 环境。
保留机器、GPU UUID、进程占用、驱动、torch/CUDA、源码 commit、扩展 so SHA256。
共享 GPU 可以做功能检查，不能据此宣布加速。无需重训或再次导出 GRC3D 的代理数据。

复用真实原始 snapshot：`xyz / rotation / scaling / opacity / shs / active_sh_degree`。
不要归一化后覆盖 snapshot 的原始 rotation；真正渲染时使用原训练器的 `F.normalize(raw_rotation)`。
复用冻结的 TRAINING camera sequence，不能混入最终测试图像。

可从现有 GRC3D 的 manifest 找到 snapshot 和 camera sequence 的真实路径，禁止猜路径后改成另一个状态。

## 1. CPU 检查

在解压后目录运行：

```bash
python -m unittest discover -s tests -v
python -m prism.cpu_probe --out results/cpu_probe_remote.json
```

7 项数学检查必须通过。CPU 数值仅用于 correctness，不据此比较 A100 的速度。

## 2. 独立原训练器等价检查（必须）

本包的 M0 是适配器内部等价检查。DSH 还必须用原训练器自身的真实渲染/损失入口，在相同 snapshot 和相机上检查：

- raw rotation → 原训练器预归一化 → gsplat 的完整链路；
- same K / 分辨率 / eps2d=0.3 / black background / packed=False / tile16 / classic RGB / absgrad=True；
- 相同 SH degree、opacity 激活、image clamp、fused L1+SSIM 与 padding；
- 六组原始参数的 forward/backward 差异。

保存原始数组或可重建这些指标的原始结果，分别记录固定容差和 5 次原入口重复的噪声地板。
不准通过修改原训练器作为对照来使适配器通过；不准在看到结果后扩大固定阈值。
超过固定阈值但落入重复噪声范围，标记 `INCONCLUSIVE_NUMERICS`，不要自动视为 PASS。

## 3. GPU M0

以下命令中的路径由 DSH 从真实 manifest 填入；`gx-path` 指向已有 `gx_common.py` 所在目录。
`gx_common` 使用的固定 fused-SSIM so 校验会失败关闭，不绕过。

```bash
python -m prism.run_gpu \
  --gx-path /root/epic05_grc3d/remote_scripts \
  --gpu 0 --scene bicycle \
  --snapshot /ABSOLUTE/REAL/snap_bicycle_005000.pt \
  --camera-seq /ABSOLUTE/REAL/camera_sequence.npy \
  --m0-only --out results/bicycle_5k_m0.json
```

程序对前三个训练相机逐一检查：

1. 原高层渲染与低层直通渲染；
2. 六组梯度；
3. 零位移时重新生成 bounds/intersections 的渲染；
4. quaternion/scale 到 covariance 参数化的渲染差异；
5. CPU 公式在 GPU float64 下的投影 vs CUDA 投影。

**所有结果只要有 BLOCKED_M0 就停止该状态。** 原始输出不得删掉。
失败属于接口/数值问题，不是 PRISM 机制的科学证伪。修复需有单独版本号、patch 和阈值不变的重跑。

## 4. 首轮真实图像探针

独立原训练器等价检查及 GPU M0 都通过后，在相同命令去掉 `--m0-only`。
首轮固定 bicycle/garden × 5K/14K × seeds 20260927/20260928/20260929，最多 12 个状态-种子单元。
先跑 bicycle 5K 的一个单元检查资源和输出，然后再批量。

相机选择：冻结训练序列的前五个不同 ID；前三个优化，后两个仅作审计。
共同可见高斯中按 seed 均匀选至多 512 个；其他高斯固定，但完整图像仍参与渲染和损失。
这个选择可能缺少视差。记录位置 Jacobian 的秩/奇异值和视角几何，秩亏时记 `INCONCLUSIVE_OBSERVABILITY`。
不要事后挑相机让结果好看；任何基于几何的替代规则需先单独登记。

现有三个臂：

- `screen_one_step`：一次局部屏幕更新后求解几何，只作线性/非线性消融；不是 LM。
- `screen_nonlinear`：三次局部屏幕更新后求解几何；每次都是完整图像真实反传。
- `direct_geometry_adam`：相同几何坐标的三次直接三维 Adam；同样更新所选高斯。

三步两个臂各有 9 次训练视角反传，另有全部真实评估开销。一步臂预算不同，不直接宣布谁更快。
当前 v0.1 的执行顺序固定，仅用于功能探针；正式时间比较必须平衡/随机化臂顺序，预热并记录初始化/JIT。
程序报告 `shared_setup_seconds`、优化时间、含评估时间及显存；每个臂的实际成本需加共享准备成本。
不得把相同起点评估/准备的成本漏掉，也不得将 CPU 小矩阵时间当成免费。

## 5. 进入 M1 正式比较前必须补齐

v0.1 不会自动裁决 GO。DSH 必须单独报告以下缺项，不能拿探针结果代替：

- 原参数空间 Adam 对照：使用冻结配方的 xyz/scaling/rotation 学习率及该状态的 optimizer state；相同活动集合。
- 多视角原参数 Adam 与合适的 LM / 二阶对照，记录调参预算。禁止把 `screen_one_step` 命名为 LM。
- 全量或分块扩展后的真实成本，不能把 512 个高斯的结果当成全量证明。
- 在等墙钟预算上的真实三维 loss 轨迹和审计相机表现；失败 proposal、回溯、重新投影/排序、评估均计入。
- 至少三个配对 seed；场景/状态分别汇报，不能只报最好一个；选择预算后不得事后更换。

初步工程筛选：若非线性屏幕臂在两个场景均未呈现真实三维收益，或收益被同预算直接优化完全覆盖，标记 `HOLD_NO_MECHANISM_ADVANTAGE`，不开展完整训练。
若通过上述检查，则结果只能叫 `GO_TO_M2`，不能叫三倍加速。
单次局部损失下降更快不能外推到收敛速度；未显著优于对照也不自动等价于科学上永久 DROP。

## 6. 后续阶段（本包未实现）

M2：持续对偶状态、可实现性残差与可观测性分离、基于像素残差的结构候选生成，真实审计视角的收益/成本裁决。
不要简单用大残差触发分裂；不可只匹配二维均值/协方差来论证分裂收益。

M3：同硬件、同质量目标、同初始化比较最强可复现的训练基线。所有新增预处理及精修成本计入。
预先登记 PSNR/SSIM/LPIPS 质量容差，比较达到共同质量目标的时间；基线没有达到质量目标时不能虚构速度比。
目标为 T_new/T_best < 1/3。只有完整训练证据才允许声称超过三倍。

## 7. 回传

交付原始 JSON、完整日志、源码 patch、环境占用、manifest 和逐状态结论。
报告按 OBSERVED / EVIDENCE / HYPOTHESIS / TEST / CONCLUSION 分段。
必须包括失败及负结果。本地首个 CPU toy 中直接 Adam 优于 PRISM，不能从报告删去。
不提交 git、不修改旧裁决、不启动昂贵完整训练，直到上述机制门槛有证据。
