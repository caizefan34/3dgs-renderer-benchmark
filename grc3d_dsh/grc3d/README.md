# GRC3D 原型实验包

目标：检验「锚定双分辨率梯度校正」是否值得进入真实 3DGS 稀疏 CUDA 开发。
当前是参考实现与可复现实验入口，未宣称原创性已获确认、训练加速或击败 SOTA。

请先阅读根目录 `DSH_EXECUTION.md`。从包含 `grc3d/` 的目录执行命令。

```bash
python -m unittest grc3d.test_core -v
python -m grc3d.probe --output outputs/oracle.json
python -m grc3d.probe --backend gsplat --bundle inputs/bicycle_5k_cam0_lag16.pt --block-size 64 --max-blocks 32 --timing-repeats 20 --output outputs/bicycle_5k_cam0_lag16.json
```

依赖：已有环境中的 PyTorch；GPU 路径额外要求已有 CUDA gsplat。CPU 本轮测试版本为 PyTorch 2.14.0+cpu / Python 3.12；这不是要求升级远端环境。不要为了此包修改已冻结的基线依赖。

文件：

- `core.py`：detached 锚点、全局归一化 L1+SSIM、Bernoulli 校正、5 像素 halo、全图精细参考 step。
- `renderers.py`：六 Gaussian 的 CPU 3D oracle（含透视投影、3D covariance、quaternion、SH3）、公开 gsplat API 适配器。
- `probe.py`：按参数组累加 fine/difference 梯度能量；保存逐块数据；粗代理耗时诊断。
- `export.py`：从已恢复的原生训练模型导出规范 tensor bundle。
- `test_core.py`：枚举全部 Bernoulli outcomes 验证期望和方差；halo 正反向；p=1 与 dense 一致；3D finite difference。
- `results/`：本轮 CPU 原始输出，仅是机制证据。

## 原生 checkpoint 接入

先由原来的 trainer/模型代码恢复 checkpoint。`snapshot(model)` 映射 Graphdeco 风格 `_xyz/_rotation/_scaling/_opacity/_features_dc/_features_rest`；DSH 必须读实际 source 验证字段与激活，不可盲目套用。

在基线真实训练轨迹保存 historical/current 两次 snapshot；相同相机、GT、SH degree 交给 `save_bundle`。例如：

```python
from grc3d.export import snapshot, save_bundle
# 在原生 trainer 中，historical_params 已由更早 iteration 的 snapshot 得到。
current_params = snapshot(model)
save_bundle("inputs/bicycle_5k_cam0_lag16.pt", current_params, historical_params,
            cam.viewmatrix, cam.K, target_hwc, sh_degree,
            {"scene": "bicycle", "iteration": 5000, "anchor_iteration": 4984,
             "camera_index": 0, "checkpoint_sha256": verified_checkpoint_sha256})
```

上例变量须来自实际 trainer；并非本包自动提供 dataset、checkpoint 或 SSH。`probe` 使用 `torch.load(weights_only=True)`；metadata 请只用字符串、数字和普通容器。

## 解释输出

`kappa = sum_b ||g_f,b-g_p,b||² / sum_b ||g_f,b||²`。完整 block 集合才是精确总和；`--max-blocks` 非零时是均匀抽样块上的比值估计，不是全图精确值。原始单位汇总可能被少数参数组主导，必须同时看每组和近零分母。模型越大，逐块 gradient diagnostic 越贵；计入诊断预算。

`variance_ratio_p0125_vs_plain_p025 = 7*kappa/3`，只比较两个 Bernoulli 稀疏估计器。不能与零方差 dense gradient 等同；更不说明 Adam 轨迹一致。

`proxy_only_forward_backward` 包括低分辨率 renderer、锚点合成、全分辨率未融合 SSIM 与 backward；不包括精细校正、anchor refresh、optimizer、ADC。它用于发现粗分支本身已过贵的风险。该实现会重做投影，不具备生产版共享投影；不得把其值当作最优实现的理论下界。

## 当前边界

精细分支仍全图渲染再取 loss blocks；没有 sparse tile CUDA、共享投影 backward 或高性能 weighted fused SSIM。`render_reference` 无 gsplat 的 tile cutoff/early termination/FOV covariance clamp，不能作后端对等验证。GPU wrapper 固定黑背景、classic、eps2d=.3、packed=False；absgrad=False 只适用于固定状态诊断。正式训练必须恢复原生 absgrad/ADC 语义。

先通过真实后端 loss/gradient parity，再解释真实场景 probe。未融合参考 SSIM 与用户当前 fused SSIM 的一致性尚未在 A100 上验证。

API 依据（实现时查阅）：https://docs.gsplat.studio/main/apis/rasterization.html
