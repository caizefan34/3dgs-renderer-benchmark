---
title: PRISM-GS 首轮实现与实验入口
created: 2026-09-27
updated: 2026-09-27
type: research-prototype
tags: [3DGS, optimization, PRISM, research]
---

# PRISM-GS v0.1

状态：**CPU_REFERENCE_READY / GPU_NOT_RUN**。它不是已经达到 3× 的方法，也不是已确认首创的完整算法。

本包实现「屏幕空间非线性拟合 → 逐高斯三维一致性求解 → 实际三维图像复核」的首轮探针。
保留直接几何 Adam 对照。完整的持久对偶迭代、结构变化、生产 CUDA 求解器和完整训练尚未实现。

## 已有内容

- `prism/geometry.py`：九自由度 SPD 几何参数、五自由度二维椭圆参数、含 FOV clamp 的透视投影、Jacobian、带阻尼与回溯的一致性求解。
- `prism/oracle.py`：仅供数学验证的微型 CPU 渲染器；全支撑 alpha 合成，不与 gsplat 数值等价。
- `prism/cpu_probe.py`：训练视角与审计视角分离的图像级测试。
- `prism/gsplat_backend.py`：真实 GPU 低层渲染适配器，使用原相机 K，改变屏幕参数后重建 tile intersections。
- `prism/run_gpu.py`：EPIC-05 固定环境入口，M0 不通过则停止。
- `tests/test_geometry.py`：Jacobian 有限差分、可实现恢复、残差正交性、可观测性等 7 项检查。
- `DSH_EXECUTION.md`：执行顺序、已知限制、需要补齐的对照及裁决规则。
- `PREREGISTRATION.json`：v0.1 固定阈值与实验边界。
- `results/`：本地原始结果及机器生成报告。

## 本地运行

需要 Python 3.10+、PyTorch 2.x、NumPy。CPU 验证不需要 gsplat。

```bash
python -m unittest discover -s tests -v
python -m prism.cpu_probe --out results/cpu_probe.json
```

GPU 请使用现有固定的 gsplat / fused-SSIM 环境，**不要升级或替换环境来强行使适配器通过**。

## 核心约束

1. 阻尼残差同时包含不良条件和正则化影响，不是「必须分裂」的证据。
2. 图像中的高斯通过 alpha 合成耦合；逐高斯可分的只有一致性子问题。
3. 优化辅助二维参数时固定深度与颜色；回到真实三维后重新计算投影、排序及 SH。该探针是分块近似，不是严格全属性 ADMM 求解器。
4. 所有比较以真实三维模型评估；辅助二维图像不用于宣称质量提高。
5. 首轮只更新三相机共同可见的一小组高斯。其结果不能外推为全场景收敛或端到端加速。
6. `screen_one_step` 仅为消融，不是 LM。直接协方差坐标 Adam 也不能代替原参数 Adam、强训练基线或调优后的 LM。
7. 首个 CPU 测试中，PRISM 改善了审计视角，但直接 Adam 改善更多；如实保留结果。

## 参考来源与原创性边界

读取的 gsplat 接口版本：`512d366b`，必须与远端源码及构建身份进一步核对。

- https://github.com/nerfstudio-project/gsplat/tree/512d366b
- https://lukashoel.github.io/3DGS-LM/ ：二阶优化已有工作。
- https://arxiv.org/abs/2607.14470 ：G²SR 已研究二维 splat 对应关系到三维恢复。
- https://vcai.mpi-inf.mpg.de/projects/SAD-GS/ ：结构感知增密已有工作。

拟研究的贡献是在线非线性屏幕拟合与三维可实现性驱动的结构选择之间的联系。
不能把小矩阵求解、ADMM、三角化、二维变量或多视角增密本身包装成首创。
