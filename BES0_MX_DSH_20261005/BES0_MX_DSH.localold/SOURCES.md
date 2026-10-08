# 原始工作与新颖性边界

本列表来自此前提出BES-GS时的检索，冻结到本包；不是穷尽的专利/论文查新。DSH实现前复核源码版本。不得把数学部件的重新组合当作首创证明。

- 3D Gaussian Splatting (2023)：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ — 原始目标与生态。
- Stochastic Ray Tracing for Reconstruction (2026)：https://arxiv.org/html/2603.23637v1 — dual-hit gradient 的重要先例；本包score项不是首创。
- StochasticSplats (ICCV 2025)：https://arxiv.org/html/2503.24366v1 — 随机透明度训练，独立随机流与L2目标问题。
- Gaussian Point Splatting (2026)：https://www.momentsingraphics.de/Siggraph2026.html 及 https://jorisar.nl/gaussian_point_splatting/ ，代码 https://github.com/JorisAR/gaussian-point-splatting — 像素事件采样/点泼溅相关先例。
- Gaussian Stippling (2026)：https://arxiv.org/html/2609.38488v1 — footprint/opacity路由、两种随机机制与重建；不能把“hybrid routing”本身当首创。
- Hybrid Transparency (2024/2025)：https://arxiv.org/html/2410.08129v2 — 精确头部与近似尾部的相关路线；不同于BES的任意交错R/D条件期望。
- Don't Splat your Gaussians (TOG 2025)：https://doi.org/10.1145/3711853 — 不同体密度模型与相关采样；majorant/null-collision是已有思想。
- gsplat 1.5.3文档：https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html — 生产语义需实际MX source审计，文档不代替kernel。

候选贡献只可表述为ratio受限的occurrence路由、前缀majorant事件跳跃、条件积分D及共享快照训练的具体系统组合，并用实现/实验与上述工作逐项比较。首次检索无完全相同方案不构成“无人做过”的证明。
