# DSH 实施任务单

把本包视为执行授权与冻结研究规格，用户会自行转发。不需要额外 PI 设计输入，也不授权外部发布/发送结果。工作限定 MX 与 BES0 新目录；不修改旧 benchmark 的默认行为。

## 必须按顺序交付

E0：通过现成脚本检验 1) 原图像的期望相等，2) 双事件 score 与 D 导数的期望梯度相等，3) prefix 跳跃分布，4) gamma 查询界，5) 错误 hazard-ratio 接受的反例。失败先修公式/实现，版本变更留痕，不能继续生产。

E1：实现 actual-native alpha wrapper、R0 全像素 bound kernel、R/D 排序表、FP64 hazard prefix、Philox event query、D 三快照。对 room 历史 checkpoint 重新核对；登记另外两场景现成 checkpoint/相机/GT。输出 bound violations、路由工作量、构建/查表/D 扫描实测成本、乐观 3x 可行性。R0 不够快可输出 NO_GO_OPPORTUNITY；R1 仅在预算内开发，按 NUMERICS.md 独立认证。

E2：实现 METHOD.md surrogate backward，包含所有 native 活跃参数。用 complete-L2 baseline 对照梯度均值/方差；正式计时包含数据取用、projection/sort、每步 bound/prefix、候选/D、backward、optimizer。对 native default L1+SSIM 另外报告。pilot 不训练相机和背景，固定 N，关闭 densification/opacity reset，避免未测子系统混入。保存每步 wall time 与设备各模块时间（模块时间不能直接相加替代墙钟）。

E3：选定配置在 3 个场景、种子 17/29/43、最多 1000 step 的 fixed-N pilot。使用共同起点 checkpoint，同 seed/order 的所有方法配对；以预注册质量门槛计 time-to-quality。如果预算不够覆盖全部 pair，结论 INCONCLUSIVE，不用补少量好看的场景替换。

渲染评估作为独立附项：spp=1/2/4/8/16，总样本时间含建表、取样、D、平均，禁止免费 TAA/denoiser；同质量的收益不能从训练速度推出。

## 禁止替换目标

- 不把更少 primitives、降分辨率、低 SH、阈值剪枝或省略 score 项作为 BES0。
- 不把 alpha 查询上界、Torch microbenchmark 或单步速度当收敛加速。
- 不把已有 StochasticSplats / dual-hit / Gaussian Stippling 等工作说成本方案首创。
- 不把固定 N pilot 推广为完整 densification 系统或全场景通用方法。
- 不修改预注册门槛后保留原 PASS；变更必须另一个协议 hash 与 run-id。

返回：结果 ZIP + 代码 patch + 简明报告。即便 NO_GO/BLOCKED，也交付原因、缺失项和可核查日志。
