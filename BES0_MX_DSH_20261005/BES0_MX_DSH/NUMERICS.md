# 数值与认证规格

## 两条实现线，不混用结论

MATH_REF：FP64 小列表、理想有效 alpha，验证采样分布/期望/梯度恒等式。参考脚本的解析椭圆范围是数学公式的浮点近似，**不是生产 fast-math 的保守性证书**。参考容差不能当作全场景证明。

NATIVE_R0：DSH 冻结并记录 gsplat 1.5.3 实际 kernel 的 alpha 操作顺序、FMA、exp 实现、cap、cutoff、支持、pixel center、排序和 early-stop。首次生产正确性实现用下述离散全枚举认证；更快认证是后续单独实现与证明任务。不能用缺失的 exp 误差界自动判 PASS。

## R0 生产边界（具体、昂贵但可实现）

对每个有效 Gaussian–tile occurrence，在 GPU 上枚举 tile 内每个真实 pixel center（边缘 tile 只含图像内像素），调用与候选点评估 **完全相同** 的 native alpha device function。包含支持测试、cap、cutoff 与相同浮点舍入；NaN/inf/无效 conic 直接 BLOCKED_NUMERICS。

min/max reduction 得到所有离散 alpha 的极值。向下/向上 nextafter 一步后限制到 [0,alpha_cap]；alpha_cap 必须与实际 source 一致且 <1。lower<=0 或 upper/lower>2 -> D。该证书覆盖实际像素，不覆盖任意连续坐标。每次几何/opacity/相机变化都重建；完整耗时、暂存内存计入该 step。不得复用过期表。

这个 R0 可能立即不满足性能目标；这是合法可复现结果。不能把认证成本移出计时来救结论。

## R1 快速认证（若继续开发）

实现严格保守的 interval arithmetic（FP32 向外舍入，逐项覆盖冻结 native FMA/non-FMA 运算），以解析矩形 q 范围或更宽的 interval-box 为起点。必须附 native exp 的可验证误差包围；没有可用包围则该 occurrence 使用 R0 或进 D。不允许经验抽样最大误差替代证书。R1 是新 implementation-id，重新跑 E0/E1/E2；不可把 R0 的 PASS 移植过去。阶段预算不保证能完成 R1。

## hazard 与随机数

R0 hazard/prefix FP64；使用 -log1p(-u)，检查有限且严格递增，不能取消小的正增量。无法解析的 occurrence 改路由 D 后重建表，记录 fallback 数与成本。指数采样使用严格 0<U<1，再 E=-log(U)，二分规则严格大于；参考 RNG 是有限精度近似，不宣称 bit-exact 连续随机分布。

生产 GPU RNG 采用 Philox：counter=(step,camera,pixel,sample,stream,query,trial)，stream A/B 分离，query B-first/B-next 分离；禁止以同一 uniform 同时做跳跃与接受。记录种子/counter layout。浮点 hazard、指数与 prefix 搜索造成的分布误差需要 E1 对照；不能把数学无偏说成机器精度严格无偏。

cap/cutoff/support 的分段导数沿用 native；在阈值边界不做有限差分判定。关闭 early-stop 的 complete-alpha baseline 是理论对照；native 默认 early-stop 额外作为现实性能/质量基线。两种基线分开报告。

## 正确性门

1. MATH_REF E0 FP64 期望图像误差 <=1e-10，期望梯度 maxabs <=1e-9。
2. NATIVE_R0 每个已检查 occurrence 的全像素上下界零违规；任何违规 NO_GO_CORRECTNESS。
3. 同一 alpha evaluator 的完全合成 FP64/FP32 对照：图像 maxabs<=1e-5（归一化 RGB）；否则解释并冻结新容差，不能沿用本 prereg PASS。
4. 扩大 MX 测试到 3 个场景的预注册 views；记录试验数、分布检验与 RNG 流独立性。统计一致不等于证明，但发现违反即停止。

最终文案必须分别陈述数学结果、机器实现容差和实测结果。
