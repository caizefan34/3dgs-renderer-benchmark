# 生产实现拆解

本包不是已实现的 CUDA 插件。以下给出 DSH 完整实施边界；E0 数学代码在 scripts/bes_reference.py。

1. 冻结 adapter：读取 checkpoint 与 camera/GT，调用原 projection/binning/sorting，保留 active SH 与 opacity activation，得到每 tile 深度有序 occurrence 列表。
2. Native alpha device function 由原源码抽取为共用函数，原完整合成、认证 kernel 与 candidate evaluation 调用同一份。独立比较原原始 kernel 与 wrapper；使用相同 CUDA flags，禁止一个 fast-math、另一个精确 math。
3. R0 bound kernel：每 occurrence 枚举 tile pixel center，warp reductions 得 min/max，nextafter 外扩，gamma 路由。R1 可替换但必须独立 source-level 数值包围证书。
4. stable partition 产生 R ordinal -> global depth ordinal 与 D ordinal -> global depth ordinal；保留 sorted native ordering。tile prefix 用 segmented FP64 scan，严格递增审计。失败项转 D、重新建表。
5. 每像素三 next-hit query：A；B；B 之后 J。各自 prefix 二分，独立 jump/accept uniforms，拒绝后 start=R_ordinal+1。背景用 sentinel N。不能读取一个 GPU tile 的 shared list 后跨 tile 共享接受结果。
6. 将三个全局阈值排序，D 顺序扫描到最大阈值或结尾，取得三个 E/T snapshot。H/J 本身属于 R；无需插入 D。colors 在当前 view 计算，背景 sentinel 正确。
7. backward：先以 Torch AD 源码实现 surrogate 精确对照，再手写 CUDA VJP。D 的 direct term 用原 compositing backward；H 的 color 与 score term 使用选定 native alpha 的 chain rule。计算 d_alpha = stopgrad(r·(F_H-F_J))/alpha_H，避免先 log 再数值截断改变目标；alpha_H>0（只接受正 alpha）。不允许 epsilon clamp 扭曲 score 而无独立偏差报告。流 A 的计算 detach 后不反传。
8. 动态几何/opacity/R-D 改变每步重建，分类和表 detach。参数在本步的分段区域内取导数，阈值/排序边界延续 native 约定。完整训练 densification 属于后续版本。
9. instrument 各阶段时间与 byte allocations；同时记录真实墙钟，JIT、prewarm 单独报告，预算内收费。内存包含排序/边界/prefix/rng/反向缓冲。

不同 train samples 可分别执行三查询和 D 扫描，首版不强制跨样本共享扫描优化；不得声称已享有这个优化。A100 SM80，prefix 双精度与 binary-search warp divergence 都是待测成本。
