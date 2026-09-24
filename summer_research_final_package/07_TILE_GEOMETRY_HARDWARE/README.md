# 07 · TILE GEOMETRY / HARDWARE MECHANISM

## 本目录内容

该目录包含瓦片几何(dile geometry)、occupancy(占用率)、硬件相关(block 形状、寄存器/共享内存/缓存)的全部证据。

## 核心证据文件（原始位置）

| 主题 | 仓库路径 |
|---|---|
| 瓦片尺寸扫描（8/16/20/32 等） | `reports/epic05/hardware-aware-tile-study-2026-08-19.md` |
| 训练阶段瓦片尺寸验证 | `reports/epic05/tile-mechanism-training-study-2026-08-19.md` |
| 梯度正确性与瓦片尺寸 | `reports/epic05/gradient-correctness-tile-size-2026-09-01.md` |
| C19 系列机制实验 | `reports/epic05/phase7c_tile_source_trace.md`、`reports/epic05/phase8*` |
| C19-2 固定几何回放 | `reports/epic05/phase7c_tile_source_trace.md`（受控对照） |
| RTX 5070 Laptop 存档 | `reports/rtx5070/`（c35-c42） |
| C43 自适应瓦片 | `reports/phase-c43*`（可查 `reports/phase-c43/c43_*.md`） |

## 关键结论速览

1. tile size 是 **运行时参数**，同一二进制不变：fwd 输出 bit-identical，梯度仅在浮点精度内变化。
2. 16→20：**占用率坍塌**导致的大洞（int/tile 尖峰）；受控回放证明不是共享内存/寄存器压力（C19-2）。
3. RTX 5070 Laptop：block(32,32) 只允许 1 block/SM（占用率 66.7%），慢 1.91x；A100 2 block/SM。
4. 场景相关性：高稀疏场景 tile16 快、高密度场景 tile32 有优势；没有 universal tile。

## 报告写作提示

- 引用原始 JSON 看原始数字（见 `17_Raw_Data` 索引）。
- 记得区分 `standard tile16`（基线）与 `tile32`（候选）的**kernel 时间 / fwd 时间 / fwd+bwd 时间**。
