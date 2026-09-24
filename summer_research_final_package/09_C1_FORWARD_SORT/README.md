# 09 · FORWARD SORT / C1（深度键压缩）

## 内容

- `C1_depth_compression.md` — 32-bit → 16-bit 深度键压缩
- `C1_key_encoding.md` / `C1_source_audit.md` — C1 Patch 的编码与源码审计
- `C18_1_incremental_sort.md` — 增量排序（增量开销与重排序收益）
- C17-2 文件夹与 C17-1 详细证据

## 已证伪假设

- **passes 12→8**：来自注释假设，并非实测；`c1_source_audit_final.md` §7 标记为 unverified（CUB runtime policy）。
- **A100 +9.8%**：仅由 sort-isolation 估算，构建失败（nvcc 11.5 vs C++17），未验证。
- **RTX 5070 实测**：32.05 → 31.78ms = **+0.8%**（不到 1%，关闭）。

## 写作提示

务必区分“假设的轮数节省”和“实测的 +0.8%”。报告里引用时只能引用实测值。
