# HIGS_R15_CAPTURE_ORACLE_MX — 实际源码驱动的最小诊断包

本包来自对上传源码 `HIGS_SOURCE_HANDOFF_ACTUAL.tar.gz` 的检查。

**核心改变：不修改 `.cu`，不重新编译 CUDA。** `gaussian_inference.py` 的 `_native_forward_capture()` 已经获取 `means2d/radii/depths/conics/opacities`，随后动态导入 `gsplat.cuda._wrapper.isect_tiles`。用临时 Python wrapper 在这一步捕获输入，并对其中 1K 个真实 projected Gaussian 调用原始 CUDA 算子获取 per-Gaussian counts；随后由 CPU 数学 Oracle 独立计算。

## MX 上一次性运行

将本 ZIP 解压到 MX，执行：

```bash
cd HIGS_R15_CAPTURE_ORACLE_MX
CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh \
  ./reference/R8_WORKING_ENTRYPOINT.py \
  /mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX
```

这里的 `reference/R8_WORKING_ENTRYPOINT.py` 是用户归档中的原版脚本：580K 确定性子集、identity camera、540×960、B1；**不能等同于 R3 的实际场景相机**。

如果 R3 真实重现脚本已恢复，可用它替换第一个位置参数（推荐）。

多 GPU 对照（独立运行，不与单卡结果混用）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_on_mx.sh \
  ./reference/R8_WORKING_ENTRYPOINT.py \
  /mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX_8GPU
```

仅当全部设备均可安全使用时执行。不要杀掉其他用户任务。可通过 `HIGS_ORACLE_N=256` 减小采样量。

重要：诊断模式在成功写入 `captured_projected_inputs.pt` 后 **有意退出**，不会输出完整渲染画面，也不计算 backward。这是诊断模式预期行为，而非 renderer PASS。

## 输出

- `captured_projected_inputs.pt`：采样 Gaussian 的真实 projected inputs 和原 CUDA per-primitive counts；保存于 CPU。
- `captured_projected_inputs.json`：形状、tile 网格、数据量、dispatch key、编译二进制映射（Linux）与 hash。
- `oracle/count_comparison.csv`：每个 camera × Gaussian 的 GPU count / 独立几何 count / difference / AABB upper bound。
- `oracle/oracle_summary.json`：校验统计和谨慎的诊断结果。
- `oracle/first_mismatch.json`：第一处数值差异，如存在。
- `capture.log` 和 `oracle.log`：原始日志。

## Oracle 常数与语义

脚本从 MX 原始 `gsplat/cuda/include/Common.h` 尝试提取 `GAUSSIAN_EXTEND` 和 `ALPHA_THRESHOLD`，若找不到会明确报错。可人工查看 **实际源文件常数** 后使用 `oracle_higs.py --extend <...> --alpha <...>`；不要根据本归档自行假定常数。

Oracle 通过凸二次型在 tile 矩形上的最小值判断椭圆相交：对矩形内部、四条边的 constrained optimum 求最小 q。它与 CUDA AccuTile strip-count 算法**计算策略独立**。边界和精度差异需要二次核查。GPU count > image tile 数量才是物理上界问题（需要确认记录语义），GPU count > padding 1px 的半径 AABB 上界则是较强的实现疑点。

## 工程限制

- 当前本地环境无 NVIDIA GPU，因此 `run_on_mx.sh` **未在 MX 上实测**。本地只运行了 CPU synthetic fixture 和代码单元测试。
- 当前工具通过 Python wrapper 采样已投影的真实输入；**不修改原 Gaussian state**，但重新对样本调用原 CUDA op，并不能自动证明全量 metadata 正确。
- 若真正的问题只发生在全量 n_isects prefix-sum、int32/uint32 溢出或多卡合并阶段，采样 oracle 可能完全通过；后续需要对那个阶段做专用安全 count-only 取证。
- 不要把报告里的 `COUNTING_BUG_CONFIRMED` 作为既定事实，也不要用简单 AABB Oracle 与 AccuTile 的边界差异断言 bug。
- 只应加载自己已核对的研究 checkpoint / source，不执行来源不明的第三方 Python 脚本。

验证运行：

```bash
python3 -m unittest discover -s tests -v
```

**结论门控**：只有真实 `captured_projected_inputs.pt`、CUDA `gpu_counts` 和 CPU `count_comparison.csv` 都产生，才称为 `ORACLE_EXECUTED_AND_COMPARED`。本地 synthetic 测试不满足该条件。
