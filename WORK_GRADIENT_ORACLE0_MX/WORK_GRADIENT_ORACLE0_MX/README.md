# WORK-GRADIENT-ORACLE0-MX — DSH 执行包

**研究命题**：在相同真实 renderer GPU 工作量下，是否能够根据 *梯度信息收益 / 计算代价* 而不是仅凭 pixel error，选出更高价值的 16×16 tile，降低 3DGS 梯度估计误差？

**状态**：`NOT_RUN`。本包是实验规范和可离线验证的数学分析器，不包含任何宣称已经在 MX 实测的结果。**需要转交 DSH 在 MX 执行**；当前 `Giannis` Desktop Commander 为 Offline，因此此处不能直接登录 MX、获取 fresh CUDA timings 或训练数据。

## 文件

- `DSH_EXECUTION_PROMPT.md`：可直接全文交给 DSH 的行动指令。
- `PROTOCOL_LOCK.json`：冻结目标、作用域、实验分组和裁决条件。
- `MATH_AND_PITFALLS.md`：估计器推导、因果边界、成本非加性和重要性采样注意事项。
- `scripts/variance_oracle.py`：接受真实 MX 的 tile-level CSV，计算独立 Bernoulli 近似成本模型下的**解析方差上界对照**。不执行 CUDA，不输出任何真实 GPU speedup。
- `scripts/test_variance_oracle.py`：仅使用 toy 向量与非均匀 tile 成本验证数学公式、边界条件与优化方向。
- `schema/tile_measurements_example.csv`：**无数据，仅有列头**；不准当实测。
- `templates/REPORT.md`：交付证据结构与 BLOCKED/DROP/GO 规范。
- `RUN_STATUS.json`：本地发布状态 NOT_RUN。
- `MANIFEST.sha256`：文件 SHA256。

## 已核对的仓库参考

- `benchmark/run_higs_train_benchmark.py`: `higs_dynamic_ts`、`make_forward_fn(...,tile_mask=...)`、`_error_guided_mask`、`_importance_l1_loss`、`_masked_l1_loss`。
- **重要区别**：现有 `error_guided` 的 `_error_guided_mask` 是 **with-replacement multinomial**，加权采用 `m/(k*p)`。本实验的分析器采用独立 Bernoulli inclusion `z_i~Ber(p_i)`，加权 `z_i/p_i`。二者的公式和工作分布不相同，不得不加说明直接把解析方差当成原实现。
- GitHub 只能作为代码定位线索；实际 MX checkout、JIT 动态库 `.so` 及训练 checkpoint 的身份需运行时核验。

## 一次性转交

将 ZIP 上传给 DSH 并要求逐字执行 `DSH_EXECUTION_PROMPT.md`。DSH 不得在 Stage0/1 缺条件时自行重训、造数据、用 580K initial state 冒充 6M late checkpoint、执行其他研究任务或开发新 kernel。
