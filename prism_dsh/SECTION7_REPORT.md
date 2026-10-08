# PRISM-GS DSH 执行报告 — Section 7 回传

| 项 | 值 |
|---|---|
| 输入包 | `PRISM_GS_DSH_v0.1.zip`，sha256 `6acd04e9e84361880a257d87123b60585f74d52c609ecab31938da3581473826` |
| 包内 MANIFEST | `MANIFEST.json` v0.1.0，15/15 条目 sha256 与字节数全部匹配（0 处不符） |
| 主机 | `dsw-933622-64cc845c7d-lqswc`（8× A100-SXM4-80GB，driver 570.133.20） |
| GPU | 仅 GPU 2 = `GPU-bcca92b9-0234-7ec9-a2f0-a63925d52146`；GPU 0/1 全程被另一 DSH 占用（32757 MiB），未触碰；GPU 3–7 未使用 |
| 运行时 | `/root/miniforge3/envs/gsplat/bin/python`，Python 3.10.21，torch 2.9.1+cu128，CUDA 12.8 |
| 固定渲染器 | `/root/epic05_p8/gsplat-main-512d366b`，pin `512d366b67073d77ca099ede742683c165dfc23b`，扩展 so sha256 `8ca8ec1e0c8618f058638b93010061a5bafb7f6c92005b54d07dc6c22658c811` |
| 固定损失 | `fused_ssim_cuda*.so` sha256 `dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb`（P11 lock），`lambda_dssim=0.2` |
| 执行日期 | 2026-09-27（全部时间戳 +08:00） |
| 最终标签 | **HOLD_NO_MECHANISM_ADVANTAGE**（首轮筛查）；未给出 `GO_TO_M2`；无加速声明 |
| 昂贵完整训练 | 未启动 |
| git | 未提交任何 commit；`/root/3dgs-renderer-benchmark` HEAD 仍为 `abfa6c3e4b23caf5a927bf3722a65cba37b6fab3` |

---

# OBSERVED

**O1 环境隔离成立。** 解压到新目录 `/root/epic05_prism/`；`/root/epic05_p8/`（渲染器）、
`/root/epic05_grc3d/`（输入与脚本）、`/root/epic05_p12/`（训练器/相机序列）在最近 60 分钟内
无任何文件写入（`find -newermt` 为空）；赛事仓库工作树只有既有的未跟踪数据目录。
另一 DSH 的 GPU 0/1 占用（32757 MiB）在本次执行前、中、后完全不变。

**O2 输入身份与原始参数。** 4 个 frozen snapshot 与 2 个 frozen TRAINING 相机序列按 sha256 复用，
未做归一化覆盖，渲染时用原训练器的 `F.normalize(raw_rotation)`：

| 输入 | sha256 | 备注 |
|---|---|---|
| `snap_bicycle_005000.pt` | `0f924d0acdc3d8301c20fa8434960f51aeda07d6185091e82d62879d5eb1d450` | 1,879,621 高斯 |
| `snap_bicycle_014000.pt` | `48888734e86cd9b917b542e1464ca919f3de0132e7484cc8223f20211bc83bdb` | 3,573,165 高斯 |
| `snap_garden_005000.pt` | `e5bb8c9590b6ec75e86a77d5e2f6bfa6dfdeb343e62f4899af5d264e2d4deea4` | 1,813,591 高斯 |
| `snap_garden_014000.pt` | `31f8a4a850b57ee44b0ba9427904569af495ec8f96897e48a39242567c6f7f60` | 2,922,031 高斯 |
| `p12a_gmainf_s42_bicycle/camera_sequence.npy` | `8debe1c2244aca0994e9bd27ddfdd440fcc60f51341351d6c899b3c637c78e2d` | len 30000，前 5 个不同 ID `[163,28,6,192,72]` |
| `p12a_gmainf_s42_garden/camera_sequence.npy` | `63ad2de2256d09056d18d6e0cda4fcedb0935de81fca3dcd5ffa146a9d385cc5` | len 30000，前 5 个不同 ID `[163,28,6,72,64]` |

**O3 CPU 检查通过，且 CPU 负结果原样保留。** v0.1.1 与 v0.1.2 均 7/7 `OK`
（`results/unit_tests.log`，2.65 s）。`cpu_probe.json` 两个版本的数值除墙钟字段
`proposal_cpu_seconds` 外**逐字节相同**：审计视角相对下降 PRISM 49.53%、直接 Adam 55.55%
——**本地 toy 里直接 Adam 仍然更好，没有 PRISM 优越性的实测证据**。

**O4 Section 2 独立原训练器等价检查 4/4 通过。** 参考路径 = 冻结 P12/GMAIN 训练器自己的入口
（`GaussianModel` 激活 + pinned gsplat + fused L1/SSIM），适配器路径 = `prism.gsplat_backend.baseline`，
每个相机重复 5 次原入口以测噪声地板；容差取自 `PREREGISTRATION.json`（与 MANIFEST 哈希绑定，
未在见到结果后调整）：

| 状态 | 前向 maxabs | 梯度 relative_l2 最大项 | 判定 |
|---|---:|---|---|
| bicycle 5000（相机 163/28/6） | 0.0 | quats 3.73e-4、log_scales 6.0e-5、其余 ≤9.3e-7 | `PASS_PARITY` |
| bicycle 14000 | 0.0 | quats 2.47e-4、log_scales 3.1e-5、其余 ≤5.7e-7 | `PASS_PARITY` |
| garden 5000 | 0.0 | quats 5.0e-5、log_scales 1.3e-5、其余 ≤6.2e-7 | `PASS_PARITY` |
| garden 14000 | 0.0 | quats 3.89e-4、log_scales 1.7e-5、其余 ≤4.0e-6 | `PASS_PARITY` |

固定阈值 `adapter_forward_maxabs 2e-6`、`gradient_relative_l2 1e-3`、`projection_* 2e-3`。
旋转/协方差链 `normalize_maxabs 0.0`、`covariance_maxabs 2.98e-7`。
唯一擦边项：bicycle 5000 相机 163 的 `sh0` 相对 l2 = 4.016e-7，略高于其自身 5 次重复噪声地板
3.988e-7（比值 1.007），但远低于绑定的固定阈值 1e-3 → 按固定阈值判定为 PASS，此擦边已登记。

**O5 v0.1.0 GPU M0 失败（原始输出保留）。** 同输入、同固定阈值下三台相机全部 `BLOCKED_M0`：
`forward 0.0`，`zero_override 4.64e-4 / 1.40e-6 / 2.98e-7`（阈值 2e-6），
`quaternion→cov_render 6.204e-3 / 8.782e-3 / 7.736e-3`（阈值 2e-5），
`sh0/shN 梯度 rel_l2 0.345/0.246/0.382`（阈值 1e-3）。原始文件
`outputs/raw_m0_v010_blocked.json` 未删改。

**O6 根因已定位（接口/数值，非机制证伪）。** 决定性实验 `diag_m0_v7e`：
把 `covars = KS2C(F.normalize(raw_quats), exp(log_scales))` 按渲染器 triu 布局镜像成 3×3 后走
`covars=` 路径，与 `quats=` 路径在 3 台相机上**逐位相同**（gap 0.0，可见 conic 100% 逐位相等，
radii 0 翻转）；而用未归一化 raw quats 计算协方差会复现 6.2e-3/8.5e-3/7.7e-3 的差距。
第二处：v0.1.0 屏幕特征走 log-Cholesky 往返，有损（zero-override 4.64e-4）。
第三处：SH 激活 `clamp_min(0)` 在 0 处包含端点，核内为严格 `>0`，造成 0.345 的相对梯度差。

**O7 v0.1.1/v0.1.2 GPU M0 通过，阈值未改。** 两版 `bicycle_5k_m0.json` 均为
`M0_ADAPTER_PASS_ONLY`，`PASS_ADAPTER` 门值 `forward = zero_override = quaternion→cov = repeat = 0.0`，
`projection_uv_maxabs 1.685e-4`、`projection_shape_maxabs 4.92e-7`，
`sh0/shN` 梯度 rel_l2 3.9e-7/5.0e-7（v0.1.1）与 4.1e-7/4.1e-7（v0.1.2），全部远低于固定阈值。

**O8 Section 4 首轮（v0.1.1）不完整：12 个单元中 5 个 `ERROR`。** bicycle 6/6 完成；
garden 5K seed 27 完成，garden 5K seed 28/29 与 garden 14K seed 27/28/29 在
`pack_geometry` 处抛 `linalg.cholesky: ... not positive-definite`。原始 JSON（含 traceback）与
日志完整保留在 `/root/epic05_prism/prism_gs_v0.1.1/results/`。

**O9 非正定来源是渲染器自身的 fp32 协方差约定。** 用同一 bootstrap 直接调用
`quat_scale_to_covar_preci`（`outputs/diag_pd_garden_*.json`）：

| 快照 | N | 非 PD 行 | 占比 | 坏行最小特征值 | 坏行 log_scale 最小 |
|---|---:|---:|---:|---:|---:|
| garden 5K | 1,813,591 | 259 | 1.43e-4 | −1.16e-9 | −17.11 |
| garden 14K | 2,922,031 | 8,094 | 2.77e-3 | −1.23e-9 | −19.75 |

坏行 `max_symmetry_error = 0.0`（完全对称）、raw 四元数范数 0.43–2.85（无退化四元数）、
尺度最小到 2.7e-9（`s_i*s_j` 向 1e-18 下溢），其最大特征值可达 0.19。即渲染器返回的 Σ 在这些
近下溢行上**失去正定性**（负特征值极小），这是其 fp32 约定的性质：光栅化器从不需要 Cholesky，
所以渲染器本身渲染这些行毫无问题，只有 PRISM 的 9 自由度诊断坐标需要 Σ 正定。
此外这类亚像素行在“共同可见”集合中被**富集**（`radii = ceil(extend*sqrt(Σ2d_00))` 对消失椭圆饱和到 1，
不会被可见性规则剔除），所以 512 抽样的命中率高于全局比例。

**O10 v0.1.2（P4）修复后 12/12 单元全部完成，且修复只触及少数行。**
`geometry_pack` 记录：bicycle 四个组 0 行修复；garden 5K seed 28/29 各 2 行、garden 14K 三 seed
7/7/8 行，12 个单元合计 26 行（每 512 抽样），最大相对抖动 1e-6（即 Σ 的绝对改动约 1e-13 量级），
其余 511/512 行逐位不变。v0.1.2 的 CPU 7/7、`cpu_probe` 数值与 v0.1.1 完全一致、M0 门值不变、
Section 2 四个状态以 v0.1.2 树重跑仍 `PASS_PARITY`（`parity_*_v012.json`）。

**O11 观测性检查未出现秩亏。** 每个单元按位置 Jacobian 的奇异值计算：`min_relative_singular`
范围 0.2098–0.3925，`fraction_below_1e-6 = 0.0`，即 12/12 单元均**不**触发
`INCONCLUSIVE_OBSERVABILITY`。

**O12 逐组结果（512 个共同可见高斯、3 个训练视角优化、2 个审计视角从不优化）。**

| 组 | 臂 | 训练 Δ | 审计 Δ | 含评估墙钟 | 峰值显存 |
|---|---|---:|---:|---:|---:|
| bicycle 5K | screen_one_step | −2.583e-5 | −2.140e-6 | 1.029 s | 2203 MB |
| | screen_nonlinear | −3.179e-5 | −3.935e-6 | 0.345 s | 2203 MB |
| | direct_geometry_adam | −4.420e-5 | +3.480e-5 | 0.287 s | 3724 MB |
| bicycle 14K | screen_one_step | −6.936e-6 | +6.900e-6 | 1.065 s | 3702 MB |
| | screen_nonlinear | −8.321e-6 | +6.260e-6 | 0.402 s | 3702 MB |
| | direct_geometry_adam | −6.638e-6 | +3.990e-5 | 0.387 s | 6443 MB |
| garden 5K | screen_one_step | −1.417e-5 | +7.170e-6 | 1.050 s | 2214 MB |
| | screen_nonlinear | −1.766e-5 | +6.740e-6 | 0.334 s | 2214 MB |
| | direct_geometry_adam | −2.796e-5 | +2.727e-5 | 0.275 s | 3670 MB |
| garden 14K | screen_one_step | −6.111e-6 | +3.713e-7 | 1.069 s | 3114 MB |
| | screen_nonlinear | −7.336e-6 | +7.078e-8 | 0.360 s | 3114 MB |
| | direct_geometry_adam | −5.463e-6 | +3.319e-5 | 0.336 s | 5401 MB |

`shared_setup_seconds` 每单元 0.107–0.137 s（三视角前向 + 状态构建，尚未计入上表），
必须与臂时间相加。反传视角数：`screen_one_step` 3、`screen_nonlinear` 9、`direct_geometry_adam` 9；
评估每臂 5 次前向。逐单元 JSON 见 `SECTION4_TABLE.md` / `SECTION4_TABLE.json`。

**O13 首轮筛查没有出现机制优势。**
* 训练侧：三种臂都只把训练损失压低 0.006%–0.035%（3 步固定学习率、512/1.8M–3.6M 高斯），
  其中 4 组里 3 组是直接 Adam 压得最多；
* 审计侧：直接 Adam 的审计退化在 4 个组里分别是屏幕臂的 4.7×–15.9×（组均值口径），
  12 个 seed 里直接 Adam 从未在审计上胜出（0/12）；但**这是与步长/步数的混杂观测量**：
  两臂的学习率（屏幕 0.03、几何 0.003）与参数化不同，直接 Adam 走的训练步更大，
  “训练步小 → 审计破坏小”本身就能解释该现象；
* 成本侧：`screen_one_step` 墙钟是直接 Adam 的 2.5–3.8 倍（虽然只用 1/3 反传视角）；
  `screen_nonlinear` 墙钟约 1.05–1.20 倍；屏幕臂峰值显存明显更低（约 0.55–0.66 倍）。
  **在任何口径下都没有可声明的加速。**

**O14 没有触碰固定资产、没有提交、没有启动完整训练。** 见开头元信息表；GPU 2 上只跑了
单元级探针（单次约 70 s）。

---

# EVIDENCE

原始 JSON、完整日志与补丁的本地路径（`3dgs-renderer-benchmark/prism_dsh/`）：

| 内容 | 路径 | 数量/规模 |
|---|---|---|
| Section 4 v0.1.2 原始 JSON（12 单元 + M0 + cpu_probe + 单测日志 + 本地裁决） | `remote_results/v012/` | 17 文件 |
| Section 4 v0.1.1 原始 JSON（含 5 个 `ERROR` + 7 个完成 + 其余） | `remote_results/v011/` | 17 文件 |
| Section 2 parity 原始 JSON（v0.1.0 绑定的 4 状态 + v0.1.2 绑定的 4 状态） | `remote_outputs/parity_*.json`、`parity_*_v012.json` | 8 文件 |
| M0 原始输出（v0.1.0 `BLOCKED_M0`、v0.1.1 `PASS`） | `remote_outputs/raw_m0_v010_blocked.json`、`raw_m0_v011_pass.json` | 2 |
| 非正定诊断 | `remote_outputs/diag_pd_garden_005000.json`、`diag_pd_garden_014000.json` | 2 |
| M0 根因诊断链（E1/E0 等） | `remote_outputs/diag_m0_v2..v7e_*.json` | 9 |
| 环境身份 | `remote_outputs/ENVIRONMENT_ISOLATION.json` | 1 |
| 完整日志（sweep、逐单元 probe、M0、cpu_probe、parity、diag） | `remote_outputs/logs/` | 26 |
| 并行证据目录清单（21 GB/84 文件 ×2） | `remote_outputs/inventory/parity_arrays_*.txt` | 2 |
| 补丁与校验 | `patches/v0.1.1/`（`PATCH_v0.1.1.diff`、`MANIFEST.v0.1.1.json`、`prism_gs/`）、`patches/v0.1.2/`（`PATCH_v0.1.2.diff`、`MANIFEST.v0.1.2.json`、`PATCH_NOTES.md`、`prism_gs/`） | 2 树 |
| 执行脚本（可重放） | `scripts/` | 20 |

关键哈希（本地 `MANIFEST.v0.1.2.json` 与远端 parity 记录一致）：
`prism/geometry.py 9e99854453ef35007f74542a44089b1dba94b1959d3711e6b65bf2d7c8a6cf55`、
`prism/run_gpu.py 12f2df808ee5b2f09ed6784be6bb180755d691e599c685421952dabff00941e1`、
`prism/gsplat_backend.py 5cc041e04b99f4df521811f2465751030b088061cc1d4564380b1e2c36b773ee`
（v0.1.1/v0.1.2 相同，P1/P2/P3 未再改动）、
`tests/test_geometry.py 104878d22475e6e2…`。
远端保留（体积过大未回传，清单已回传）：`/root/epic05_prism/evidence/parity_arrays/`（21 GB，84 文件）、
`/root/epic05_prism/outputs/parity_arrays_v012/`（21 GB，84 文件）。

**逐状态结论**

| 状态 | 结论标签 | 依据 |
|---|---|---|
| CPU 数学 | `CPU_REFERENCE_READY` | 7/7 通过；toy 中直接 Adam 更优（负结果保留） |
| Section 2 原训练器等价 | `PARITY_PASS` | 4 状态 × 3 相机 × 5 重复，固定阈值内 |
| M0 v0.1.0 | `BLOCKED_M0`（保留） | 接口/数值失败，非机制证伪 |
| M0 v0.1.1 / v0.1.2 | `M0_ADAPTER_PASS_ONLY` | 门值 0.0，阈值未改 |
| Section 4 v0.1.1 | `PARTIAL`（6/12 完成，5 ERROR） | `pack_geometry` Cholesky 中止 |
| Section 4 v0.1.2 | `COMPLETE`（12/12），筛查标签 **`HOLD_NO_MECHANISM_ADVANTAGE`** | 见 O12/O13 |
| Section 5 控制 | `NOT_RUN` | 缺原参数空间 Adam、LM/二阶、全量成本、等墙钟轨迹 → 不发 `GO_TO_M2` |

---

# HYPOTHESIS

1. **机制假设（原本要检验的）**：在多视角屏幕空间做非线性拟合、再映射回 3D 共识解，
   会比直接在 3D 参数空间做 Adam 更有效地把训练视角的监督转成真实三维几何改进
   （在审计视角上表现为同等或更小的退化）。
2. **本轮的竞争解释（必须排除）**：观测到的“屏幕臂审计退化更小”可以由**步长差异**解释——
   屏幕臂在训练损失上走得更少，自然在审计上破坏更小；要排除它必须做**等墙钟轨迹**与
   **原参数空间、同调优程度的 Adam/LM 对照**。
3. **数值假设（已证实）**：pinned 渲染器的 fp32 quat→covar 公约在近下溢尺度行上返回非正定 Σ；
   这类行对任何需要 Cholesky 的三维参数化都是真实约束，必须在方法层面定义行为
   （v0.1.2 选择“相对迹的极小抖动 + 计数上报”）。
4. **未检验**：持续对偶变量、结构增删（split/clone/prune）、全量成本与真实训练轨迹下的收益。

---

# TEST

* **设计**：完全沿用 `DSH_EXECUTION.md` 的固定矩阵与固定阈值；未做任何阈值、相机规则、
  抽样规则、臂定义的调整；v0.1.2 的新版本只为修复数值鲁棒性而存在，门与阈值逐字未改。
* **判定门槛（预注册）**：M0 固定容差；Section 4 只做筛查，允许 `HOLD_NO_MECHANISM_ADVANTAGE`；
  `GO_TO_M2` 只能在前述控制补齐后给出；**不得**把 `screen_one_step` 称作 LM，
  其 `trace` 只记录屏幕步，`fit_history` 才是 float64 阻尼 GN + 回溯的共识解。
* **未做/不得声称**：没有全量训练、没有结构变化、没有加速数字（GPU 与他人共享，
  GPU 0/1 全程被占用，任何耗时都只是诊断量）；`screen_one_step` 的 3 次反传 < 9 次
  反传不等于更快，实测墙钟更慢。
* **可重放命令**：

```bash
cd /root/epic05_prism/prism_gs_v0.1.2
/root/miniforge3/envs/gsplat/bin/python -m unittest discover -s tests -v
/root/miniforge3/envs/gsplat/bin/python -m prism.cpu_probe --out results/cpu_probe.json
PRISM_GPU=2 /root/miniforge3/envs/gsplat/bin/python -m prism.run_gpu \
  --gx-path /root/epic05_grc3d/scripts --gpu 2 --scene bicycle \
  --snapshot /root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt \
  --camera-seq /root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy \
  --m0-only --out results/bicycle_5k_m0.json
bash /root/epic05_prism/scripts/run_section4_sweep_v012.sh      # 12 单元
bash /root/epic05_prism/scripts/v012_core.sh                    # 单测+probe+M0+parity×4
/root/miniforge3/envs/gsplat/bin/python /root/epic05_prism/scripts/diag_pd_v1.py \
  --scene garden --state 014000 \
  --snapshot /root/epic05_grc3d/inputs/snapshots/snap_garden_014000.pt \
  --out /root/epic05_prism/outputs/diag_pd_garden_014000.json
```

---

# CONCLUSION

1. **接口层已可复现地通过**：M0 适配器门在 v0.1.1/v0.1.2 上全绿，Section 2 独立原训练器等价检查
   4/4 在预注册固定容差内通过（v0.1.2 树绑定重跑）。v0.1.0 的 `BLOCKED_M0` 已定位为
   原始四元数未归一化、log-Cholesky 屏幕往返有损、SH 激活端点语义三处接口/数值问题，
   已按版本化补丁修复，**阈值与规则一字未改**，原始失败输出完整保留。
2. **首轮真实图像探针已完整（12/12，v0.1.2）**，观测性无秩亏；但**没有任何口径下的机制优势或加速**：
   训练损失下降幅度与直接 Adam 同量级且多数情况更小，审计退化虽更小却与步长混杂，
   墙钟在最优的那一种屏幕臂上也只是持平。据此判定 **`HOLD_NO_MECHANISM_ADVANTAGE`**，
   **不进入 M1，不发 `GO_TO_M2`，不做三倍加速声明**。
3. **必须补的控制（进入下一轮的最小集合）**：原参数空间（means/quats/log_scales/opacity/SH）
   的 Adam 对照；调优后的 LM/二阶对照；含结构操作的**全量成本**；**等墙钟轨迹**；
   ≥3 配对 seed 的配对顺序检验；并把 `screen_one_step` 定位为诊断臂而非方法代表。
4. **数值层面的独立发现值得留在记录里**：pinned 渲染器在其自身 fp32 协方差公约下会对
   近下溢尺度的高斯返回非正定 Σ（garden 14K 约 0.28% 的行），而它自己渲染这些行没有问题；
   任何以 Cholesky/平方根为坐标的三维方法都要显式定义此处的行为并上报修复量。
5. **未启动任何昂贵完整训练，未修改旧裁决，未提交 git。**

---

## 附录：与 v0.1.0 的补丁摘要

| 版本 | 变更 | 触发原因 |
|---|---|---|
| v0.1.1 | P1 屏幕特征改为渲染器原生 conic（去 log-Cholesky 往返）；P2 SH 激活改 `relu()`（严格 `>0`）；P3 `covariance()` 在 CUDA 上委托渲染器 `quat_scale_to_covar_preci(F.normalize(q))` 并镜像 triu 布局 | M0 的 `cov`/`zero`/`sh` 三类失败 |
| v0.1.2 | P4 `pack_geometry` 先对称化，用 `cholesky_ex` 定位非正定行，对这些行做相对迹的对角抖动升级（1e-6→1e-1）并上报行数/最大抖动；`run_gpu.py` 记录 `geometry_pack` | Section 4 中 5 个 garden 单元 Cholesky 中止 |

`covars=` API 备注（两版一致）：`gsplat.rasterization(covars=…)` 只接受 `[..., N, 3, 3]`，
Python 包装层总会用 `tri_indices` 转 6 向量，传 `[N, 6]` 会被静默错索引成 `[6]` 并抛错。
