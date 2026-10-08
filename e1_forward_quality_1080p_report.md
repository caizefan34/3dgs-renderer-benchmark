## 2026-10-06 — COMAX0 E1 前向质量数值（1080p，frozen test split，NUMERICS-ONLY）

- **分类**：NEW_LANE_RESULT + AMENDMENT_E1_NUMERICS（E1_NUMERICS_SHARED_GPU_AMENDMENT 下的前向质量-对-GT 数值）。证据=fact（MX 独占干净 A100，严格冻结 guard 下实测）。
- **触发**：延续 PASS_TO_E1 / INCONCLUSIVE_BUDGET 冻结态。目的=在动 E1 完整（预算不可行的）精化 pilot 之前，补上**预注册主分辨率 1080p 的前向质量-对-GT 数值**（H0 只在 720p 测过；CLAMP_VS_ORACLE 只证了 clamp-vs-oracle 忠实性，非 vs GT）。
- **守卫**：经 run_on_mx.py 启动，自动写 guard proof + 20s 空闲门。GPU2 = GPU-25086196-7911-cc05-1452-1885d070d710（A100-40GB，allowed[1-5]，protected[0,6,7] 未碰）；启动时 idle_samples 全为 0 MiB/0%，无 co-resident 进程。TF32-off，power-clamp>=0 忠实 NaN-free 渲染器。
- **方法**：render-only（no_grad），unrefined iter_15000 checkpoint。baseline = gsplat fully_fused_projection + alpha-composite（最强合格基线）；comax = comax_eta_ckpt_clamp（eta=1，frozen kNN-64 grouping）。子集 = frozen test split（39 帧）stride=7 取 6 视图（0/56/112/168/224/280），**测量前**确定性定义。复用 common.Metrics（PSNR/SSIM/LPIPS）。
- **判定**：**quality_forward_pass = FALSE**（delta PSNR -6.99 dB << 门 -0.15 dB）。这是**前向、未精化**数值，**不是** E1 精化质量门，也**不是**计时/性能声明。master verdict **UNCHANGED = INCONCLUSIVE_BUDGET**。

## OBSERVED（MX 实跑，守卫守护）
- 6 视图均值：baseline PSNR 30.10 / SSIM 0.911 / LPIPS 0.200；comax PSNR 23.11 / SSIM 0.835 / LPIPS 0.293。
- delta（comax - baseline）：**PSNR -6.99 dB** / SSIM -0.076 / LPIPS +0.092。
- 逐视图 delta PSNR：view0 -6.66、view56 -7.29、view112 -8.79、view168 -7.65、view224 -5.00、view280 -6.57（6/6 远低于 -0.15 门，方向一致）。
- 无 NaN：baseline_nan_frac=0、comax_nan_frac=0（6/6 视图两侧）。peak GPU 内存 = 822 MiB（无 OOM）。metrics_complete=true。
- comax-vs-GT 均值 23.11 dB 与 H0 720p frozen-grouping 值 23.83 dB 一致（分辨率无关）。
- 运行：e1_fq_full_20261006_a（GPU2）elapsed 1820.8s（~30.3 min），COMPLETED exit 0；冒烟 e1_fq_smoke_20261006_a（GPU1，view0）362s，与 full 的 view0（-6.66 dB）一致。
- 预算：本 run 后 charged 37595.7s（10.443h），remaining 1.557h（93.4 min），51 runs。

## EVIDENCE（产物，均 hash 校验）
- run/e1_fq_full_20261006_a/e1/{E1_FORWARD_QUALITY_1080P.json, GUARD.json, RUN_STATUS.json, console.log}。
- run/e1_fq_smoke_20261006_a/e1/{E1_FORWARD_QUALITY_1080P.json, RUN_STATUS.json, GUARD.json}。
- 新入口 package/COMAX0_MX_DSH/scripts/comax_gs/e1_forward_quality_1080p.py（sha d1adda20a344ad57b712e8e0530c4d24c5268b8dfc0937c002d98969b1d4ca5f，py_compile OK）。
- COMAX0_E1_FORWARD_QUALITY_1080P/{REPORT.md, SOURCE_MANIFEST.sha256, MANIFEST.sha256}。
- 源码 hash（无漂移）：comax_eta_ckpt_clamp、common.py、prereg、mx_target（见 SOURCE_MANIFEST.sha256）。

## INFERENCE
- 缺口来源是 **frozen kNN-64 grouping**（有损表示），不是分辨率、不是 NaN、不是 fast-math：comax-vs-GT 在 720p（H0 23.83）与 1080p（本 23.11）一致；baseline-vs-GT（30.10）是全高斯上限。grouping 把 ~7 dB 质量留在了桌上。
- 这是**精化前的起始缺口**。E1 完整 pilot（1000 步精化，预算不可行）正是要检验精化能否把该缺口收敛到 -0.15 dB 内。本测量只给出起点，不判定 E1 精化门。
- 与 H0 MARGINAL_HEADROOM 并置：frozen grouping 是速度-质量权衡的"质量保持"侧（rho=0.36，算术上限 1.56x），但相对全高斯基线仍付出 ~7 dB 前向代价；要更高 rho 须更粗 grouping（H0 已证 voxel 崩到 14.9 dB vs GT）。→ 速度/质量张力是**根本性**的（grouping 粒度控制），与实现/包络无关。

## HYPOTHESIS（FUTURE_HYPOTHESIS，未用于挽救）
- 精化（1000 步，固定分组）可能回收大部分缺口（grouped 表示仍含 98804 组 / 924550 高斯的容量）；也可能因固定有损分组而**封顶**（FUTURE_HYPOTHESIS）。二者只能由完整 E1 pilot 判定，而该 pilot 在当前参考渲染器下预算不可行（>29s/步）。
- 要真正判定"精化后 comax 是否达 -0.15 dB 门"，须生产级 CUDA 渲染器（另一条线）+ 可负担的 1000 步精化预算。

## NOT_MEASURED
- E1 精化质量（1000 步配对精化 pilot，预算不可行）。任何墙钟速度/CUDA 内核速度/吞吐/带宽/E2-E3 成本。反向（VJP/JVP）。完整 39 帧 test split。精化能否闭合 7 dB 缺口。

## CONCLUSION
- 在预注册主分辨率 1080p 上，frozen grouping 的 comax 前向渲染相对全高斯 gsplat 基线付出 **~7 dB PSNR** 代价（6 视图一致，无 NaN，822 MiB）。分辨率无关、源于 grouping 表示。
- 这是 NUMERICS-ONLY、前向、未精化证据；**不是** E1 精化质量门，**不是**计时/性能声明。performance_evidence_valid=false，timing_claim=false。
- master verdict **UNCHANGED = INCONCLUSIVE_BUDGET**（本证据不推进 reducer；E1 完整 pilot 仍预算不可行，E2 计时仍不合法，E3 依赖 E2）。
- 未重建最终交付 ZIP（无新的性能/精化质量证据）。未碰外部/用户任务、未碰受保护卡。
