# 冻结实验协议 BES0-v1

## 执行与预算

所有科学计算（CPU toy 也包括）在 MX。历史主机 bms-39468022-001，8×A100-PCIE-40GB；允许索引 1/2/3/4 中一张已空闲卡，禁止 0/5/6/7。12 小时累计单卡墙钟上限；launcher 前 20 秒空闲观察。可信 SSH profile、站点预约和干扰日志由 DSH 附证据。不得升级共享环境或结束他人进程。

E0 <=20min；E1 <=60min；E2 <=180min；E3 <=420min；剩余40min用于受守卫约束的重测（总计仍<=720min）。每次新增 run-id，未完成记录按 timeout 收费。预算耗尽不得扩展训练，返回 INCONCLUSIVE_BUDGET。

## 固定参数与数据

room checkpoint/相机按 mx_target.json 核对；另两个已存在场景由 DSH 登记，不自动训练新 checkpoint。3 场景各固定 12 个开发 views、12 个 heldout views，按 camera-id 排序等间隔选取、确保互不重叠；训练只用其余已有训练 split。split 不足则 BLOCKED_INPUTS。view列表与hash在首次运行前保存。

gamma=2；alpha/source语义不变；N、SH、分辨率不变。生产候选 tile=8 或16，train sample=1/2/4，bound=R0 或有证书R1；最多 12 个开发配置，不能测试集搜索。native 优化 baseline tile=16/32，各自合法配置选择开发集最快正确版本。只改后端时必须相同 optimizer/learning schedule与camera sequence。

E3固定N，关闭 densification 与 opacity reset，种子17/29/43，1000step，上限以预算为先。对应场景共同 checkpoint 起点，不训练相机/背景。原训练活跃 geometry/scale/rotation/opacity/SH 都参与；不能只训练颜色然后声称训练加速。

## E0：现成数学检验

执行 run_e0.py，FP64 小列表穷举，独立事件流。确定性期望/梯度检查与 seeded 采样统计都写 E0.json；保存完整日志。E0 PASS 只证明该参考测试，不证明 native CUDA。

## E1：生产正确性与工作量机会

1. 记录原 source commit、compiler flags、runtime版本与 alpha 语义。guard之外，driver核对torch/CUDA/gsplat版本与SM80；不一致BLOCKED_ENV。
2. 开发 views 生产 R0/R1 上下界全像素审计、candidate与native alpha一致性、R/D partition（无重复/遗漏）、sort与背景一致性。任何 bound违规停止。
3. 在 native完整kernel 前后测 projection、sort、alpha/composite、backward的实际阶段耗时与内存；不能将跨kernel异步时间相加当完整wall。
4. 统计按真实 pixel-occurrence 加权的 R/D 比例、l=0原因、hazard fallback、建表/二分/D成本、每query候选分布。基于最有利“R事件/score全免费”消融实际测一个完整工作流（保留必须的projection/sort/bounds/prefix/D/optimizer），与baseline墙钟比较。此消融只用于投资筛选，不是严格硬件下界或性能承诺。
5. 若这个最有利消融在room及另外两场景均达不到3x，或开发最佳实际完整step达不到3x，记录 NO_GO_OPPORTUNITY。如仅room输入存在，可输出PARTIAL/BLOCKED，不能总体PASS。

gamma路径期待查询上界只针对R，不将D调用漏计。原生wrapper baseline要与实际native输出对齐（NUMERICS容差），生产采样至少对small native fixtures枚举期望梯度，并对真实场景像素抽样均值/方差。统计原始samples保存，不能只写“看起来相似”。

## E2：完整step与收敛风险

对3场景开发 views，完整L2 baseline与原厂default损失各保留强基线。预热30step，5组×100step配对计时（若预算不够提前INCONCLUSIVE），组顺序用种子17固定shuffle。报告所有组 wall，取每场景中位step；JIT编译另外报告且占budget。

至少保存参数grad在6个small native fixtures上对解析完整L2梯度的均值与方差，包含interleaved R/D、几何/opacity/SH、背景。FP64参考的deterministic identity必须通过；真实GPU gradient MC 以标准误差报告，不因“单次不接近”误判无偏。E2 actual step>=3x是进入昂贵pilot的投资门，**不是主要结论**；可因variance选择更大sample，但E2结束后冻结唯一配置与implementation hash。

峰值allocated/reserved都报告；主要门使用实际进程GPU高水位，candidate<=较快合格baseline。若局部参考峰值或CUDA进程高水位不齐，INCONCLUSIVE。

## E3：主要判据，time-to-quality

三个场景×三种子配对实验，所有方法每100step与step0在同一12 heldout views评估，无TAA/denoiser。每对方法的起点、数据序列和step计时可审计。原厂default 1000step endpoint确定该pair的质量目标；PSNR/SSIM/LPIPS均使用同一冻结代码、GT与weights。

某checkpoint“达到目标”要求 mean PSNR>=目标-.2dB、SSIM>=目标-.005、LPIPS<=目标+.01，并且所有view PSNR损失<=1dB、LPIPS增长<=.03。只在已评估checkpoint判达到，不插值。不因baseline endpoint之后候选变好而延长1000step预算。

Baseline time是default与complete-L2中能达到目标的方法的最早checkpoint累计时间的最小值；default至迟1000step可达到自己的目标。Candidate time为最早达到目标的checkpoint累计时间。step0已达到目标（共同初始checkpoint）则这pair目标不可辨识，标INCONCLUSIVE_TARGET，不制造无限speedup。

主要时间包括本pair从checkpoint加载、建图/建表、编译、每步训练及同一scheduled评估的墙钟；另报warm后的纯训练时间，不能用它替换主门。各pair分别计实际冷启动时间，缓存命中情况明确；重用编译产物时三方法按同样cache policy。

每pair speedup=t_baseline/t_candidate。总结果 geometric_mean(9pair)>=3.0、每个pair>=2.0，且内存门通过。全部pair都必须有原始证据与质量门；固定N pilot通过只叫GO_FIXED_N_PILOT。完整30k/densification与通用训练加速尚未验证，不自动占用本包额外预算。

## 独立渲染附项

spp=1/2/4/8/16，在heldout views同质量评估，三seed；全pipeline含建表和多sample平均。对照最快正确native默认renderer。以同质量门的最小spp总时间报告，不只单sample FPS。没有通过只叫training-only；若未测标NOT_RUN。对渲染不承诺3x。

## 留痕

保存原始输出而非手抄总结；selected_config包含bounds证书id、tile/sample/RNG/版本/协议/实现hash。任何门槛/输入/实现变动意味着旧证据不再支持新版本PASS。
