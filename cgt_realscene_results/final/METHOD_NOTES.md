# CGT-Splat G1 方法学与度量说明

## 工作单元与teacher
- 工作单元 = 一个 gsplat tile（tile x Gaussian 组）。用 gsplat.rasterization(packed=True, tile_size=16) 取 meta 的 isect_offsets/flatten_ids/gaussian_ids/means2d/conics/opacities/depths。
- 按深度排序 tile 内高斯，在 16x16 像素上做 front-to-back alpha 合成得教师 (E,T)。POWER_CLIP=-4.5（~3sigma）裁剪高斯功率。
- colors 索引修复：packed 模式下 means2d/conics/opacities 是 packed 索引，但 SH 求出的 colors 是 original 索引，必须 colors[meta[gaussian_ids]] 重映射，否则颜色错配。

## 学生（2D CGT 混合）
- 复用 cgt_reference.py：predict_and_vjp（高斯门控 softmax + 4 维专家 (E_rgb,T)）、objective（值 MSE + 有限差分导数监督 dweight=0.02）、fit_cgt（L-BFGS-B，rank 属于 {4,8,16}）。
- even 像素训练，odd 像素 held-out 测保真；h=1e-3 中心差分。

## 保真度量（预注册）
- 值：held-out max_abs 小于等于 0.02 且 PSNR 大于等于 30 dB（(E,T) 对黑/白/彩背景）。
- 雅可比：jac_abs = 学生-教师雅可比差 Frobenius / sqrt(n_te*4*2)（RMS/元素）；门 jac_abs 小于等于 max(0.5*resp_range/16, 0.02/16)。resp_range = 教师 (E,T) 在 held-out 像素总变差。
- 为何不用原始 jac_rel：jac_rel = ||J_s-J_t||_F / ||J_t||_F。近平 tile 的 ||J_t||_F 约 1e-4，比值炸到 1e27，纯分母假象。jac_abs 是良定义的绝对度量；scale-aware floor 让近平 tile 只要求绝对小导数。
- rank 非单调：jac_abs 随 rank 升（高 rank 值更好但雅可比更差，与合成反例一致）。未自动升级 rank。

## 工作权重与 Amdahl
- tile_work = tile 内高斯交点数（= flatten_ids 段长）。q_weighted = 通过 tile 的 work / 总 work（cumulative 小于等于 R）。
- S_inference(gamma) = Itot / [sum_pass R_best*gamma + sum_fail M]，R_best=最小通过 rank。gamma=1/2/4 敏感性。
- 这是 Amdahl 上界（零开销假设），非真实加速。

## 抽样
- 分层随机（按 tile_work decile，每 decile 采 n_sel/10），非 top-N（top-N 全是 2000+ 深遮挡 tile，不代表）。
- 6 视图/组合 x 250 tile x 3 rank = 4500 记录。room 视图 25350 tile、garden/bicycle 63860-68575 tile。

## MX SSH 传输（可复用）
- 单连接 stdin 管道最稳：本地脚本文件通过 ssh 执行 tr -d 去 CR 再 bash -s。
- scp + 快速重连会被服务器丢（Connection closed）；detached 用 setsid nohup 重定向到 /dev/null 和 log。
- 启动 detached 的 SSH 命令必须立即退出（尾部 sleep/检查会 hold 住通道触发断连）；用独立轮询命令查状态。
- 登录横幅 SSH warring 混入 stdout，污染 stat/标记解析，用 echo MARK:v + Select-String。

## 局限
- 逐视图教师（非跨视图 3D 一致）；held-out 像素（非视图）；无连续域证书；Amdahl 零开销上界。
