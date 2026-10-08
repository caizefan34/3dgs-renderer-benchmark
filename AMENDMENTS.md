# AMENDMENTS.md — ACG0-R0 (MX)

## AMEND-001 · 允许 GPU 集合扩展以使用空闲卡 (2026-10-05)

- 原因: 用户声明 "GPU0-7 都可用"。实时 nvidia-smi 核验显示 GPU0-5 仍被共驻任务
  (PIDs 3197539/3198984/3200555/368063/3205844/3202843/3214442, 均 liaoyuanjun, 运行约 8h,
  22-35GB, 35-99% util) 占用; 仅 GPU6/GPU7 真正空闲 (0 MiB, 0%, 无 compute 进程)。
  协议禁止驱逐共驻任务, 故不触碰 0-5。为利用空闲卡且不扰动任何人, 扩展允许集合纳入 6/7。
- 旧值: allowed_gpu_indices=[1,2,3,4]; protected_gpu_indices=[0,5,6,7]
- 新值: allowed_gpu_indices=[1,2,3,4,6,7]; protected_gpu_indices=[0,5]
- 文件: ACG0_MX_DSH/config/mx_target.json
- 选择: GPU6 (uuid GPU-9cdad9db-1d9f-3512-9133-aff7d8086485) 当前空闲, 用于 E0->E3 链。
- 未改变研究假设/算法/阈值; 属运行资源配置变更, 非 REQUIRES_NEW_PROTOCOL。
- 证据: 空闲卡使用不驱逐任何共驻进程; 全程经 run_on_mx.py 守卫 (assert_idle 校验 index 在允许集合且卡空闲)。
