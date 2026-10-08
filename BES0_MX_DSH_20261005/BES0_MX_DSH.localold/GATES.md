# 自动裁决顺序

1. 缺stage、模板null、raw hash不符、协议/实现/配置不匹配 -> INCONCLUSIVE_EVIDENCE。
2. 任意 correctness 数值门或 bound/routing/source语义门失败 -> NO_GO_CORRECTNESS。
3. host/input/env/budget等真实阻断 -> 对应BLOCKED_* / INCONCLUSIVE_BUDGET，保留证据。
4. E1 room和其他两场景都没有乐观3x机会，或E2完整step<3x -> NO_GO_OPPORTUNITY。
5. E3任意pair未达同质量、候选峰值内存超baseline、任意pair<2x、九pair几何平均<3x -> NO_GO_FIXED_N_PILOT。
6. 全部门通过 -> GO_FIXED_N_PILOT，不是“颠覆性已证实”。render字段独立输出render_same_quality_success，不改变训练门。

reduce_verdict.py 只读取JSON、验证raw文件hash与冻结配置并执行门，不运行科学计算。输入status=PASS并非充分证据；必须附checks/measurements/raw_files。它不能验证手工测量的真实性，DSH仍须交代码与日志接受审查。

最终报告必须包含本版仅固定N、全部科学试验在MX、source算法先例、未完成事项。NO_GO不允许改称已有创新成果；INCONCLUSIVE不等于失败证据。
