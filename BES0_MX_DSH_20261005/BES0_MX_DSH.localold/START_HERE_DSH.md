# 给 DSH：BES0，全部科学实验只在 MX

目标：证伪或验证 BES-GS 能否将相同质量的训练时间缩短至少 3 倍。先读 METHOD.md / NUMERICS.md / PROTOCOL.md / GATES.md。**不要在本机执行 reference、toy、profile、训练或渲染。** 本地只可 AST/JSON/哈希/ZIP 校验。交付状态 NOT_RUN；没有启动 DSH。

本包已有：MX 守卫、限时启动器、可执行 E0 数学参考、预注册协议、阶段模板、裁决/归档器。尚未有：生产 CUDA 渲染器、真实场景 adapter、E1/E2/E3 driver；DSH 必须实现这些，不得将 Torch 参考性能当 CUDA 性能。

1. 本地可运行 `python3 scripts/verify_package.py`。将完整目录传到 DSH 已可信的 MX SSH profile；不要猜 endpoint 或绕过 host-key 检查。
2. 在 MX 使用配置里的 nerficg Python 执行 `scripts/mx_guard.py --inspect`。核实主机、GPU、版本、checkpoint；历史路径不代表现在仍存在。
3. 依照现场预约规则，占用 1/2/3/4 中一张空闲卡；0/5/6/7 禁用。只可单卡，不能驱逐别人任务。用实际 UUID 替换下方占位符。
4. 先运行 E0（launcher 对 e0 不要求 checkpoint）。然后 E1 做生产正确性与机会筛选；通过后才 E2/E3。

```bash
/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python scripts/run_on_mx.py \
  --run-id bes0_001 --gpu-uuid GPU-REPLACE-WITH-OBSERVED-UUID \
  --stage e0 --entry scripts/run_e0.py --max-minutes 20
```

添加且审计 driver 后，按同一 launcher 调 `scripts/run_e1.py`（60 分钟）、`scripts/run_e2.py`（180 分钟）、`scripts/run_e3.py`（420 分钟）。所有 driver 第一项必须 `require_guarded_process()`，之后才 import torch/科学库。上面阶段上限合计 680 分钟，包含 stage 内编译与预热；全任务累计上限 12 小时。未完成历史按声明 timeout 计费，不得删除。

5. 提交 config/selected_config.json，绑定 protocol/implementation hash；E2 后冻结配置，E3 禁止测试集调参。使用新的 run-id，保留前序。
6. 用 `scripts/collect_review.py --output review --selected-config /ABS/PATH/selected_config.json --e0 /ABS/PATH/E0_STAGE_DIR --e1 /ABS/PATH/E1_STAGE_DIR --e2 /ABS/PATH/E2_STAGE_DIR --e3 /ABS/PATH/E3_STAGE_DIR` 汇总原始输出（未运行的stage省略参数）。它保留stage原JSON，给汇总JSON中的raw路径加前缀，并附GUARD/RUN_STATUS；不改测量值。附 raw logs、timing CSV、source patch、机型/版本、配置、hash、checkpoint、camera/seed 清单。运行 `python3 scripts/reduce_verdict.py review`；人工填写 `review/BES0_REPORT.md` 后运行 `python3 scripts/package_results.py review /ABS/PATH/BES0_RESULTS.zip`。这些整理脚本不是实验，可本地使用。

需要输入缺失就返回 BLOCKED_INPUTS；预算不够返回 INCONCLUSIVE_BUDGET。不要杜撰另外两个场景的路径、GT 或 benchmark。不要只交截图、FPS 或“成功运行”的描述。
