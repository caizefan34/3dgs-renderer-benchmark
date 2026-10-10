# R17 DSH 执行指令（MX / A100-40GB）

> 将本文件全文作为 prompt 交给 DSH。DSH 需要在 MX 机器上执行 R17 实验包并回传结果。
> 若 DSH 因工具权限被阻塞：提交**具体失败命令、exit code、stderr**，不要再次询问是否执行。
> 用户已授权开展 R17；未执行的步骤必须标记 NOT_EXECUTED，禁止虚构任何结果。

---

## 0. 身份与目标

你是 MX 上的实验执行者。执行包 `TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX` 已经准备就绪，
你需要：

1. 在 MX 上部署执行包（解压到工作目录）；
2. 运行 `run_on_mx.sh`（单进程、单 GPU、逐步执行 preflight → P0…P6 → collect）；
3. 收集全部产物、按严格标准判定成功/失败、打包回传。

实验目的（供你理解上下文，不要自行修改任何脚本或实验设计）：判定 R3–R16 是否给 HiGS
渲染器喂入了违反输入契约的参数（raw logit opacity / log-scale / 未归一化四元数）以及合成
相机，并验证这是否（而非 HiGS 本身）导致此前的 intersection/显存爆炸。

## 1. 硬性约束（违反任何一条即为执行失败）

- **只读原则**：严禁修改冻结渲染器源码树（`$HIGS_PROJECT` 下任何文件）、二进制
  （`csrc.so`）、原始 checkpoint。执行包自身只在进程内 monkeypatch
  `gsplat.cuda._wrapper.isect_tiles` 且 in `finally` 还原；不要以任何方式绕过。
- **禁止篡改产物**：不得手工编辑任何产出的 CSV/JSON；不得用不同 seed / checkpoint
  重跑到同一 `R17_OUT` 目录。
- **禁止虚构**：任何步骤失败或未运行，必须如实汇报（collect 会输出 EXECUTED /
  EMPTY / NOT_EXECUTED / PARSE_ERROR；失败时汇报具体命令 + exit code + stderr）。
- **禁止复用 R3 加速声明**：P4 的所有数字必须来自本次实测；不得引用 R3 的
  HN-540-B4 数字做横向比较。
- **阻塞上报**：如果工具权限阻塞执行，提交具体失败命令 + exit code + stderr，
  不要再次请求执行许可。

## 2. 执行步骤（按序）

```bash
# 2.1 部署（假设包已上传到 MX 的家目录；若路径不同自行调整）
cd ~
unzip -o TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX.zip
cd TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX
chmod +x run_on_mx.sh

# 2.2 全量运行（默认路径已在 README_EXECUTION.md 中列出；如 MX 路径不同用 env 覆盖）
CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh
```

预期：P3（8 cells × fwd+bwd @ 580K + 逐 cell oracle replay）是耗时大头。运行结束后：

```bash
# 2.3 结果自检
cat $R17_OUT/RUN_STATUS.json | python3 -m json.tool | head -80
cat $R17_OUT/logs/step_exit_codes.txt

# 2.4 打包回传（R17_OUT 默认 /mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX）
cd /mnt/storage_pool/liaoyuanjun
tar czf TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX_return.tar.gz \
    TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX
sha256sum TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX_return.tar.gz
```

## 3. 严格成功标准（逐条核验并在回报中给出证据）

| ID | 标准 | 证据 |
|---|---|---|
| S1 | preflight 执行完毕 | `raw/R17_PREFLIGHT.json` 存在且可解析；`blocking_violations` 内容如实上报（即使为空；mismatch 只记录不阻塞） |
| S2 | 10 个 required deliverables 全部 EXECUTED | `RUN_STATUS.json → summary.executed == 10` |
| S3 | P3 全部 8 个 cells 有明确状态 | `R17_580K_CONTROLLED_COMPARISON.csv` 每行 fwd_status ∈ {PASS, SAFE_ABORT, OOM, OUT_OF_HEADROOM, ERROR}；PASS 行必须带 fwd_ms/peak_vram_gb |
| S4 | P4 两个协议 × 两个后端都有实测或明确不可执行记录 | `R17_VALIDATED_BENCHMARK.csv` 行覆盖 {B1,B4}×{std_ll,higs}；跳过的行必须带 free_gb 证据 |
| S5 | P6 给出全部 4 项裁决 + R3–R16 disposition | `R17_FINAL_VERDICT.json` 中 OPACITY/SCALE/ROTATION_INPUT_CONTRACT、CAMERA_PROTOCOL、BENCHMARK_VALIDITY、HIGS_MEMORY_SCALABILITY 均有终值（非 TODO/占位） |
| F | 任一步不可执行 | `NOT_EXECUTED`（禁止填充）；回报中附：完整命令、exit code、stderr 原文 |

**允许的"可解释失败"**（不构成执行失败，但必须带证据）：
- P5 4.41M full 场景 OOM / gate 判定 SKIP_FULL_INSUFFICIENT_HEADROOM —
  count-only 估计与 gate 决策即为记录证据；
- 某 cell `OUT_OF_HEADROOM`（free_gb < 20）——记录 free_gb 即可。

## 4. 回报格式（必须包含）

1. 每步 exit code（`logs/step_exit_codes.txt` 全文）；
2. `RUN_STATUS.json → summary`（required / executed / not_executed 列表）；
3. 四项裁决的终值（从 `R17_FINAL_VERDICT.json` 摘抄，原文键值）；
4. P3/P4 关键表的关键列（尤其 P4 的 B1/B4 × std_ll/higs 的 fwd/bwd 中位数）；
5. 回传包路径 + sha256；
6. 失败项（若有）的完整命令、exit code、stderr 原文。

不要对结果做解释性推测——解释由研究侧完成；你只负责如实执行与如实回报。

## 5. 附注

- conda 激活失败只警告不阻断（runner 会继续），此时用 `PYTHON=/path/to/python` 覆盖；
- `R17_CAMERAS_JSON` 若自动解析失败，显式指向 bicycle `cameras.json`；
- 每个 step 是独立进程，重跑单个 step 不会破坏其它 step 的产物；
  重跑后再次执行 `python3 scripts/r17_collect.py` 刷新 RUN_STATUS/MANIFEST。
