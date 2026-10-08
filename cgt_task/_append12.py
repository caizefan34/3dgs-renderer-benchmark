import io
path=r"D:\学习\3dgs-renderer-benchmark\CGT_G1_REALSCENE_COMPRESSIBILITY_MX_20261004.md"
sec12="""

## 12. teacher px/py bug 根因修复 + 严格区间证书 + G3 四场景（2026-10-04 第三轮，fail-closed 全量重跑）
- **根因 bug（影响所有旧结果）**：`g1_compressibility.py` 的 `tile_teacher` 原为 `dx = px[:,None,:] - mu[None,:,:]`，**只用 px、忽略 py**，使 teacher 沿对角线 y=x 求值而非真实 2D 像素。确认测试：高斯峰在 (2,3)，旧 tile_teacher 返回 alpha=0.60（应为 0.99），且 (2,3) 与 (2,2) 输出相同。**修复**为 `dx = np.concatenate([px[:,None,:], py[:,None,:]], axis=2) - mu[None,:,:]`（line 131，两侧同步）。`g3_gpu_microbench.py` 的 `tile_teacher_torch` 有同一 bug（`dx = px[:,None,:] - mu[None,:,:]`），一并修复为 `torch.cat([px,py],dim=2)`。
- **后果与归档**：所有旧结果（G1/G2-dense/G3 用 buggy 对角 teacher）失效。归档于 MX `final/G1_RAW_RECORDS/buggy_teacher/`（6 g1 + 2 g3）与 `stale_superseded_buggy_teacher/`（4 旧 g1_room_r*.json）。G1/G2-dense/G2-rigorous/G3 全部用修复后 teacher 重跑。
- **影响评估（关键）**：G1 压缩性**几乎不变**——q_weighted_r16 与修复前一致到 4 位小数（bicycle 0.9984、garden 0.9925、room r30k 0.9897、room r10k 0.9983、room r15k 0.9963、room r5k 0.9903），6/6 仍 PASS。bug 主要影响 G2 的**保真度**指标（稠密/严格证书），不改变 G1 可压缩性结论。修复后重跑，未挑有利结果。
- **新增严格证书 `g2_rigorous_certificate.py`（区间算术 branch-and-bound，plan §5.5 要求的 interval bounds）**：对 tile 2D 光线域做分支定界区间算术，给出连续域 max 误差的**严格上界**。student 区间盒 = rank-R expert 预测的凸包（student 是 expert 的凸组合）；teacher 区间盒由 conic power 区间得到。**validated=True**（合成验证 bound/pointmax 比 1.0-4.6，界紧且有效）。3 场景 room(r15k)/garden(r30k)/bicycle(r30k) × 20 tile/rank × rank{4,8,16}，pooled 60 tile/rank。
  - pooled 严格界通过率（≤0.02）：rank4 **0.75**、rank8 **0.633**、rank16 **0.617**；worst-tile 严格界达 0.79（room rank4 M=310）。
  - **严格证明 student 并非处处复现 teacher**：pass_rate<1.0 意味着连续域上存在非空 tile 集，其误差严格超过 0.02 门。这是**证明**（proven），非抽样。高阶 rank 仍非单调。
- **G3 扩到 4 场景**（bicycle, room r5k/r30k, garden；此前仅 room r15k+garden 且用 buggy teacher）。修复后重跑。池化：intrinsic γ（launch-free 数学，中位数）rank4=26.4x（范围 ~1.3-78x，随场景 M_rep/R）；per-tile γ（launch-bound）1.65-1.82x；S（含 dispatch 开销）1.38-1.49x；h_overhead~12%。仍远低于 3x，非单调。
- **裁决更新**：`G2_continuous_domain_certificate` `PARTIAL_DENSE_COVERAGE`→`DENSE_COVERAGE_EMPIRICAL_AND_RIGOROUS_INTERVAL_ARITHMETIC_DONE`；新增 `G2_rigorous_interval_certificate: DONE_VALIDATED_PASS_RATE_BELOW_1_PROVES_NOT_EVERYWHERE`。总裁决：`G1_PASS_G2_DENSE_AND_RIGOROUS_INTERVAL_CERTIFICATE_DONE_G3_MEASURED_LAUNCH_BOUND_G4_BATCHING_RECOVERABLE_NOT_A_SPEEDUP_CLAIM`。gpus_used 修正为 {0,1,2,3,4,5,6}（本轮实际）。
- **交付物**：12+ 项全部重算/重生成。G2_CERT/ 下新增 `G2_RIGOROUS_CERTIFICATE.json`（aggregate）+ 3 个 per-scene `G2_RIGOROUS_*.json`；`G2_CONTINUOUS_DOMAIN_CERTIFICATE.json`（dense aggregate）、`G3_GPU_MICROBENCH.json`（4 场景 aggregate）、`CGT_MASTER_VERDICT.json`、`G2_G5_STATUS.json`、`CGT_MASTER_REPORT.md`、`AMDAHL_MIN_Q.json`、`G1_PER_VIEW_SUMMARY.csv`（27000 行）、`MANIFEST.sha256`（103 条，含 logs/原始记录/构建脚本/buggy 归档，双侧 sha256sum -c 全 OK）。MX `/home/liaoyuanjun/cgt_realscene/final/` + 本地 `cgt_realscene_results/final/` 同步。
- **fail-closed 自检**：发现 root-cause teacher bug 后未沿用旧结果；全部重跑并归档 buggy 版本；G2 严格证书 pass<1.0 如实上报未掩盖；未放门/未删困难 tile/未把未加权高斯数或 Amdahl 界当加速/未把 S_batch 当加速。严格区间证书把"student 处处复现 teacher"从"未验证"升级为"已证伪"。
- **可复用坑**：(1) `analyze_g1.py` glob `g1_*.json` 会同时吃旧 `g1_room_r*.json` 与新 `g1_room_room_r*.json`（同 scene+ckpt key → 双重计数）→ 重跑前须把旧 buggy 文件移出 glob 路径（重命名/隔离）。(2) G3 JSON 字段语义：`per_rank[R].gamma_math_median` 是 per-tile launch-bound 中位比；真正的 launch-free intrinsic 在 `headline.intrinsic_gamma_R*`（≈M_rep/R）——聚合时别混用。(3) Python heredoc 经 stdin 走 `bash -s` 不是 python → 用 `python3 - <<'PYEOF'` 包裹。(4) 远程输出含中文/特殊符号（≥、×）→ 本地 `_ssh_run.py` 须 `encoding="utf-8",errors="replace"` + `sys.stdout/stderr.reconfigure`，否则 GBK 解码崩。
"""
with io.open(path,"a",encoding="utf-8") as f:
    f.write(sec12)
print("appended section 12; new line count:", sum(1 for _ in io.open(path,encoding="utf-8")))
