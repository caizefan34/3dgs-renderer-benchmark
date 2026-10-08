import io
# CHANGELOG: append new dated entry
cl=r"D:\学习\_meta\CHANGELOG.md"
entry="""

## 2026-10-04 CGT-Splat teacher px/py bug 根因修复 + 严格区间证书 + G3 四场景（MX，fail-closed 全量重跑）
- **根因 bug**：`g1_compressibility.py` `tile_teacher` 原 `dx=px[:,None,:]-mu[None,:,:]` 只用 px 忽略 py（teacher 沿对角线 y=x 求值）。确认测试：高斯峰(2,3) 旧返回 alpha=0.60(应 0.99)，(2,3)与(2,2)输出相同。修复为 `np.concatenate([px,py],axis=2)-mu`（line 131）。`g3_gpu_microbench.py` `tile_teacher_torch` 同一 bug，一并修。
- **全量重跑**：G1(6 组合)/G2-dense(3 场景)/G2-rigorous(3 场景)/G3(4 场景) 全用修复后 teacher 重跑。旧 buggy 结果归档 `G1_RAW_RECORDS/buggy_teacher/`(6 g1+2 g3) + `stale_superseded_buggy_teacher/`(4 旧 g1_room_r*.json)。
- **影响**：G1 几乎不变（q_weighted_r16 一致到 4 位小数，6/6 PASS）；bug 主要影响 G2 保真度指标。
- **新增严格证书** `g2_rigorous_certificate.py`（区间算术 branch-and-bound，plan §5.5 interval bounds）：student 区间盒=expert 预测凸包，teacher 区间盒=conic power 区间，validated=True(合成 bound/pointmax 1.0-4.6)。pooled 严格界通过率 rank4/8/16=0.75/0.633/0.617；**严格证明 student 非处处复现 teacher**(pass<1.0)。
- **G3 扩 4 场景**(bicycle/room r5k/r30k/garden)：intrinsic γ 中位 R4=26.4x(范围1.3-78x)，per-tile γ 1.65-1.82x，S 1.38-1.49x，仍<3x 非单调。
- **裁决**：G2 证书 PARTIAL_DENSE_COVERAGE→DENSE_AND_RIGOROUS_INTERVAL_DONE；总裁决 `G1_PASS_G2_DENSE_AND_RIGOROUS_INTERVAL_CERTIFICATE_DONE_G3_MEASURED_LAUNCH_BOUND_G4_BATCHING_RECOVERABLE_NOT_A_SPEEDUP_CLAIM`；gpus_used={0,1,2,3,4,5,6}。
- **交付物**：12+ 项重算。MANIFEST 49→103 条(含 logs/原始记录/构建脚本/buggy 归档)，MX+本地双侧 sha256sum -c 全 OK。同步 `cgt_realscene_results/final/`(104 文件)。
- **fail-closed**：发现 root-cause bug 后未沿用旧结果；全部重跑+归档；G2 严格证书 pass<1.0 如实上报；未放门/未删困难 tile/未把 Amdahl 界或 S_batch 当加速。详见 `3dgs-renderer-benchmark\CGT_G1_REALSCENE_COMPRESSIBILITY_MX_20261004.md` §12。
"""
with io.open(cl,"a",encoding="utf-8") as f: f.write(entry)
print("CHANGELOG appended")
# todo: mark rigorous done, add teacher-fix entry
td=r"D:\学习\_meta\todo.md"
t="""
- [done] CGT-Splat teacher px/py bug 修复 + 严格区间证书 + G3 四场景（2026-10-04 第三轮，MX）。根因：`tile_teacher` 只用 px 忽略 py（对角线求值），修 `concatenate([px,py])`；g3 `tile_teacher_torch` 同 bug 一并修。G1/G2-dense/G2-rigorous/G3 全量重跑，buggy 归档 `buggy_teacher/`+`stale_superseded_buggy_teacher/`。G1 不变(q_weighted_r16 4 位一致，6/6 PASS)；新增 `g2_rigorous_certificate.py` 区间算术界 validated=True，pooled pass 0.75/0.633/0.617 **严格证明 student 非处处复现 teacher**；G3 扩 4 场景(intrinsic γ 中位 26.4x，per-tile S 1.38-1.49x)。裁决 `G1_PASS_G2_DENSE_AND_RIGOROUS_INTERVAL_CERTIFICATE_DONE_G3_MEASURED_LAUNCH_BOUND_G4_BATCHING_RECOVERABLE_NOT_A_SPEEDUP_CLAIM`。MANIFEST 103 条双侧 sha256sum -c 全 OK。详见 CGT_G1_REALSCENE_COMPRESSIBILITY_MX_20261004.md §12。
- [next] 剩余 NOT_REACHED：G4 真·融合 CGT CUDA 核 vs gsplat 微基准（G4 批处理探针已证 student 181× 可批，但非融合核）；G5 原生 CGT 训练(路线 B §6.3)+渲染器集成+≥3x 验证。均需重大新基础设施（scope 决策）。G2 严格证书已把"处处复现"从未验证升级为已证伪，后续若推进须直面不可压缩尾(§6.1 更细空间分区或更高 rank)。
"""
with io.open(td,"a",encoding="utf-8") as f: f.write(t)
print("todo appended")
