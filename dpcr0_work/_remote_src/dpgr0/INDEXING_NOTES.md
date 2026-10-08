# DPGR0 Stage1 索引语义（已核实，IntersectTile.cu:198/458 + DPC0 dpc_stage1.cu）
- flatten_ids[idx] = g（compact visible index）。DPC0 用 means2d[g]/conics[g]/colors[g*CDIM]/opacities[g] **无 image 偏移**读取（line 160-167）。
- 归约/输出 key = image_id*N + g（line 268），flat 写入 [I*N] 缓冲。
- 本 harness 每次 backward 只渲染 1 个相机（I=1，render_loss 单 cam），故 image_id=0、key=g；但仍按 image_id*N+g 原样复刻以保一般性。
- DPGR0 策略：range-build 用稳定 radix 按 key 排序 isect（只排 int32 key+int32 idx，~24MB，远小于 9-float partials），得到 Gaussian-major 段；gather 每 warp 拥有一段，逐 tile 重放反向扫描（re-1→g 位置）捕获 partial，片上累加，升序 tileID，一次写（DIRECT）或 32-tile 分块写 chunk partial 再归约（CHUNKED）。
- 求和顺序：DIRECT 段内升序 tileID = DPC0 升序 isect 序 → 与 DPC0 逐位一致；CHUNKED 分块求和仅差结合性（cosine 门内）。
