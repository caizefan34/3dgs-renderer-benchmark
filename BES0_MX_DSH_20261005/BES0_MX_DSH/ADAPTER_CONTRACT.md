# 场景与 kernel adapter 契约

driver 由 DSH 实现，文件名 scripts/run_e1.py/run_e2.py/run_e3.py。不得把模板当结果；第一条科学行为前调用 mx_guard.require_guarded_process()。

登记 config/scene_manifest.json（templates 有样例），每场景包含 checkpoint SHA256、数据 split、camera-id 清单与 hash、GT 路径/hash、native repo commit、分辨率/N/SH、验证/开发/测试划分。room 的历史数字只是验证线索；未找齐另两个真实场景 -> BLOCKED_INPUTS。E1 可先仅 room，但 E2/E3 PASS 要求三个场景齐备。

统一 adapter 接口（DSH 实现，签名固定）：

```python
load_scene(scene_record, device) -> Scene
project_and_sort(scene, camera, tile_size) -> TileLists
native_alpha(tile_lists, occurrence_ids, pixel_xy) -> Tensor
build_bounds(tile_lists, mode) -> BoundsAndRouting
native_complete(scene, camera, return_cache=False) -> ImageAndCache
native_default(scene, camera, return_cache=False) -> ImageAndCache
bes_sample(scene, camera, routing, stream_spec) -> EventImagesAndCache
bes_l2_surrogate(event_images, target) -> Scalar
optimizer_step(scene, grads, optimizer_state) -> None
```

`native_complete` 关闭 early-stop，只合成同一 native 有效列表；`native_default` 原厂默认包括 early-stop 与原损失。完整 L2 训练性能比较必须同一更新器/学习率/活跃参数/相机序列。所有 loss reduce=像素及 RGB 平均，1/2 L2，必要时配对重新设置完全相同 LR；记录，不在测试集调参。

R/D 分类按 occurrence，不是 primitive 全局属性；camera pixel center 与 tile edge 固定源定义。颜色范围、背景、SH clamp 等记录源码。PSNR/SSIM/LPIPS实现、模型权重版本/hash、GT颜色空间明确。缺失LPIPS依赖不能默认该门 PASS。

每 stage JSON 共同字段：schema, stage, status, protocol_sha256, implementation_sha256, source_commit, guard_sha256, selected_config_sha256, inputs_verified, raw_files, checks。raw_files 相对 review 目录的路径/SHA256，源日志需包含对应 GUARD 与 RUN_STATUS；任何文件缺失/改动则不能 PASS。模板中的 null/空对象必须留到真实测量后才填。
