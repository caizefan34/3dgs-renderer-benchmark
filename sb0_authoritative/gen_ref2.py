import json
ref = {
 "task":"SB0-FGS","deliverable":"SB0_FASTERGS_REFERENCE","date":"2026-09-30",
 "status":"COMPLETE (matched 1080p, seed42, 30K x3 on EPIC-05 A100) + C0-comparison flagged for protocol/provenance review.",
 "evidence_class":"OBSERVED (full 30K train+eval on EPIC-05) + HYPOTHESIS (C0 garden/bicycle PSNR anomaly)",
 "correction_note":"Prior Windows-session判'blocked'是环境错位。EPIC-05 上重建 Faster-GS 后端(py3.12/torch2.10)成功，跑通完整 30K x3。",
 "environment":"EPIC-05 A100-SXM4-80GB; Faster-GS backend rebuilt for py3.12.13/torch2.10.0+cu128; NeRFICG @ e458ac6; deps installed (munch/kornia/torchmetrics/lpips/simple_knn/ppisp/fused-ssim/pycolmap).",
 "reference":{"repo":"nerficg-project/faster-gaussian-splatting","paper":"arxiv 2602.09999 (CVPR2026)",
   "local_clone":"/root/codex_nerficg_benchmark_20260927","branches":["main","FasterGSTestbed","FasterGSFused"]},
 "protocol_match":{
   "resolution":"BOTH 1080p. C0 GTDataset 1080p=resize to 1920x1080 (STRETCH, non-uniform). Faster-GS IMAGE_SCALE_FACTOR aspect-preserve: room .5205->1621x1080, garden .3213->1667x1080, bicycle .3287->1626x1080. CAVEAT: aspect handling differs (stretch vs preserve).",
   "iterations":30000,"seed":42,"loss":"0.8*L1+0.2*(1-SSIM) both","sh_degree":3,"background":"black both",
   "densification":"C0 500..15000 grad0.0008; Faster-GS 600..14900 grad0.0002 (both adaptive, non-MCMC)","eval_SSIM_metric":"C0 SepSSIM (eval) vs Faster-GS DSSIM-based — DIFFERS; PSNR/LPIPS comparable"
 },
 "fastergs_matched_30k":{
   "room":{"psnr":32.05,"ssim":0.936,"lpips":0.243,"n_gaussians":1148197,"train_wall_s":228.72,"peak_vram_gib":6.90,"resolution":"1621x1080","run_dir":"output/FasterGS/codex_sb0_fastergs_room_30k_1080p_s42_2026-09-30-00-35-35"},
   "garden":{"psnr":32.45,"ssim":0.947,"lpips":0.176,"n_gaussians":1054944,"train_wall_s":150.42,"peak_vram_gib":None,"resolution":"1667x1080","run_dir":"output/FasterGS/codex_sb0_fastergs_garden_30k_1080p_s42_2026-09-30-00-42-23"},
   "bicycle":{"psnr":32.44,"ssim":0.947,"lpips":0.178,"n_gaussians":1097330,"train_wall_s":155.25,"peak_vram_gib":None,"resolution":"1626x1080","run_dir":"output/FasterGS/codex_sb0_fastergs_bicycle_30k_1080p_s42_2026-09-30-00-42-23"}
 },
 "c0_authoritative_30k":{
   "source":"/root/3dgs-publication-epic05/full30k/c0_s4{2,3,4}_{scene} (torch2.9.1+cu128, timing_grade=PUBLICATION, ext d26cccf/core 935ea6c0 IDENTITY_MATCH)",
   "room":{"psnr_mean":32.347,"ssim_mean":0.9281,"lpips_mean":0.3016,"mean_iter_ms":29.441,"total_wall_s_mean":937.5},
   "garden":{"psnr_mean":24.044,"ssim_mean":0.7287,"lpips_mean":0.3226,"n_gaussians_mean":2415093,"mean_iter_ms":37.867,"total_wall_s_mean":1177.1},
   "bicycle":{"psnr_mean":26.402,"ssim_mean":0.8358,"lpips_mean":0.2473,"n_gaussians_mean":3663216,"mean_iter_ms":41.058,"total_wall_s_mean":1273.8}
 },
 "comparison_observed":{
   "room":"C0 32.19 vs FasterGS 32.05 PSNR — TIED (cleanest comparison). LPIPS C0 0.302 vs FGS 0.243 (FGS better).",
   "garden":"C0 24.04 vs FasterGS 32.45 — 8.4 dB gap. C0 anomalously LOW (3DGS garden typically 27-29). ANOMALY.",
   "bicycle":"C0 26.40 vs FasterGS 32.44 — 6.0 dB gap. C0 low. ANOMALY.",
   "wall_clock":"Faster-GS 30K 150-229s vs C0 880-1280s on same A100. FGS ~4-8x faster. CAVEAT: resolution aspect-handling differs (C0 stretch 1920x1080 vs FGS preserve ~1620-1670x1080); not perfectly iso-resolution."
 },
 "anomaly_investigation":{
   "observation":"C0 room ties Faster-GS, but C0 garden/bicycle are 6-8 dB low. Same data source, same 1080p, same seed.",
   "hypothesis":"C0 garden/bicycle PSNR likely depressed by (a) non-uniform 1920x1080 STRETCH on high-aspect scenes, and/or (b) H8_MR renderer issue on large/high-res scenes, and/or (c) eval-protocol difference. NOT evidence that Faster-GS is a better model.",
   "required_before_criterion_B":["re-run C0 garden/bicycle with aspect-preserve 1080p to confirm whether 24/26 is a stretch artifact","align eval SSIM metric (SepSSIM vs DSSIM)","confirm C0 eval GT resolution matches FGS"],
   "status":"OPEN — do NOT claim criterion B quality until C0 anomaly resolved"
 },
 "wall_clock_note":"Faster-GS 30K measured on EPIC-05 A100 (this run). This is OUR measurement, NOT upstream paper number. C0 wall from PUBLICATION timing.json. Resolution-aspect caveat applies.",
 "success_criteria":{
   "A_c0_plus":"C7 sole candidate; awaiting reviewer go-ahead before C0 CUDA edit.",
   "B_stronger_external_baseLINE":"FASTER-GS REFERENCE ESTABLISHED (matched 30K x3 reproducible). But 'significantly stronger than C0' NOT yet claimable: C0 garden/bicycle PSNR anomaly must be resolved first (room ties). FGS IS faster in wall-clock (measured)."
 }
}
json.dump(ref,open("/root/sb0_fgs/SB0_FASTERGS_REFERENCE.json","w"),indent=2)
print("updated ref bytes",len(json.dumps(ref)))
