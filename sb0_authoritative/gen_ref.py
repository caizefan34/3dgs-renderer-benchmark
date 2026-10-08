import json
ref = {
 "task":"SB0-FGS","deliverable":"SB0_FASTERGS_REFERENCE","date":"2026-09-30",
 "status":"PARTIAL — authoritative external-baseline environment CONFIRMED on EPIC-05; room 30K reproduced (resolution-mismatched); garden/bicycle 30K + protocol-matched room PENDING; reference runtime env needs rebuild.",
 "evidence_class":"OBSERVED (repo/binary/logs/results on EPIC-05) + PENDING (matched-res runs, garden/bicycle)",
 "correction_note":"Prior Windows-session判定'everything blocked'是环境错位。EPIC-05(权威)上 C0 身份匹配、NeRFICG 存在、GitHub 可达、8xA100 空闲。",
 "reference":{
   "repo":"https://github.com/nerficg-project/faster-gaussian-splatting",
   "paper":"https://arxiv.org/abs/2602.09999 (CVPR 2026)",
   "local_clone":"/root/codex_nerficg_benchmark_20260927 (git origin nerficg-project/nerficg @ e458ac6)",
   "branches":["main","FasterGSTestbed","FasterGSFused"],
   "method_paths":{
     "FasterGS":"src/Methods/FasterGS/FasterGSCudaBackend (adam/src/adam.cu, separate backward)",
     "FasterGSFused":"src/Methods/FasterGSFused/FasterGSFusedCudaBackend (fused adam_step_helper in kernels_backward.cuh)"
   },
   "build_state":"prebuilt _C.cpython-311-*.so present; runtime was py3.11.16/torch2.8.0/cu12.8; CURRENT EPIC-05 env is py3.12/torch2.10.0+cu128 => reference backend MUST be rebuilt for step-4 matched runs."
 },
 "protocol_differences":[
   {"aspect":"iterations","faster_gs":30000,"c0":30000,"same":True},
   {"aspect":"loss","faster_gs":"0.8*L1+0.2*(1-DSSIM)","c0":"0.8*L1+0.2*(1-SSIM), fused train / SepSSIM eval","same":True,"note":"weights identical; eval SSIM metric differs (Faster-GS DSSIM-based vs C0 SepSSIM) — align before quality compare"},
   {"aspect":"SH_degree","faster_gs":3,"c0":3,"same":True},
   {"aspect":"background","faster_gs":"black(0,0,0)","c0":"black","same":True},
   {"aspect":"densification_window","faster_gs":"600..14900","c0":"500..15000","same":False,"note":"~400-iter offset; both 30K cadence"},
   {"aspect":"grad_threshold","faster_gs":0.0002,"c0":0.0008,"same":False,"note":"C0 config densify_grad_threshold=0.0008 (ORIGINAL_METHOD pixel-space signed 2D-grad L2)"},
   {"aspect":"resolution_room","faster_gs":"IMAGE_SCALE_FACTOR=0.5138 => 1600x1066","c0":"1080p long-edge => ~1080x720","same":False,"note":"CRITICAL: existing room 30K ran at HIGHER res than C0; not a clean quality/speed match. Must rerun Faster-GS at C0 1080p for criterion B."},
   {"aspect":"morton_zorder","faster_gs":"YES (every 5000->15000)","c0":"NO","same":False,"note":"C8; framework layer"}
 ],
 "room_30k_observed":{
   "run_dir":"/root/codex_nerficg_benchmark_20260927/output/FasterGS/codex_fastergs_room_1600_s0_2026-09-27-04-19-14",
   "env":"py3.11.16 / torch2.8.0+cu128 / cuda12.8 / GPU[2] / seed0 / IMAGE_SCALE_FACTOR=0.5138",
   "final_gaussians":1191971,"peak_vram_gib":6.79,"complete_wall_seconds":251.31,
   "metrics":{"fastergs_30000":{"PSNR":32.0295334,"SSIM":0.9375882,"LPIPS":0.1702169},
              "fastgs_30000":{"PSNR":32.2642975,"SSIM":0.9394113,"LPIPS":0.1614227}},
   "caveat":"metrics at 1600x1066; NOT protocol-matched to C0 1080p. wall 251s is at higher res + Faster-GS optimizer (fewer it/s? actually ~119 it/s training loop)."
 },
 "c0_room_30k_authoritative":{
   "source":"/root/3dgs-publication-epic05/full30k/c0_s4{2,3,4}_room",
   "env":"torch2.9.1+cu128 / timing_grade=PUBLICATION",
   "psnr_mean":32.347,"ssim_mean":0.9281,"lpips_mean":0.3016,"n_gaussians_mean":834693,
   "mean_iter_ms":29.441,"wall_s_mean":937.5,"note":"C0 1080p. Direct PSNR/SSIM roughly comparable to Faster-GS room but eval SSIM metric differs; LPIPS gap large (0.30 vs 0.17) partly resolution-driven."
 },
 "step4_pending":[
   {"item":"rebuild Faster-GS + FasterGSFused backend for py3.12/torch2.10 (or recreate py3.11/torch2.8 env)","status":"TODO"},
   {"item":"room 30K at C0-matched 1080p (IMAGE_SCALE_FACTOR to 1080p long-edge)","status":"TODO"},
   {"item":"garden 30K + bicycle 30K (matched protocol)","status":"TODO"},
   {"item":"align eval SSIM metric (Faster-GS vs SepSSIM) before quality claim","status":"TODO"}
 ],
 "success_criteria":{
   "A_c0_plus":"C7 sole implementation candidate (see SB0_OPPORTUNITY_ORACLES.json); requires reviewer go-ahead + isolated worktree + correctness gate before any C0 CUDA edit.",
   "B_stronger_external_baseLINE":"UNLIKELY from audited components — garden config close to C0 (same loss/SH/bg/densification family). Existing room Faster-GS LPIPS lower but at mismatched resolution; confirm only via matched-res 30K x3 (PENDING)."
 },
 "note":"No Faster-GS wall-clock/quality claimed as C0 O_net. Upstream paper speedups NOT used as C0 measured value."
}
json.dump(ref,open("/root/sb0_fgs/SB0_FASTERGS_REFERENCE.json","w"),indent=2)
print("ref bytes",len(json.dumps(ref)))
