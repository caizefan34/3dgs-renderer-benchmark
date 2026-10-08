import json, os, datetime
ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
SYN = os.path.join(ROOT, "FUSED_HOTSPOT0")
os.makedirs(os.path.join(SYN, "source_map"), exist_ok=True)
NOW = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
KDIR = "pylibs/FasterGSFusedCudaBackend/rasterization"; K = ROOT + "/" + KDIR
source_map = {
  "schema": "FUSED_HOTSPOT0_SOURCE_KERNEL_MAP_V1", "generated_utc": NOW,
  "method": "Static inspection of FasterGSFused CUDA backend source (no names from papers).",
  "backend_source_root": KDIR,
  "entry_point": {"python": "Methods/FasterGSFused/Renderer.py::render_image_training -> _C.forward / _C.backward",
                  "cuda_forward": KDIR+"/src/rasterization_api.cu::forward_wrapper -> forward()",
                  "cuda_backward": KDIR+"/src/rasterization_api.cu::backward_wrapper -> backward()"},
  "kernels": [
    {"component":"P","name":"preprocess_cu","file":KDIR+"/include/kernels_forward.cuh","line":14,"role":"Project 3D Gaussians to 2D; covariance/conic, depth, opacity, SH color per primitive.","semantic_class":"EXACT_RENDER_REQUIRED","block_size":"block_size_preprocess=128"},
    {"component":"I","name":"apply_depth_ordering_cu","file":KDIR+"/include/kernels_forward.cuh","line":205,"role":"Order primitives by depth per tile (pre-sort key prep).","semantic_class":"BOOKKEEPING_REQUIRED","block_size":"block_size_apply_depth_ordering=256"},
    {"component":"I","name":"create_instances_cu","file":KDIR+"/include/kernels_forward.cuh","line":218,"role":"Generate per-tile instances (duplications); gate every instance with will_primitive_contribute exact closest-corner max-contribution tile reject. Sequential path for first n_sequential_threshold=4 tiles + cooperative warp path beyond.","semantic_class":"EXACT_RENDER_REQUIRED","block_size":"block_size_create_instances=256","mechanisms_present":["exact_tile_contribution_rejection (will_primitive_contribute, kernel_utils.cuh:62)","warp_level_cooperative_culling (compute_cooperatively, warp.shfl)"]},
    {"component":"I","name":"extract_instance_ranges_cu","file":KDIR+"/include/kernels_forward.cuh","line":323,"role":"Compute per-tile instance index ranges for bucketed backward.","semantic_class":"BOOKKEEPING_REQUIRED","block_size":"block_size_extract_instance_ranges=256"},
    {"component":"I","name":"extract_bucket_counts","file":KDIR+"/include/kernels_forward.cuh","line":342,"role":"Count buckets per tile for bucketed backward/checkpoint replay.","semantic_class":"BOOKKEEPING_REQUIRED","block_size":"block_size_extract_bucket_counts=256"},
    {"component":"S","name":"DeviceRadixSort (sort instances by tile,depth)","file":KDIR+"/include/buffer_utils.h (radix sort utility)","role":"Sort generated instances by (tile, depth) into ordered per-tile blend lists.","semantic_class":"BOOKKEEPING_REQUIRED","note":"Sort passes: RadixSortScan / RadixSortUpsweep (from FP2 trace)."},
    {"component":"F","name":"blend_cu","file":KDIR+"/include/kernels_forward.cuh","line":354,"role":"Forward alpha-blending per tile (block_size_blend=256 = one 16x16 tile/block); shared-memory staging; early-stop at min_alpha_threshold=1/255; transmittance early-termination at 1e-4.","semantic_class":"EXACT_RENDER_REQUIRED","block_size":"block_size_blend=tile_width*tile_height=256","mechanisms_present":["shared_memory_staging","per_pixel_alpha_early_stop (1/255)","transmittance_early_termination (1e-4)"]},
    {"component":"L","name":"l1_loss / fusedssim","file":"Methods/FasterGSFused/Loss.py (torch)","role":"L1 + DSSIM loss (LAMBDA_L1=0.8, LAMBDA_DSSIM=0.2).","semantic_class":"LOSS"},
    {"component":"PB","name":"preprocess_backward_cu","file":KDIR+"/include/kernels_backward.cuh","line":15,"role":"Backward projection: dL/dmean2d, covariance/scale/rotation gradients, SH color gradient.","semantic_class":"BACKWARD_ONLY","block_size":"block_size_preprocess_backward=128"},
    {"component":"B","name":"blend_backward_cu","file":KDIR+"/include/kernels_backward.cuh","line":252,"role":"Backward alpha-blending: per-Gaussian image-space gradient accumulation over contributing tiles (bucketed).","semantic_class":"BACKWARD_ONLY"},
    {"component":"BO","name":"adam_step_invisible + adam_step_helper (fused in backward)","file":KDIR+"/include/kernels_backward.cuh","line":464,"role":"Fused Adam update of means/SH/opacity/scales/rotations moments INSIDE the backward kernel (per-Gaussian).","semantic_class":"OPTIMIZER_FUSED","block_size":"block_size_adam_step_invisible=256","mechanisms_present":["backward_optimizer_fusion"]},
    {"component":"D","name":"densification/pruning (update_gaussian_statistics, clone/split/prune, opacity reset)","file":"Methods/FasterGSFused/{Model,Trainer}.py (torch) + preprocess_backward stat accumulation","role":"Densification every DENSIFICATION_INTERVAL=100 (600..14900), opacity reset every 3000. GPU stat accumulation via preprocess_backward.","semantic_class":"OPTIONAL_STRATEGY","note":"Active in S1(5k)/S2(10k); inactive at S3(15k mature, post DENSIFICATION_END_ITERATION=14900)."}
  ],
  "known_mechanisms_already_intrinsic": ["exact_tile_contribution_rejection (will_primitive_contribute)","warp_level_cooperative_culling","per_pixel_alpha early-stop (1/255)","transmittance early-termination (1e-4)","bucketed per-Gaussian backward (instance ranges + bucket counts)","backward/optimizer fusion (Adam inside backward)","shared-memory staging (one tile per block, 256 threads)"],
  "config": {"tile_width":16,"tile_height":16,"block_size_blend":256,"n_sequential_threshold":4,"min_alpha_threshold":"1/255","transmittance_threshold":"1e-4","dilation":0.3,"block_size_preprocess":128,"block_size_create_instances":256},
  "baseline_overlap_rule": "Any candidate mechanism already present above must NOT be promoted (section 24)."
}
for p in [os.path.join(SYN,"source_map","FUSED_HOTSPOT0_SOURCE_KERNEL_MAP.json"), os.path.join(SYN,"FUSED_HOTSPOT0_SOURCE_KERNEL_MAP.json")]:
    json.dump(source_map, open(p,"w"), indent=2)
print("WROTE source kernel map; kernels:", len(source_map["kernels"]))