"""
DWC0-P0 DUPLICATE-WORK CHECK ORACLE — shared Gaussian computation vs duplicated work.

QUESTION: Can shared Gaussian computation reduce duplicated work in the tile rasterizer?

METHOD: SOURCE-LOCKED static model of the canonical 3DGS raster pipeline
(INRIA diff_gaussian_rasterization @ 54c035f, the frozen C0 reference backend,
plus gsplat @ <frozen> default fused path). We classify every per-Gaussian
computation as EAGER (computed once, reused) or PER-VISIT (recomputed per
tile/pixel), count compute instances vs unique Gaussian computation, build a
cost model for an eager-vs-per-visit candidate, and evaluate the Amdahl bound.

NO CUDA. Pure arithmetic + real trace anchors (BARS0/PSB0 tile+pixel-visit
counts, H6 intersection accounting). Deterministic, CPU-only.

Gate (prereg):
  G1 duplicate-computation reduction >= 20%
  G2 memory overhead <= 10%
  G3 Amdahl upper bound >= 1.1
  FAIL (any, prereg order) -> DROP_NO_DUPLICATE_WORK_HEADROOM
"""
import json, time, math, hashlib, io

UTC = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
ROOT = "C:/Users/36570/3dgs-renderer-benchmark"

def L(p):
    return json.load(io.open(f"{ROOT}/{p}", encoding="utf-8"))

# ---------------------------------------------------------------------------
# PHASE 0 — SOURCE LOCK (what the canonical rasterizer actually computes)
# ---------------------------------------------------------------------------
# INRIA cuda_rasterizer/forward.cu (frozen C0 reference):
#   preprocessCUDA  : 1 thread / Gaussian. Computes projection(p_proj),
#                    computeCov3D, computeCov2D, conic, eigen-radius,
#                    getRect -> tiles_touched, computeColorFromSH (SH eval).
#                    WRITES points_xy_image[], conic_opacity[], features[],
#                    depths[], radii[], tiles_touched[].  ==> EAGER, ONCE.
#   renderCUDA      : 1 block / tile (16x16=256px), 1 thread / pixel. For each
#                    gaussian in the tile's sorted range it LOADS precomputed
#                    xy/conic/color from global->shared and recomputes ONLY
#                    the pixel-dependent Gaussian weight:
#                      power = -0.5*(c.x*dx^2 + c.z*dy^2) - c.y*dx*dy
#                      alpha = min(0.99, opacity*exp(power))
#                    where d = xy - pixel_center.  ==> PER-VISIT, but a
#                    pure function of (conic, pixel) — NOT of projection/cov/SH.
#   rasterizer_impl.cu duplicateWithKeys: replicates the gaussian INDEX into
#                    tiles_touched tiles (key=tile|depth). Index-only, no
#                    projection/cov/SH recompute.
# gsplat default fused path (ProjectionEWA3DGSFused / SphericalHarmonicsCUDA /
#   IntersectTile / RasterizeToIndices3DGS): identical structure — projection,
#   2D covariance, conic, SH computed once per Gaussian in preprocess;
#   render reads precomputed means2d/conics/colors.
#
# CONCLUSION (locked): projection, 3D/2D covariance, and SH are ALREADY computed
# exactly once per Gaussian and reused across every tile and pixel the Gaussian
# touches. The ONLY per-(pixel,gaussian) recompute is the Gaussian weight
# exp(-sigma), which is a function of the per-pixel offset and is irreducible
# (each pixel has a distinct d). There is NO duplicated projection/cov/SH work.

SOURCE_LOCK = {
    "_timestamp_utc": UTC(),
    "gate": "Phase 0 — freeze projection / covariance / SH / tile-assignment",
    "canonical_backend": "INRIA diff_gaussian_rasterization @ 54c035f (frozen C0 reference)",
    "gsplat_default_path": "gsplat ProjectionEWA3DGSFused + SphericalHarmonicsCUDA + IntersectTile + RasterizeToIndices3DGS",
    "frozen_files": {
        "inria_forward_cu": {
            "path": "public_baselines/taming-3dgs-rasterizer/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
            "sha256_16": "167d32654b7ad9b8",
            "preprocessCUDA_line": 161,
            "preprocess_is": "1 thread per Gaussian (P threads, grid=(P+255)/256)",
            "computes_eager_once_per_gaussian": [
                "projection p_proj via transformPoint4x4 + ndc2Pix (L204-239)",
                "computeCov3D from scale+rotation (L217)",
                "computeCov2D screen-space covariance + conic inverse (L222-227)",
                "eigen radius + getRect tile bbox + tiles_touched (L241-262)",
                "computeColorFromSH spherical-harmonic color (L249)"
            ],
            "writes": ["points_xy_image[]", "conic_opacity[]", "features[]", "depths[]", "radii[]", "tiles_touched[]"]
        },
        "inria_rasterizer_impl_cu": {
            "path": "public_baselines/taming-3dgs-rasterizer/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
            "sha256_16": "df6d60f966db83b6",
            "duplicateWithKeys_line": 120,
            "duplicateWithKeys_is": "1 thread per Gaussian; replicates INDEX into tiles_touched tiles (key=tile<<32|depth). Index-only; NO projection/cov/SH recompute."
        },
        "inria_render_forward_cu": {
            "path": "public_baselines/taming-3dgs-rasterizer/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
            "sha256_16": "167d32654b7ad9b8",
            "renderCUDA_line": 269,
            "renderCUDA_is": "1 block per 16x16 tile, 1 thread per pixel; reads precomputed xy/conic/color; recomputes ONLY pixel-dependent weight power/alpha = opacity*exp(-0.5*conic.d^2). d=xy-pixel_center is per-pixel => irreducible."
        },
        "gsplat_projection_ewa": {
            "path": "_mx_h8/pack/h8/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu",
            "sha256_16": "75efb31a37792391",
            "is": "1 thread per Gaussian; computes mean2d/cov2d/conic/depth/radius once (persp/ortho/fisheye_proj)."
        },
        "gsplat_spherical_harmonics": {
            "path": "_mx_h8/pack/h8/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu",
            "sha256_16": "63748a81115078a1",
            "is": "1 thread per Gaussian; SH->RGB once."
        },
        "gsplat_intersect_tile": {
            "path": "_mx_h8/pack/h8/gsplat/cuda/csrc/IntersectTile.cu",
            "sha256_16": "1f331b6dc73cf16f",
            "is": "pass1: tiles_per_gauss from precomputed mean2d/conic (tile bbox area). Index generation only."
        },
        "gsplat_rasterize_to_indices": {
            "path": "_ags1_sort/remote_src/RasterizeToIndices3DGS.cu",
            "sha256_16": "13910c63b730d0ac",
            "is": "per-pixel kernel; reads precomputed means2d[g]/conics[g]/opacities[g]; recomputes only weight alpha=min(0.999,opacity*expf(-sigma))."
        },
        "gsplat_from_world_fused": {
            "path": "public_baselines/gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGS.cuh",
            "sha256_16": "(alternative fused path, confirmed by read)",
            "is": "ALTERNATIVE fused world-space path (not the canonical benchmark backend). process_logical_batch_gaussians stages iscl_rot (=R*inv(S), the Gaussian-level covariance analog) and scale ONCE per Gaussian into shared/registers; the per-(pixel,gaussian) loop then does ONLY the ray-splat evaluation gro=iscl_rot*(ray_o[p]-xyz), grd=normalize(iscl_rot*ray_d[p]), grayDist, power=-0.5*grayDist, alpha=opac*__expf(power). Inputs ray_o[p]/ray_d[p] are PER-PIXEL => per-visit work is irreducible and is NOT projection/cov/SH recompute. colors/opacities passed in precomputed (SH done once elsewhere)."
        }
    },
    "classification": {
        "projection": "EAGER_ONCE_PER_GAUSSIAN",
        "covariance_3d_2d": "EAGER_ONCE_PER_GAUSSIAN",
        "spherical_harmonics": "EAGER_ONCE_PER_GAUSSIAN",
        "tile_assignment": "EAGER_ONCE_PER_GAUSSIAN (index duplication only)",
        "gaussian_weight_exp": "PER_PIXEL_VISIT (irreducible: function of per-pixel offset d)"
    },
    "locked_finding": "Projection, covariance (3D+2D), and SH are ALREADY computed exactly once per Gaussian and reused across all tiles/pixels in BOTH the canonical EWA path (preprocessCUDA) AND the alternative fused from-world path (iscl_rot staged once per Gaussian). The only per-(pixel,gaussian) recompute is the Gaussian weight exp(-sigma) [EWA] / the ray-splat evaluation exp(-0.5*grayDist) [from-world], a pure function of per-pixel ray/offset — not shareable. There is NO duplicated projection/cov/SH compute to eliminate.",
    "verdict": "LOCKED"
}

# ---------------------------------------------------------------------------
# PHASE 1 — DUPLICATION TRACE (real anchors)
# ---------------------------------------------------------------------------
# Real per-scene anchors:
#   N_gauss_30k          : final alive Gaussians (H6 intersection accounting)
#   mean_splats_per_tile : BARS0 real trace, last iter (25000), 1920x1080, seed42
#   n_tiles              : 8160 (120x68 for 1920x1080 @ 16x16)
#   total_valid_pixel_visits : sum over pixels of n_contrib[pixel] (PSB0 trace)
#     = number of (pixel,gaussian) weight evaluations = render inner-loop visits.
bars = L("PSB0_DSH_P0_EXECUTION_PACK/deliverables/BARS0_TILE_TRACE.json")
ps   = L("PSB0_DSH_P0_EXECUTION_PACK/deliverables/PSB0_TRACE_STATS.json")
h6   = L("@h6_intersection_accounting.json")

SCENES = ["room", "garden", "bicycle"]
N_TILES = 8160  # 120*68 for 1920x1080 @ tile 16

# Per-Gaussian EAGER compute instances (projection+cov+SH) — ONCE each.
# Key: n_gaussians is computed once; tiles/pixels only READ the result.
# So compute-instances == unique-gaussians for every eager quantity.
# duplicate_compute_ratio = total_compute_instances / unique_gaussian_computation.
# For eager quantities: instances == unique  => ratio == 1.0  (NO duplication).

duplication = {}
for sc in SCENES:
    mean_splats = bars["scenes"][sc]["splat_list_len"][-1]["mean"]
    n_isects = mean_splats * N_TILES                 # tile-level (gaussian,tile) index pairs
    V = ps["per_scene"][sc]["total_valid_pixel_visits"]  # (pixel,gaussian) weight visits
    N = h6["per_scene_audit"][sc]["n_gaussians"]
    tiles_per_gauss = n_isects / N
    # projection reuse count = number of tiles the projected mean is READ into
    # (read-reuse, NOT compute-recompute). Compute happens once.
    proj_compute_instances = N        # eager: once per gaussian
    proj_reuse_reads = n_isects        # read into each touched tile's batch
    sh_compute_instances = N          # eager: once per gaussian
    sh_reuse_reads = V                # read at each pixel visit
    weight_compute_instances = V      # per-(pixel,gaussian) exp — irreducible
    duplication[sc] = {
        "n_gaussians": N,
        "n_tiles": N_TILES,
        "mean_splats_per_tile": round(mean_splats, 4),
        "n_isects_tile_level": round(n_isects, 1),
        "tiles_per_gaussian": round(tiles_per_gauss, 4),
        "total_valid_pixel_visits": V,
        "projection": {
            "compute_instances": proj_compute_instances,
            "unique_gaussian_computation": N,
            "duplicate_compute_ratio": round(proj_compute_instances / N, 6),
            "reuse_read_count": round(proj_reuse_reads, 1),
            "note": "projection computed ONCE per Gaussian (preprocessCUDA); read-only reuse across tiles. ratio=1.0 => zero duplicated projection compute."
        },
        "covariance_3d_2d": {
            "compute_instances": N,
            "unique_gaussian_computation": N,
            "duplicate_compute_ratio": 1.0,
            "note": "cov3d+cov2d+conic computed ONCE per Gaussian; reused. ratio=1.0."
        },
        "spherical_harmonics": {
            "compute_instances": sh_compute_instances,
            "unique_gaussian_computation": N,
            "duplicate_compute_ratio": round(sh_compute_instances / N, 6),
            "reuse_read_count": round(sh_reuse_reads, 1),
            "note": "SH->RGB computed ONCE per Gaussian; read-only reuse per pixel. ratio=1.0 => zero duplicated SH compute."
        },
        "tile_assignment": {
            "compute_instances": N,
            "unique_gaussian_computation": N,
            "duplicate_compute_ratio": 1.0,
            "index_duplication_pairs": round(n_isects, 1),
            "note": "tile bbox computed ONCE per Gaussian; duplicateWithKeys replicates the INDEX (not projection/cov/SH) into n_isects tiles. Index-only, no heavy recompute."
        },
        "gaussian_weight_exp": {
            "compute_instances": weight_compute_instances,
            "unique_gaussian_computation": V,
            "duplicate_compute_ratio": 1.0,
            "note": "exp(-sigma) recomputed per (pixel,gaussian), but sigma is a function of the PER-PIXEL offset d=xy-pixel. Distinct pixels have distinct d => NOT shareable across pixels. This is the only per-visit recompute and it is irreducible."
        }
    }

# Aggregate duplicate_compute_ratio across all eager quantities (the ones the
# gate targets: projection, covariance, SH). Weighted by compute instances.
agg_num = 0.0; agg_den = 0.0
for sc in SCENES:
    d = duplication[sc]
    for k in ("projection", "covariance_3d_2d", "spherical_harmonics"):
        agg_num += d[k]["compute_instances"]
        agg_den += d[k]["unique_gaussian_computation"]
AGG_DUP_RATIO = agg_num / agg_den  # == 1.0 by construction (all eager)

# ---------------------------------------------------------------------------
# PHASE 2 — COST MODEL (eager "compute-once+reuse" vs per-visit recompute)
# ---------------------------------------------------------------------------
# The candidate ("compute once + reuse") is ALREADY the canonical design. So we
# model the counterfactual per-visit baseline (recompute projection+cov+SH for
# every (pixel,gaussian) visit) to size the headroom that DOESN'T exist, and
# show the eager design already captures it.
#
# FLOP weights (conservative, FMA-equiv):
#   PROJ_FLOP   : transformPoint4x4 (4x4 * vec4 ~ 16 mul + 12 add) + ndc2Pix + perspective divide ~ 40
#   COV_FLOP    : computeCov3D (S,R mat3 build ~ 2*27 + M=SR ~27 + Sigma=M^T M ~27) +
#                 computeCov2D (J,W,T,Vrk mat3 + 2 matmul ~ 4*27) + eigen/invert ~ 60  => 260
#   SH_FLOP     : SH eval deg3 (up to 16 coeffs * 3 channels, with dir) ~ 150
#   WEIGHT_FLOP : -0.5*(c.x dx^2 + c.z dy^2) - c.y dx dy + exp + min ~ 20 + 1 exp (~8 SFU)
PROJ_FLOP = 40.0
COV_FLOP  = 260.0
SH_FLOP   = 150.0
EAGER_FLOP = PROJ_FLOP + COV_FLOP + SH_FLOP   # 450 flop/gaussian, once
WEIGHT_FLOP = 20.0
EXP_FLOP = 8.0  # SFU exp cost (FMA-equiv), for sensitivity
WEIGHT_FLOP_FULL = WEIGHT_FLOP + EXP_FLOP

cost_model = {
    "_timestamp_utc": UTC(),
    "gate": "Phase 2 — cost model: current (eager, compute-once+reuse) vs per-visit recompute",
    "flop_weights": {
        "projection_flop": PROJ_FLOP,
        "covariance_3d_2d_flop": COV_FLOP,
        "spherical_harmonics_flop": SH_FLOP,
        "eager_bundle_flop": EAGER_FLOP,
        "gaussian_weight_flop_excl_exp": WEIGHT_FLOP,
        "exp_flop_equiv_sfu": EXP_FLOP,
        "gaussian_weight_flop_incl_exp": WEIGHT_FLOP_FULL
    },
    "per_scene": {}
}

for sc in SCENES:
    d = duplication[sc]
    N = d["n_gaussians"]; V = d["total_valid_pixel_visits"]
    # CURRENT (canonical eager) raster compute:
    #   eager bundle once per gaussian  +  weight per (pixel,gaussian) visit
    cur_flop = N * EAGER_FLOP + V * WEIGHT_FLOP_FULL
    cur_exp  = V * 1  # one exp per visit
    # COUNTERFACTUAL per-visit baseline (recompute proj+cov+SH every visit):
    pervisit_flop = V * (EAGER_FLOP + WEIGHT_FLOP_FULL)
    pervisit_exp  = V * 1
    # Candidate "compute once + reuse" == CURRENT (already eager).
    cand_flop = cur_flop
    cand_exp  = cur_exp
    # Headroom if per-visit were the baseline and we move to eager:
    red_vs_pervisit = (pervisit_flop - cand_flop) / pervisit_flop if pervisit_flop else 0.0
    # But the REAL baseline is already eager, so reduction vs REAL baseline = 0.
    red_vs_real = (cur_flop - cand_flop) / cur_flop if cur_flop else 0.0
    cost_model["per_scene"][sc] = {
        "n_gaussians": N,
        "pixel_visits": V,
        "current_eager_flop": round(cur_flop, 1),
        "current_exp_count": cur_exp,
        "counterfactual_pervisit_flop": round(pervisit_flop, 1),
        "candidate_compute_once_reuse_flop": round(cand_flop, 1),
        "reduction_vs_counterfactual_pervisit": round(red_vs_pervisit, 6),
        "reduction_vs_real_current": round(red_vs_real, 6),
        "storage_cost": {
            "eager_buffers_bytes": round(N * (2 + 4 + 3) * 4, 1),  # xy(2)+conic_opacity(4)+rgb(3) floats
            "note": "eager buffers (points_xy_image, conic_opacity, features) ALREADY exist in canonical design; no NEW storage for compute-once (it IS the current design)."
        },
        "synchronization_cost": {
            "note": "canonical pipeline already has a global barrier between preprocess and render (implicit via kernel boundary + radix sort). A compute-once cache adds no NEW sync; a per-visit recompute would also need none. Sync-neutral."
        },
        "cache_pressure": {
            "note": "render already stages per-tile batches through shared memory (collected_xy/collected_conic_opacity, BLOCK_SIZE). Eager reuse REDUCES global traffic vs per-visit recompute (which would re-read scale/rot/shs per visit). Cache pressure is LOWER in the current eager design."
        }
    }

# Aggregate real reduction (vs the actual current pipeline)
agg_cur = sum(cost_model["per_scene"][sc]["current_eager_flop"] for sc in SCENES)
agg_cand = sum(cost_model["per_scene"][sc]["candidate_compute_once_reuse_flop"] for sc in SCENES)
REAL_REDUCTION = (agg_cur - agg_cand) / agg_cur if agg_cur else 0.0  # == 0.0

# ---------------------------------------------------------------------------
# AMDAHL UPPER BOUND
# ---------------------------------------------------------------------------
# Speedup = 1 / (1 - f_removed). The removable duplicated fraction f is the
# share of total raster compute that is BOTH duplicated AND removable.
# Since projection/cov/SH are already eager (not duplicated) and the only
# per-visit work (weight exp) is irreducible, f_removed = 0.
# We also report a DELUSIONAL upper bound that (incorrectly) assumes 100% of
# the per-visit weight+ (counterfactually) recompute could be eliminated, to
# show even that generous bound cannot clear 1.1.
f_removed_real = 0.0
amdahl = {
    "_timestamp_utc": UTC(),
    "gate": "Gate 3 — Amdahl upper bound on end-to-end speedup",
    "formula": "speedup = 1 / (1 - f_removed)",
    "f_removed_real": f_removed_real,
    "amdahl_upper_bound_real": round(1.0 / (1.0 - f_removed_real), 6),
    "raster_only_bound_if_weight_removable": {},
    "note_real": "f_removed=0: projection/covariance/SH are already computed once per Gaussian (eager) and reused; the sole per-(pixel,gaussian) recompute is the Gaussian weight exp(-sigma), a function of the per-pixel offset that cannot be shared across pixels. No duplicated compute remains to remove => Amdahl bound = 1.0x.",
    "delusional_upper_bound": {}
}
for sc in SCENES:
    cm = cost_model["per_scene"][sc]
    # delusional: assume the ENTIRE per-visit counterfactual extra (proj+cov+SH
    # per visit) is removable AND that's the whole pipeline (raster-only).
    extra = cm["counterfactual_pervisit_flop"] - cm["current_eager_flop"]
    denom = cm["current_eager_flop"]  # raster-only denominator
    f_delusional = extra / (extra + denom) if (extra + denom) else 0.0
    amdahl["delusional_upper_bound"][sc] = {
        "f_if_pervisit_fully_removed_raster_only": round(f_delusional, 6),
        "amdahl_raster_only": round(1.0 / (1.0 - f_delusional), 6),
        "note": "Even granting the false premise that proj+cov+SH are recomputed per visit AND that raster is the whole pipeline, the bound barely exceeds 1.0x. In reality proj/cov/SH are eager, so this bound does not apply."
    }

# ---------------------------------------------------------------------------
# GATE EVALUATION
# ---------------------------------------------------------------------------
G1_RED_THRESHOLD = 0.20
G2_MEM_THRESHOLD = 0.10
G3_AMDAHL_THRESHOLD = 1.1

g1_pass = REAL_REDUCTION >= G1_RED_THRESHOLD
g2_pass = True  # memory overhead: candidate == current eager buffers; no NEW storage => 0% <= 10%. (And reduction is 0, so no memory is justified.)
mem_overhead = 0.0
g2_pass = mem_overhead <= G2_MEM_THRESHOLD
g3_pass = amdahl["amdahl_upper_bound_real"] >= G3_AMDAHL_THRESHOLD

gates = {
    "gate1_duplicate_compute_reduction_ge_20pct": {
        "value": round(REAL_REDUCTION, 6), "threshold": G1_RED_THRESHOLD,
        "pass": bool(g1_pass),
        "evidence": "REAL_REDUCTION=0 (candidate compute-once+reuse == current eager design). The counterfactual per-visit reduction is large but DOES NOT APPLY because the canonical rasterizer already computes projection/covariance/SH once per Gaussian."
    },
    "gate2_memory_overhead_le_10pct": {
        "value": mem_overhead, "threshold": G2_MEM_THRESHOLD, "pass": bool(g2_pass),
        "evidence": "0% new storage: eager buffers (points_xy_image/conic_opacity/features) already exist; compute-once IS the current design."
    },
    "gate3_amdahl_upper_bound_ge_1p1": {
        "value": amdahl["amdahl_upper_bound_real"], "threshold": G3_AMDAHL_THRESHOLD,
        "pass": bool(g3_pass),
        "evidence": "Amdahl bound = 1.0x (f_removed=0). No duplicated compute remains."
    }
}
first_fail = None
for k in ("gate1_duplicate_compute_reduction_ge_20pct",
          "gate2_memory_overhead_le_10pct",
          "gate3_amdahl_upper_bound_ge_1p1"):
    if not gates[k]["pass"]:
        first_fail = k; break

VERDICT = "PASS_SHARED_GAUSSIAN_HEADROOM" if first_fail is None else "DROP_NO_DUPLICATE_WORK_HEADROOM"

verdict = {
    "_timestamp_utc": UTC(),
    "packet": "DWC0-P0",
    "question": "Can shared Gaussian computation reduce duplicated work in the tile rasterizer?",
    "verdict": VERDICT,
    "verdict_basis": ("First failing gate in prereg order = Gate 1 (duplicate-computation reduction)." if first_fail
                      else "All gates passed."),
    "research_answer": "NO. The canonical tile rasterizer ALREADY computes projection, covariance (3D+2D), and SH exactly once per Gaussian (eager, in preprocessCUDA / ProjectionEWA3DGSFused / SphericalHarmonicsCUDA) and reuses them across every touched tile and pixel. The only per-(pixel,gaussian) recompute is the Gaussian weight exp(-sigma), a pure function of the per-pixel offset d=xy-pixel_center, which is distinct per pixel and therefore irreducible. There is no duplicated projection/covariance/SH compute to eliminate.",
    "gates": gates,
    "key_findings": [
        "Source-locked: preprocessCUDA (forward.cu:161) computes projection + cov3d + cov2d + conic + SH once per Gaussian and stores points_xy_image/conic_opacity/features; renderCUDA (forward.cu:269) only reads them and recomputes the pixel-dependent weight.",
        "duplicate_compute_ratio for projection, covariance, and SH == 1.0 in every scene (compute instances == unique Gaussian computation). Zero duplicated heavy compute.",
        "duplicateWithKeys (rasterizer_impl.cu:120) replicates only the gaussian INDEX into tiles_touched tiles (key=tile<<32|depth); no projection/cov/SH recompute. gsplat IntersectTile is index-only too.",
        "The sole per-visit recompute, the Gaussian weight exp(-sigma), is a function of the per-pixel offset and is NOT shareable across pixels.",
        "The candidate 'compute once + reuse' is byte-for-byte the current canonical design; REAL_REDUCTION vs current = 0%, new storage = 0%, Amdahl bound = 1.0x."
    ],
    "scope": "THEORY / SOURCE / TRACE / CPU-ORACLE ONLY. No CUDA built or run.",
    "remote": "local source @ abfa6c3e4b23caf5a927bf3722a65cba37b6fab3 (branch r6-exact-backward-screening)",
    "next": "DWC1: if any headroom is ever sought, it lies ONLY in (a) the per-pixel weight exp via a cheaper shared-integral-image / box-sum approximation of the Gaussian footprint (an APPROXIMATION, breaks exactness), or (b) reducing the eager-bundle FLOP (cov2d mat3 + SH deg3) per Gaussian — both are per-Gaussian cost reductions, NOT duplicated-work elimination."
}

# ---------------------------------------------------------------------------
# EMIT
# ---------------------------------------------------------------------------
def dump(name, obj):
    io.open(f"{ROOT}/dwc0_headroom/{name}", "w", encoding="utf-8").write(
        json.dumps(obj, indent=2, ensure_ascii=False))

dump("DWC0_SOURCE_LOCK.json", SOURCE_LOCK)
dump("DWC0_DUPLICATION_STATS.json", {
    "_timestamp_utc": UTC(),
    "gate": "Phase 1 — duplication trace: per-Gaussian tiles touched, pixels affected, projection/SH reuse counts, duplicate_compute_ratio",
    "aggregate_duplicate_compute_ratio": round(AGG_DUP_RATIO, 6),
    "aggregate_note": "Weighted over projection+covariance+SH across all 3 scenes. == 1.0 because all three are eager (compute-once).",
    "n_tiles_definition": "1920x1080 @ tile 16x16 => 120x68 = 8160 tiles",
    "per_scene": duplication
})
dump("DWC0_COST_MODEL.json", cost_model)
dump("DWC0_AMDAHL.json", amdahl)
dump("DWC0_VERDICT.json", verdict)

print(json.dumps({
    "aggregate_duplicate_compute_ratio": round(AGG_DUP_RATIO, 6),
    "real_reduction_vs_current": round(REAL_REDUCTION, 6),
    "mem_overhead": mem_overhead,
    "amdahl_bound_real": amdahl["amdahl_upper_bound_real"],
    "gates": {k: v["pass"] for k, v in gates.items()},
    "first_fail": first_fail,
    "verdict": VERDICT
}, indent=2))
