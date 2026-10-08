#!/usr/bin/env python3
"""OPTLAWS0 / OC0 — shared constants and frozen definitions.

OC0 is the "Occlusion-Gradient Conflict Census" sidecar of the OPTLAWS0
campaign (see OC0_PREREG.md, frozen 2026-09-29). It is DIAGNOSTIC ONLY:

  * no training, no densification, no formal test views;
  * no renderer / optimizer / loss / resolution change;
  * MG0 gates and MG0 scripts are untouched (mg0_core is imported read-only);
  * no scientific verdict is issued here -- raw artifacts are returned.

Every constant in this module is a LITERAL from the frozen preregistration.
Nothing here is selected or tuned after seeing data.
"""
import os

CELLS = (("room", 20000), ("room", 25000), ("bicycle", 20000), ("bicycle", 25000))

# ---------------------------------------------------------------- selection
CAM_STRIDE = 2               # "every second camera in frozen order" -> 32 cams
CAM_SUBSET_SIZE = 32
N_HOT = 1024                 # 1024 highest-V_j rows from MG0-S
N_UNIFORM = 1024             # 1024 uniform nonzero-V_j rows
SELECTION_SEED = 20260929    # frozen seed (same value as the MG0-S sample)
SELECTION_SIZE = N_HOT + N_UNIFORM

# ---------------------------------------------------------------- conflicts
SEVERE_COSINE = -0.25        # severe_ab = (cosine_ab < -0.25); NOT tuned
CONFLICT_COSINE = 0.0        # conflict_ab = (cosine_ab < 0); NOT tuned

# ---------------------------------------------------------------- replay math
# Exact production rasterizer semantics (gsplat `rasterize_to_pixels_3dgs`):
#   px = x + 0.5, py = y + 0.5
#   sigma = 0.5*(c0*dx^2 + c2*dy^2) + c1*dx*dy
#   alpha = min(MAX_ALPHA, opac*exp(-sigma))
#   skip if sigma < 0 or alpha < ALPHA_THRESHOLD
#   next_T = T*(1-alpha); if next_T <= TRANSMITTANCE_THRESHOLD -> pixel is
#   done and THIS gaussian is excluded ("exclusive" saturation)
#   last_ids[pixel] = flat index (into flatten_ids) of the last accepted gaussian
MAX_ALPHA = 0.999
ALPHA_THRESHOLD = 1.0 / 255.0
TRANSMITTANCE_THRESHOLD = 1e-4   # log(1e-4) = -9.210340371976184, exact

# ---------------------------------------------------------------- pairwise
T_EPS = 1e-6                 # eps in abs(log(mean_T_a+eps) - log(mean_T_b+eps))
ROLE_T_MIDPOINT = 0.5        # a-priori transmittance midpoint for the
                             # visibility-role mismatch indicator (NOT tuned)

PAIRWISE_FEATURES = (
    "abs_logT_diff",           # abs(log(mean_T_a+eps) - log(mean_T_b+eps))
    "abs_depth_pct_diff",      # abs(mean_depth_percentile_a - ._b)
    "abs_loghits_diff",        # abs(log(accepted_hits_a+1)-log(accepted_hits_b+1))
    "abs_mean_alpha_diff",     # abs(mean_alpha_a - mean_alpha_b)
    "role_mismatch",           # visibility-role mismatch indicator (0/1)
    "abs_prior_count_diff",    # abs(mean_prior_contrib_a - ._b)
)
# "T/depth-role" feature subset used by discovery criterion 2.
T_DEPTH_ROLE_FEATURES = ("abs_logT_diff", "abs_depth_pct_diff")

# ---------------------------------------------------------------- paths
MG0_ROOT = os.environ.get("MG0_ROOT", "/root/epic05_mg0")
ARTIFACTS = os.path.join(MG0_ROOT, "artifacts")
LOGS_DIR = os.path.join(MG0_ROOT, "logs")
OC0_ROOT = os.path.join(MG0_ROOT, "optlaws0")
FROZEN_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
OC0_WT = os.environ.get("OC0_WORKTREE", os.path.join(MG0_ROOT, "c0_worktree_oc0"))
PARQUET_DIR = os.path.join(ARTIFACTS, "oc0")

# Cameras used for the full-image RGB parity comparison (a fixed subset of the
# OC0 32-camera set: positions 0,4,8,...,28 -> 8 cameras). Frozen here.
PARITY_CAM_SLOTS = (0, 4, 8, 12, 16, 20, 24, 28)

# Frozen parity envelope (declared a priori; the replay must match the native
# production renderer to this tolerance before any renderer feature is used).
PARITY_MAX_ABS_RGB = 2.0 / 255.0      # per-channel absolute RGB tolerance
PARITY_MEAN_ABS_RGB = 0.25 / 255.0
PARITY_MAX_ABS_ALPHA = 2.0 / 255.0
PARITY_MAX_ABS_LOSS = 2.0e-3
# The four max-abs numbers above are SINGLE-PIXEL extremes, and no independent
# re-derivation of the loop can match them at the loop's DISCRETE boundaries
# (`sigma >= 0`, `alpha >= 1/255`, `next_T <= 1e-4`): the CUDA kernel is
# compiled with FMA contraction and fast-math expf, so a value straddling a
# boundary by less than one ulp can land on either side, and the affected
# gaussian is then admitted or dropped outright.  Acceptance therefore couples
# the max-abs envelope with an exceedance budget: the fraction of pixels that
# exceed the max-abs tolerances must be negligible.  Both numbers are reported
# verbatim in OC0_INSTRUMENTATION_PARITY.json so the independent analyst can
# inspect the boundary population directly.
PARITY_MAX_EXCEEDANCE_FRACTION = 1e-4   # 0.01% of pixels, declared a priori


def artifact(name):
    return os.path.join(ARTIFACTS, name)


def instrumentation_present(module):
    """True iff `module` carries the OC0 additive publication hook.

    The hook is installed at FORWARD time -- the inserted block publishes into
    ``sys.modules[__name__]._OC0_LAST`` while the native forward runs -- so
    ``hasattr(module, "_OC0_LAST")`` is always False before the first render.
    Presence is therefore established from the module SOURCE, which is exactly
    what the patch changed.  Read-only; never mutates the module.
    """
    p = getattr(module, "__file__", None)
    if p and os.path.isfile(p):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return "_OC0_LAST" in f.read()
    return hasattr(module, "_OC0_LAST")


def parquet_path(scene, stage):
    return os.path.join(PARQUET_DIR, f"OC0_RENDERER_FEATURES_{scene}_{stage}.parquet")


def cell_tag(scene, stage):
    return f"{scene}_{stage}"