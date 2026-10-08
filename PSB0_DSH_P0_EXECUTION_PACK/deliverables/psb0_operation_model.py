"""
PSB0-P0 OPERATION MODEL (Gate 4) — native stride-32 checkpoint+replay vs associative
segment-scan state recovery. No CUDA. FLOP/memory/depth model parameterized by exp cost.

Native model frozen from nerficg@e458ac6e FasterGSFused backward.cu::blend_backward_cu.
Counts use REAL per-scene valid-pixel-visit totals (BARS0/APBC0 traces, 1920x1080, seed42).
"""
import json, time

# Real per-scene totals (total_valid_pixel_visits = sum over pixels of n_contrib[pixel])
SCENES = {
    "room":    {"V": 14894848, "nbuckets": 58472},
    "garden":  {"V": 35872768, "nbuckets": 140684},
    "bicycle": {"V": 25459968, "nbuckets": 101725},
}
TILE_PIX = 256          # 16x16
CKPT_FLOATS = 4         # float4 (color.rgb + T)
FLOAT_BYTES = 4
BUCKET = 32

# --- per-visit state-recovery cost (excludes gradient math, identical in both) ---
# Native replay step: recompute gaussian exp + alpha + blending_weight + color_after update + T update
NATIVE_EXP = 1          # expf recomputed in backward replay
NATIVE_FLOP = 11        # alpha(1 mul)+bw(1 mul)+color_after(3mul+3sub)+T(1sub+1mul) = 11

# Scan: build per-splat transform M_k=(A_k,B_k), A_k=a_k*c_k(3), B_k=1-a_k(1) => 5 flop + 1 exp
SCAN_BUILD_FLOP = 5
SCAN_BUILD_EXP = 1
# Blelloch work-efficient inclusive prefix scan: 2(N-1) composes; compose=(A1+B1*A2, B1*B2)=7 flop
SCAN_COMPOSE_FLOP = 7
SCAN_COMPOSE_PER_VISIT = 2 * SCAN_COMPOSE_FLOP   # 14 flop/visit amortized
# C_after_k = C_total - C_inc_k : 3 sub
SCAN_CAFTER_FLOP = 3

def model(exp_cost, prebuilt):
    rows = {}
    for name, s in SCENES.items():
        V = s["V"]; nb = s["nbuckets"]
        # NATIVE backward state-recovery
        nat_flop = V * (NATIVE_EXP * exp_cost + NATIVE_FLOP)
        nat_exp  = V * NATIVE_EXP
        # memory: forward stores bucket checkpoints float4[nb*TILE_PIX], backward reads them
        nat_mem_bytes = 2 * nb * TILE_PIX * CKPT_FLOATS * FLOAT_BYTES  # write+read
        # SCAN backward state-recovery
        if prebuilt:
            # transforms A_k,B_k stored in forward (reuse alpha computation); backward scan has 0 exp
            scan_flop = V * (SCAN_BUILD_FLOP*0 + SCAN_COMPOSE_PER_VISIT + SCAN_CAFTER_FLOP)  # build paid in forward
            scan_exp = 0
            scan_mem_bytes = 2 * V * CKPT_FLOATS * FLOAT_BYTES  # store+read transforms (~= native)
        else:
            scan_flop = V * (SCAN_BUILD_EXP*exp_cost + SCAN_BUILD_FLOP + SCAN_COMPOSE_PER_VISIT + SCAN_CAFTER_FLOP)
            scan_exp = V * SCAN_BUILD_EXP
            scan_mem_bytes = 0  # no checkpoints, no stored transforms (recomputed)
        # depth: native replay sequential within bucket(32); scan parallel prefix log2(V/nbuckets per pixel)
        import math
        nat_depth = BUCKET
        scan_depth = max(1, int(math.ceil(math.log2(max(2, V // max(1,nb) )))))  # ~log2(splats/pixel)
        red = (nat_flop - scan_flop) / nat_flop if nat_flop else 0.0
        rows[name] = {
            "V": V, "nbuckets": nb,
            "native_state_flop": nat_flop, "native_exp": nat_exp,
            "scan_state_flop": scan_flop, "scan_exp": scan_exp,
            "state_flop_reduction": red,
            "native_mem_bytes": nat_mem_bytes, "scan_mem_bytes": scan_mem_bytes,
            "mem_reduction": (nat_mem_bytes - scan_mem_bytes)/nat_mem_bytes if nat_mem_bytes else 0.0,
            "native_sync_depth": nat_depth, "scan_sync_depth": scan_depth,
        }
    return rows

def gate(rows):
    # PASS iff >=20% state-recovery flop reduction in >=2/3 scenes
    wins = sum(1 for r in rows.values() if r["state_flop_reduction"] >= 0.20)
    return wins >= (2*len(rows)//3 + (1 if len(rows)%3 else 0)) or wins >= 2, wins

out = {"_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "model": "state-recovery work = FLOPs to recover (T_k, C_after_k) per valid visit; gradient math excluded (identical both)",
       "exp_cost_sensitivity": {}}
for prebuilt in (True, False):
    for exp_cost in (4, 8, 12, 16, 20):
        rows = model(exp_cost, prebuilt)
        passed, wins = gate(rows)
        key = f"prebuilt={prebuilt},exp_cost={exp_cost}"
        out["exp_cost_sensitivity"][key] = {
            "prebuilt_transform": prebuilt, "exp_cost_flop_equiv": exp_cost,
            "per_scene_reduction": {n: round(r["state_flop_reduction"],4) for n,r in rows.items()},
            "scenes_ge_20pct": wins, "gate4_pass": bool(passed),
        }
# headline: prebuilt transform (fair: forward already computes alpha), exp_cost=8 (SFU-precise expf)
head = model(8, True)
passed, wins = gate(head)
out["headline"] = {
    "assumption": "transform pre-built in forward (alpha already computed there); exp_cost=8 FMA-equiv (precise expf on A100)",
    "per_scene": {n: {k:(v if not isinstance(v,float) else round(v,4)) for k,v in r.items()} for n,r in head.items()},
    "scenes_ge_20pct": wins, "gate4_pass": bool(passed),
    "note": "reduction = (native - scan)/native state-recovery FLOPs; scan trades replay recurrence (11 flop+1exp) for Blelloch compose (14 flop) + C_after(3); saving limited to backward exp recomputation",
}
print(json.dumps(out, indent=1))
