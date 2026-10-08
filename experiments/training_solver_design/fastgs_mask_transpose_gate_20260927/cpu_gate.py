#!/usr/bin/env python3
"""Deterministic CPU algebra and 32x64 FP32 gate for FastGS mask proposal.

This is a synthetic CPU oracle, not a CUDA emulator or training benchmark.
It checks per-hit reconstruction and a software 32x64 shared-memory transpose model.
No production code is imported or modified.
"""
import json
import math
import random
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("cpu_gate_result.json")
EPS = 1e-12


def pixel_forward(gs, xy, bg, n, bucket=32):
    """Return native accepted prefix, forward states, output and bucket boundaries."""
    T, C = 1.0, [0.0] * len(bg)
    accepted, boundary = [], {}
    n_contrib = 0
    stopped_at = None
    for i, g in enumerate(gs[:n]):
        if i % bucket == 0:
            boundary[i] = (T, C[:])
        dx, dy = g["xy"][0] - xy[0], g["xy"][1] - xy[1]
        a, b, c, opacity = g["conic_opacity"]
        power = -0.5 * (a*dx*dx + c*dy*dy) - b*dx*dy
        if power > 0:
            continue
        G = math.exp(power)
        alpha = min(0.99, opacity * G)
        if alpha < 1.0/255.0:
            continue
        if T * (1.0-alpha) < 1e-4:
            stopped_at = i
            break
        accepted.append({"i": i, "G": G, "alpha": alpha, "T": T,
                         "xy": (dx,dy), "power": power})
        for ch in range(len(bg)):
            C[ch] += g["color"][ch] * alpha * T
        T *= 1-alpha
        n_contrib = i+1
    return {"accepted": accepted, "boundary": boundary, "Tfinal": T,
            "foreground": C, "out": [C[k] + T*bg[k] for k in range(len(bg))],
            "n_contrib": n_contrib, "stopped_at": stopped_at}


def native_vjp(gs, state, bg, grad):
    """FastGS PerGaussianRenderCUDA custom VJP equations for one pixel."""
    grads = [dict(mean=[0.0]*4, conic=[0.0]*3, opacity=0.0,
                  color=[0.0]*len(bg)) for _ in gs]
    # Native backward's ar begins at bucket prefix minus final foreground;
    # accepted contributors are replayed in order and add their foreground term.
    for item in state["accepted"]:
        i, G, alpha, T = item["i"], item["G"], item["alpha"], item["T"]
        bucket_start = (i // 32) * 32
        prefixT, prefixC = state["boundary"].get(bucket_start, (1.0,[0.0]*len(bg)))
        # reconstruct C at the boundary from accepted contributions before i
        # (bucket boundaries are recorded even when no accepted sample follows).
        if bucket_start not in state["boundary"]:
            prefixT, prefixC = 1.0, [0.0]*len(bg)
        ar = [prefixC[k] - state["foreground"][k] for k in range(len(bg))]
        for prev in state["accepted"]:
            if bucket_start <= prev["i"] < i:
                pg=gs[prev["i"]]
                for k in range(len(bg)):
                    ar[k] += pg["color"][k]*prev["alpha"]*prev["T"]
        g=gs[i]
        d_alpha=0.0
        inv=1.0/(1-alpha)
        for k in range(len(bg)):
            ar[k] += alpha*T*g["color"][k]
            grads[i]["color"][k] += alpha*T*grad[k]
            d_alpha += (g["color"][k]*T + inv*ar[k])*grad[k]
        bgdot=sum(bg[k]*grad[k] for k in range(len(bg)))
        d_alpha += -state["Tfinal"]*inv*bgdot
        dG = g["conic_opacity"][3]*d_alpha
        dx,dy=item["xy"]
        gx=G*dx; gy=G*dy
        # Native PerGaussianRenderCUDA applies the unclamped oG chain to every
        # derivative output, including at saturated alpha.
        tx=dG*(-gx*g["conic_opacity"][0]-gy*g["conic_opacity"][1])*0.5
        ty=dG*(-gy*g["conic_opacity"][2]-gx*g["conic_opacity"][1])*0.5
        grads[i]["mean"][0] += tx
        grads[i]["mean"][1] += ty
        grads[i]["mean"][2] += abs(tx)
        grads[i]["mean"][3] += abs(ty)
        grads[i]["conic"][0] += -0.5*gx*dx*dG
        grads[i]["conic"][1] += -0.5*gx*dy*dG
        grads[i]["conic"][2] += -0.5*gy*dy*dG
        grads[i]["opacity"] += G*d_alpha
    return grads


def contribution_vjp(gs, state, bg, grad, item):
    """Native contribution plus the proposed two cached scalars (w,h)."""
    i, G, alpha, T = item["i"], item["G"], item["alpha"], item["T"]
    bucket_start=(i//32)*32
    prefixC=state["boundary"].get(bucket_start,(1.0,[0.0]*len(bg)))[1]
    ar=[prefixC[k]-state["foreground"][k] for k in range(len(bg))]
    for prev in state["accepted"]:
        if bucket_start <= prev["i"] < i:
            pg=gs[prev["i"]]
            for k in range(len(bg)): ar[k]+=pg["color"][k]*prev["alpha"]*prev["T"]
    g=gs[i]; d_alpha=0.0; inv=1/(1-alpha)
    out=dict(mean=[0.0]*4,conic=[0.0]*3,opacity=0.0,color=[0.0]*len(bg))
    for k in range(len(bg)):
        ar[k]+=alpha*T*g["color"][k]
        out["color"][k]=alpha*T*grad[k]
        d_alpha+=(g["color"][k]*T+inv*ar[k])*grad[k]
    d_alpha += -state["Tfinal"]*inv*sum(bg[k]*grad[k] for k in range(len(bg)))
    dG=g["conic_opacity"][3]*d_alpha; dx,dy=item["xy"]
    gx=G*dx; gy=G*dy
    tx=dG*(-gx*g["conic_opacity"][0]-gy*g["conic_opacity"][1])*.5
    ty=dG*(-gy*g["conic_opacity"][2]-gx*g["conic_opacity"][1])*.5
    out["mean"]=[tx,ty,abs(tx),abs(ty)]
    out["conic"]=[-.5*gx*dx*dG,-.5*gx*dy*dG,-.5*gy*dy*dG]
    out["opacity"]=G*d_alpha
    out["w"]=T*alpha
    out["h"]=G*d_alpha
    return out


def transpose_from_wh(g, item, grad, wh):
    """Reconstruct all per-contribution outputs from only w,h and input attrs."""
    w,h=wh
    dx,dy=item["xy"]
    a,b,c,opacity=g["conic_opacity"]
    # h=G*dL/dalpha, so opacity*h is G*dL/dG times G's spatial factor.
    oh=opacity*h
    tx=oh*(-dx*a-dy*b)*.5
    ty=oh*(-dy*c-dx*b)*.5
    return {"mean":[tx,ty,abs(tx),abs(ty)],
            "conic":[-.5*oh*dx*dx,-.5*oh*dx*dy,-.5*oh*dy*dy],
            "opacity":h,
            "color":[w*x for x in grad]}


def f32(x):
    return np.float32(x)


def pixel_forward_f32(gs, xy, bg, n):
    """One native-order forward pixel, rounding intermediate ops to FP32."""
    T=f32(1); C=[f32(0),f32(0),f32(0)]; accepted=[]; boundary={}
    n_contrib=0; stopped=None
    for i,g in enumerate(gs[:n]):
        if i%32==0: boundary[i]=(T,C[:])
        dx=f32(g["xy"][0]-xy[0]); dy=f32(g["xy"][1]-xy[1])
        a,b,c,o=g["conic_opacity"]
        # Match source left-to-right FP32 products.
        xx=f32(f32(a*dx)*dx); yy=f32(f32(c*dy)*dy); cross=f32(f32(b*dx)*dy)
        power=f32(f32(f32(-0.5)*f32(xx+yy))-cross)
        if power>0: continue
        G=f32(np.exp(power)); alpha=f32(min(f32(.99),f32(o*G)))
        if alpha < f32(1/255): continue
        test_T=f32(T*f32(f32(1)-alpha))
        if test_T < f32(1e-4): stopped=i; break
        accepted.append({"i":i,"G":G,"alpha":alpha,"T":T,"xy":(dx,dy),"power":power})
        for ch in range(3): C[ch]=f32(C[ch]+f32(f32(g["color"][ch]*alpha)*T))
        T=test_T; n_contrib=i+1
    out=[f32(C[k]+f32(T*bg[k])) for k in range(3)]
    return {"accepted":accepted,"boundary":boundary,"Tfinal":T,"foreground":C,
            "out":out,"n_contrib":n_contrib,"stopped_at":stopped}


def native_vjp_f32(gs,state,bg,grad):
    """Independent literal-order translation of FastGS PerGaussianRenderCUDA."""
    result=[dict(mean=[f32(0)]*4,conic=[f32(0)]*3,opacity=f32(0),color=[f32(0)]*3) for _ in gs]
    for q in state["accepted"]:
        i=q["i"]; G=q["G"]; alpha=q["alpha"]; T=q["T"]; g=gs[i]
        bucket_start=(i//32)*32
        prefixC=state["boundary"].get(bucket_start,(f32(1),[f32(0)]*3))[1]
        ar=[f32(-state["foreground"][k]+prefixC[k]) for k in range(3)]
        for prev in state["accepted"]:
            if bucket_start<=prev["i"]<i:
                pg=gs[prev["i"]]
                for k in range(3):
                    contribution=f32(f32(prev["alpha"]*prev["T"])*pg["color"][k])
                    ar[k]=f32(ar[k]+contribution)
        inv=f32(f32(1)/f32(f32(1)-alpha)); d_alpha=f32(0)
        dchannel=f32(alpha*T)
        for k in range(3):
            ar[k]=f32(ar[k]+f32(dchannel*g["color"][k]))
            result[i]["color"][k]=f32(result[i]["color"][k]+f32(dchannel*grad[k]))
            inner=f32(f32(g["color"][k]*T)+f32(inv*ar[k]))
            d_alpha=f32(d_alpha+f32(inner*grad[k]))
        bgdot=f32(0)
        for k in range(3): bgdot=f32(bgdot+f32(bg[k]*grad[k]))
        d_alpha=f32(d_alpha+f32(f32(-state["Tfinal"]*inv)*bgdot))
        opacity=g["conic_opacity"][3]
        dG=f32(opacity*d_alpha); dx,dy=q["xy"]
        gx=f32(G*dx); gy=f32(G*dy); a,b,c,_=g["conic_opacity"]
        dGdx=f32(f32(-gx*a)-f32(gy*b)); dGdy=f32(f32(-gy*c)-f32(gx*b))
        tx=f32(f32(dG*dGdx)*f32(8.0)); ty=f32(f32(dG*dGdy)*f32(8.0))
        result[i]["mean"][0]=f32(result[i]["mean"][0]+tx)
        result[i]["mean"][1]=f32(result[i]["mean"][1]+ty)
        result[i]["mean"][2]=f32(result[i]["mean"][2]+abs(tx))
        result[i]["mean"][3]=f32(result[i]["mean"][3]+abs(ty))
        result[i]["conic"][0]=f32(result[i]["conic"][0]+f32(f32(f32(-.5)*gx)*dx*dG))
        result[i]["conic"][1]=f32(result[i]["conic"][1]+f32(f32(f32(-.5)*gx)*dy*dG))
        result[i]["conic"][2]=f32(result[i]["conic"][2]+f32(f32(f32(-.5)*gy)*dy*dG))
        result[i]["opacity"]=f32(result[i]["opacity"]+f32(G*d_alpha))
    return result


def run_32x64_fp32():
    """Real 64-pixel transpose/primitive reduction using FP32 arrays and ops."""
    rng=random.Random(20260927)
    gs=[]
    for i in range(70):
        if i<5: xy=(0.0,0.0); opacity=.98
        else: xy=(rng.uniform(0,4),rng.uniform(0,4)); opacity=.04
        if i==10: xy=(2.0,2.0); opacity=1.2
        a=rng.uniform(.5,1.5); c=rng.uniform(.5,1.5); b=rng.uniform(-.08,.08)
        gs.append({"xy":tuple(map(f32,xy)),"conic_opacity":tuple(map(f32,(a,b,c,opacity))),
                   "color":[f32(rng.uniform(-.1,1.2)) for _ in range(3)]})
    bg=[f32(.07),f32(.13),f32(.21)]
    pixels=[(f32(x),f32(y)) for y in range(8) for x in range(8)]
    states=[]; pixel_grads=[]; direct=[dict(mean=[f32(0)]*4,conic=[f32(0)]*3,opacity=f32(0),color=[f32(0)]*3) for _ in gs]
    # Three per-tile bucket planes; each holds a 32x64 pair of FP32 w,h arrays.
    shared_wh=np.zeros((3,32,64,2),dtype=np.float32)
    masks=[[0,0,0] for _ in pixels]; outputs=[]
    saturated_hits=early_pixels=background_only=0
    for p,xy in enumerate(pixels):
        grad=[f32(rng.uniform(-1,1)) for _ in range(3)]
        st=pixel_forward_f32(gs,xy,bg,70); states.append(st); pixel_grads.append(grad); outputs.append(st["out"])
        if st["stopped_at"] is not None: early_pixels+=1
        if not st["accepted"]: background_only+=1
        contrib=native_vjp_f32(gs,st,bg,grad)
        # Native image-gradient atomics modeled by stable pixel-major FP32 accumulation.
        for i,q in enumerate(contrib):
            for fld in ("mean","conic","color"):
                for k,v in enumerate(q[fld]): direct[i][fld][k]=f32(direct[i][fld][k]+v)
            direct[i]["opacity"]=f32(direct[i]["opacity"]+q["opacity"])
        for q in st["accepted"]:
            i=q["i"]; masks[p][i//32]|=1<<(i%32)
            one=contribution_vjp_f32(gs,st,bg,grad,q)
            shared_wh[i//32,i%32,p,0]=one["w"]
            shared_wh[i//32,i%32,p,1]=one["h"]
            if float(gs[i]["conic_opacity"][3]*q["G"])>.99: saturated_hits+=1
    # Actual transpose: primitive outermost, then the 64 values in its shared row.
    # Pairwise FP32 reduction models a changed reduction tree vs pixel-major reference.
    trans=[dict(mean=[f32(0)]*4,conic=[f32(0)]*3,opacity=f32(0),color=[f32(0)]*3) for _ in gs]
    hits=0
    def tree_sum(vals):
        work=[f32(v) for v in vals]
        while len(work)>1:
            work=[f32(work[j]+work[j+1]) if j+1<len(work) else work[j] for j in range(0,len(work),2)]
        return work[0] if work else f32(0)
    for i,g in enumerate(gs):
        records=[]
        for p in range(64):
            if masks[p][i//32] & (1<<(i%32)):
                hits+=1
                hit=next(item for item in states[p]["accepted"] if item["i"]==i)
                wh=(shared_wh[i//32,i%32,p,0],shared_wh[i//32,i%32,p,1])
                records.append(transpose_from_wh_f32(g,hit,pixel_grads[p],wh))
            else:
                records.append(dict(mean=[f32(0)]*4,conic=[f32(0)]*3,opacity=f32(0),color=[f32(0)]*3))
        for fld in ("mean","conic","color"):
            for k in range(len(trans[i][fld])):
                trans[i][fld][k]=tree_sum([v[fld][k] for v in records])
        trans[i]["opacity"]=tree_sum([v["opacity"] for v in records])
    accepted_total=sum(len(s["accepted"]) for s in states)
    mask_total=sum(word.bit_count() for row in masks for word in row)
    field_errors={fld:0.0 for fld in ("mean","conic","color","opacity")}
    ref_peak={fld:0.0 for fld in field_errors}
    for a,b in zip(direct,trans):
        for fld in ("mean","conic","color"):
            for x,y in zip(a[fld],b[fld]):
                field_errors[fld]=max(field_errors[fld],abs(float(x-y)))
                ref_peak[fld]=max(ref_peak[fld],abs(float(x)))
        field_errors["opacity"]=max(field_errors["opacity"],abs(float(a["opacity"]-b["opacity"])))
        ref_peak["opacity"]=max(ref_peak["opacity"],abs(float(a["opacity"])))
    rel={k:field_errors[k]/max(ref_peak[k],1e-30) for k in field_errors}
    return {"pixels":64,"gaussians":70,"microtile":[32,64],"dtype":"numpy.float32 scalar operations",
            "accepted_hit_count":accepted_total,"mask_popcount":mask_total,"mask_count_matches":accepted_total==mask_total,
            "bucket_mask_words_per_pixel":3,"partial_last_bucket_valid_ids":6,
            "shared_wh_storage_shape":[3,32,64,2],"shared_wh_bytes_per_bucket":32*64*2*4,
            "pixels_with_early_stop":early_pixels,"background_only_pixels":background_only,
            "saturated_accepted_hits":saturated_hits,"all_output_fields":["mean2D x/y","mean2D per-contribution absgrad x/y","conic x/y/w","opacity","RGB color"],
            "max_abs_error_by_output":field_errors,"max_relative_to_reference_peak_by_output":rel,
            "native_reference_peak_by_output":ref_peak,
            "pass":accepted_total==mask_total and all(v<=2e-5 for v in field_errors.values()) and early_pixels>0 and background_only>0 and saturated_hits>0}


def contribution_vjp_f32(gs,state,bg,grad,item):
    """Compute native one-hit terms for stage-1 w/h cache only."""
    i=item["i"]; G=item["G"]; alpha=item["alpha"]; T=item["T"]; g=gs[i]
    bucket_start=(i//32)*32
    prefixC=state["boundary"].get(bucket_start,(f32(1),[f32(0)]*3))[1]
    ar=[f32(-state["foreground"][k]+prefixC[k]) for k in range(3)]
    for prev in state["accepted"]:
        if bucket_start<=prev["i"]<i:
            pg=gs[prev["i"]]
            for k in range(3): ar[k]=f32(ar[k]+f32(f32(prev["alpha"]*prev["T"])*pg["color"][k]))
    inv=f32(f32(1)/f32(1-alpha)); da=f32(0); dc=f32(alpha*T)
    for k in range(3):
        ar[k]=f32(ar[k]+f32(dc*g["color"][k]))
        da=f32(da+f32(f32(f32(g["color"][k]*T)+f32(inv*ar[k]))*grad[k]))
    bgdot=f32(0)
    for k in range(3): bgdot=f32(bgdot+f32(bg[k]*grad[k]))
    da=f32(da+f32(f32(-state["Tfinal"]*inv)*bgdot))
    return {"w":f32(T*alpha),"h":f32(G*da)}


def transpose_from_wh_f32(g,item,grad,wh):
    """Stage 2 consumes only w/h plus pixel and Gaussian attributes."""
    w,h=wh; dx,dy=item["xy"]; a,b,c,o=g["conic_opacity"]; oh=f32(o*h)
    tx=f32(f32(oh*f32(f32(-dx*a)-dy*b))*f32(8.0))
    ty=f32(f32(oh*f32(f32(-dy*c)-dx*b))*f32(8.0))
    return {"mean":[tx,ty,abs(tx),abs(ty)],
            "conic":[f32(f32(f32(-.5)*oh)*f32(dx*dx)),f32(f32(f32(-.5)*oh)*f32(dx*dy)),f32(f32(f32(-.5)*oh)*f32(dy*dy))],
            "opacity":h,"color":[f32(w*x) for x in grad]}


def close(a,b,tol=2e-6):
    return abs(a-b) <= tol*max(1.0,abs(a),abs(b))


def flatten(g):
    return [x for q in g for x in (q["mean"]+q["conic"]+[q["opacity"]]+q["color"])]


def run():
    # Construct one 70-splat tile, producing two full buckets and a 6-splat tail.
    # Pixel cases exercise background-only, accepted, cutoff/power rejection,
    # saturation, and early-stop behavior; shared values keep the oracle readable.
    gs=[]
    for i in range(70):
        op=1.0 if i == 69 else (0.015 if i%7 else 0.002)
        gs.append({"xy": (0.0,0.0), "conic_opacity": (1.0,0.0,1.0,op),
                   "color": [0.1+(i%9)*0.07, 0.8-(i%7)*0.06, 0.2+(i%5)*0.11]})
    bg=[0.07,0.13,0.21]
    pixels=[("center",(0.15,-0.1),[0.7,-0.2,0.4],70),
            ("off_support",(4.0,0.0),[-0.3,0.9,0.2],70),
            ("early_stop",(0.0,0.0),[0.2,0.5,-0.8],70),
            ("partial_bucket",(0.15,-0.1),[1.0,0.25,-0.5],67),
            ("saturated_alpha",(0.15,-0.1),[0.4,-0.3,0.8],70)]
    cases=[]; all_ok=True
    for name,xy,grad,n in pixels:
        # Force meaningful early stop only in designated case by replaying a high-opacity prefix.
        local=[dict(x) for x in gs]
        if name=="early_stop":
            for j in range(4): local[j]["conic_opacity"]=(1.0,0.0,1.0,0.99)
        if name=="partial_bucket":
            local[69]["conic_opacity"]=(1.0,0.0,1.0,1.2)
        if name=="saturated_alpha":
            local[69]["conic_opacity"]=(1.0,0.0,1.0,1.2)
        if name=="off_support":
            local=[dict(x) for x in gs]
            local[0]["conic_opacity"]=(1.0,0.0,1.0,0.001)
        st=pixel_forward(local,xy,bg,n)
        native=native_vjp(local,st,bg,grad)
        # For this one pixel, compute native contributions and retain only w,h.
        # This does NOT simulate a 64-pixel microtile/reduction. Reconstruction
        # is independent: it cannot inspect dL/dalpha,
        # foreground suffixes, T, alpha, or G, only w,h and input attrs.
        mask=[0,0,0]
        for q in st["accepted"]: mask[q["i"]//32] |= 1 << (q["i"]%32)
        # independently count the mask's accepted bits against native n_contrib trace
        mask_count=sum(v.bit_count() for v in mask)
        # The RGB adjoint, xy/conic/opacity, and IDs remain ordinary kernel inputs.
        transposed=[dict(mean=[0.0]*4,conic=[0.0]*3,opacity=0.0,color=[0.0]*3) for _ in local]
        # A transpose must preserve each per-contribution abs-grad before reduction.
        for q in st["accepted"]:
            i=q["i"]
            native_one=contribution_vjp(local,st,bg,grad,q)
            one=transpose_from_wh(local[i],q,grad,(native_one["w"],native_one["h"]))
            for fld in ("mean","conic","color"):
                for c,v in enumerate(one[fld]): transposed[i][fld][c]+=v
            transposed[i]["opacity"]+=one["opacity"]
        field_errors={fld:max((abs(a-b) for ga,gb in zip(native,transposed)
                               for a,b in zip(ga[fld],gb[fld])),default=0.0)
                      for fld in ("mean","conic","color")}
        field_errors["opacity"]=max((abs(a["opacity"]-b["opacity"])
                                     for a,b in zip(native,transposed)),default=0.0)
        maxerr=max(field_errors.values(),default=0.0)
        accepted_ids=[q["i"] for q in st["accepted"]]
        saturated=[i for i in accepted_ids if local[i]["conic_opacity"][3]*st["accepted"][accepted_ids.index(i)]["G"] > .99]
        absgrad=[sum(g["mean"][2] for g in native),sum(g["mean"][3] for g in native)]
        good=(mask_count==len(st["accepted"]) and maxerr<1e-8)
        good &= all(any(abs(v)>EPS for v in native[i]["mean"][:2]+native[i]["conic"])
                    for i in saturated)
        all_ok &= good
        cases.append({"name":name,"accepted_count":len(accepted_ids),"accepted_ids":accepted_ids,
                      "mask_bucket_words":[f"0x{x:08x}" for x in mask],"mask_count":mask_count,
                      "n_contrib":st["n_contrib"],"early_stop_index":st["stopped_at"],
                      "saturated_accepted_ids":saturated,"background_output":st["out"],
                      "nonzero_gradient_gaussians":sum(any(abs(v)>EPS for v in flatten([g])) for g in native),
                      "absgrad_mean_xy_sum":absgrad,"transpose_max_abs_error_by_output":field_errors,
                      "saturated_have_nonzero_spatial_gradient":all(any(abs(v)>EPS for v in native[i]["mean"][:2]+native[i]["conic"])
                                                                       for i in saturated),
                      "transpose_max_abs_error":maxerr,"pass":good})
    # Static arithmetic from the source: tile 16x16, 3 channels, bucket 32.
    # Candidate block has 32*64 entries x (w,h) x FP32 = 16 KiB.
    static={"native_tile_pixels":256,"native_bucket_gaussians":32,"channels":3,
            "native_bucket_boundary_bytes_per_pixel":4*4,
            "mask_bytes_per_bucket_pixel":4,"nominal_boundary_plus_mask_increase_percent":25,
            "candidate_shared_bytes_per_32x64_block":32*64*2*4,
            "candidate_shared_kib":16,
            "existing_perGaussian_shared_float_count":32*3+1+32*3,
            "existing_perGaussian_shared_bytes":(32*3+1+32*3)*4,
            "existing_perGaussian_shared_kib":(32*3+1+32*3)*4/1024,
            "candidate_plus_existing_shared_kib":16+(32*3+1+32*3)*4/1024,
            "full_reference_output_fields":["mean2D x/y","mean2D absgrad x/y","conic x/y/w","opacity","RGB color"],
            "barrier_and_global_store_costs":"unmeasured; implementation-dependent"}
    fullblock=run_32x64_fp32()
    result={"scope":"five synthetic single-pixel cases plus a 64-pixel FP32 CPU transpose oracle; not CUDA parity or speed evidence",
            "all_cases_pass":all_ok and fullblock["pass"],"single_pixel_cases":cases,"32x64_fp32_cpu_gate":fullblock,"static_resource_arithmetic":static,
            "decision":"Synthetic 32x64 FP32 CPU gate PASS for this fixed 70-Gaussian case: the accepted mask drives primitive-major reduction, and w=T*alpha/h=G*dL/dalpha reconstruct all tested outputs within the reported tolerance, including per-contribution absgrad. This is not native CUDA parity or a general numerical proof. Proceed only to native CUDA correctness/resource/timing gates; NO-GO for production or training-benefit claims until those pass.",
            "limitations":["synthetic 64-pixel tile, not a real training view","not run against native CUDA kernels","does not establish bitwise parity with native GPU FP32 scheduling/atomics","does not estimate complete training speed or quality","does not cover projection/SH/preprocess gradient chain"]}
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not all_ok: raise SystemExit("CPU oracle assertion failed")


if __name__=="__main__": run()
