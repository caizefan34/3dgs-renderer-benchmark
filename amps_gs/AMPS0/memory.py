import json
# Theoretical optimizer-state memory accounting (objective section 24).
# NOTE: This is THEORETICAL state-byte accounting, NOT measured peak VRAM.
# Measured peak VRAM requires A100 + Faster-GS fused backend (not available locally).
attrs=[("means",3,"fp32"),("scales",3,"fp32"),("quats",4,"fp32"),("opacity",1,"fp32"),
       ("sh0",3,"fp32"),("sh_rest",45,"bf16_in_C2plus")]
def state_bytes(dtype): return 2 if dtype=="bf16" else 4
rows=[]
total_params=0; total_mv_fp32=0
for name,n,dt in attrs:
    param_b=n*4
    mv_b=n*2*4   # m+v in fp32 baseline
    total_params+=n; total_mv_fp32+=mv_b
    rows.append({"attribute":name,"n_scalars":n,"param_bytes_per_gauss":param_b,
                 "m_v_fp32_bytes_per_gauss":mv_b,"m_v_bf16_bytes_per_gauss":n*2*2})
# candidate totals (m/v bytes only)
cands={"C0(all fp32)":{"means":"fp32","scales":"fp32","quats":"fp32","opacity":"fp32","sh0":"fp32","sh_rest":"fp32"},
       "C1":{"sh_rest":"mixed_v_bf16"},
       "C2":{"sh_rest":"bf16"},
       "C3":{"sh_rest":"bf16","sh0":"bf16","opacity":"bf16"},
       "C4":{"*":"bf16"}}
def mv_total(pol):
    tot=0
    for name,n,_ in attrs:
        dt=pol.get("*") or pol.get(name,"fp32")
        if dt=="mixed_v_bf16":  # C1: m fp32, v bf16 -> n*(4+2)
            tot+=n*(4+2)
        elif dt=="bf16":
            tot+=n*2*2
        else:
            tot+=n*2*4
    return tot
mem={"schema":"AMPS0_MEMORY","generated":"2026-10-05",
 "scope":"THEORETICAL optimizer-state byte accounting (objective section 24). NOT measured peak VRAM.",
 "measured_peak_vram_available":False,
 "measured_peak_vram_reason":"Requires A100 + Faster-GS fused backend; local host is RTX 5070 8GB and cannot build the fused CUDA backend. The FGS fused runs (fgs_strong_baseline_final) report peak VRAM 8.96-11.12 GiB but do NOT break out optimizer-state bytes separately.",
 "per_attribute_state_bytes_per_gauss":rows,
 "total_scalars":total_params,
 "total_m_v_fp32_bytes_per_gauss":total_mv_fp32,
 "candidate_m_v_bytes_per_gauss":{c:mv_total(p) for c,p in cands.items()},
 "candidate_m_v_reduction_vs_C0":{c:round(1-mv_total(p)/(total_params*8),4) for c,p in cands.items()},
 "note":"m+v = 2 moments x 4 bytes (fp32) = 8 bytes/scalar baseline; bf16 = 4 bytes/scalar. C1 (v only bf16) = 6 bytes/scalar for sh_rest. These are state-byte ONLY; they exclude params, grads, conversion buffers, densification workspace, and peak VRAM. Per section 24, memory-reduction claims MUST be measured, not theoretical. No memory-reduction claim is made by AMPS0.",
 "claim_policy":"No memory reduction claimed from theoretical state bytes (objective section 24)."}
json.dump(mem, open("amps_gs/AMPS0/AMPS0_MEMORY.json","w"), indent=2)
print("MEMORY written. total scalars:", total_params, "m/v fp32 bytes/gauss:", total_mv_fp32)
for c in cands: print(f"  {c}: {mem['candidate_m_v_bytes_per_gauss'][c]} bytes/gauss, reduction {mem['candidate_m_v_reduction_vs_C0'][c]}")
