import json, os
ROOT = "amps_gs/AMPS0"
os.makedirs(ROOT, exist_ok=True)

attrs = [
  {"name":"means","group":"geometry","n":3},
  {"name":"scales","group":"geometry","n":3},
  {"name":"quats","group":"geometry","n":4},
  {"name":"opacity","group":"geometry","n":1},
  {"name":"sh0","group":"appearance","n":3,"subtype":"sh0"},
  {"name":"sh_rest","group":"appearance","n":45,"subtype":"sh_rest"},
]
total_scalars = sum(a["n"] for a in attrs)

FP32 = 7*4   # R param, R grad, R m, R v, W param, W m, W v = 28
BF16 = 4+4+2+2+4+2+2  # 20

candidates = {
  "C0": {"label":"reference (all FP32 m/v)","mv":{}},
  "C1": {"label":"conservative: sh_rest m=FP32 v=BF16","mv":{"sh_rest":{"m":"fp32","v":"bf16"}}},
  "C2": {"label":"PRIMARY: sh_rest m/v BF16","mv":{"sh_rest":{"m":"bf16","v":"bf16"}}},
  "C3": {"label":"expanded: sh_rest+sh0+opacity BF16","mv":{"sh_rest":"bf16","sh0":"bf16","opacity":"bf16"}},
  "C4": {"label":"ceiling: ALL m/v BF16","mv":{"*":"bf16"}},
}

def mv_bf16_set(cand):
    m = candidates[cand]["mv"]
    s = set()
    if "*" in m:
        for a in attrs: s.add(a.get("subtype") or a["name"])
    else:
        for k,v in m.items():
            if v=="bf16" or (isinstance(v,dict) and v.get("m")=="bf16") or (isinstance(v,dict) and v.get("v")=="bf16"):
                s.add(k)
    return s

def opt_bytes(cand):
    bf = mv_bf16_set(cand)
    total=0; mv=0; total_mv=0
    for a in attrs:
        key = a.get("subtype") or a["name"]
        is_bf = key in bf
        per = BF16 if is_bf else FP32
        total += a["n"]*per
        mv += a["n"]*(4 if is_bf else 8)   # m+v R+W bytes
        total_mv += a["n"]*16
    return total, mv, total_mv

def e2e(p,s): return 1.0/((1-p)+p*(1-s))

pvals = [0.0594,0.14,0.20,0.30,0.40]
rows=[]
for cand in candidates:
    total,mv,total_mv = opt_bytes(cand)
    s_total = (total_scalars*FP32 - total)/(total_scalars*FP32)
    s_mv = (total_mv - mv)/total_mv
    row={"cand":cand,"label":candidates[cand]["label"],
         "opt_bytes_per_gauss":total,"opt_bytes_fp32":total_scalars*FP32,
         "opt_traffic_reduction":round(s_total,4),
         "mv_only_traffic_reduction":round(s_mv,4),
         "e2e_at_p":{str(p):round(e2e(p,s_total),4) for p in pvals}}
    rows.append(row)

out={
 "schema":"AMPS0_TRAFFIC_MODEL","generated":"2026-10-05",
 "host":"RTX 5070 Laptop (local); analytical model host-independent",
 "attribute_taxonomy":attrs,"total_scalars":total_scalars,
 "adam_bytes_per_scalar_fp32":FP32,"adam_bytes_per_scalar_bf16_mv":BF16,
 "adam_access_pattern":"R param(4) R grad(4) R m R v W param(4) W m W v",
 "assumption_memory_bound_optimizer":True,
 "assumption_note":"Adam streaming kernel; AI ~0.36 FLOP/byte << A100 ridge ~20; optimizer time scales with HBM bytes for large N. BF16<->FP32 convert = 1 instr on A100 (negligible vs ~10 FLOP Adam math).",
 "e2e_formula":"T_new/T_old=(1-p)+p*(1-s); speedup=1/that; p=optimizer fraction; s=opt traffic reduction",
 "candidates":rows,
 "gates":{"traffic_gate_e2e_min":1.05,"e2e_promote_min":1.08,"e2e_strong_min":1.15,
   "measured_p":{"full_adam_M0_room15k":0.0594,"sh_adam_LAVOS0_all_scenes":0.14},
   "note":"At measured p (0.059-0.14), even C4(all BF16) yields E2E<1.05x. DROP_TRAFFIC_HEADROOM triggered."}
}
json.dump(out, open(ROOT+"/AMPS0_TRAFFIC_MODEL.json","w"), indent=2)
print("total_scalars",total_scalars,"FP32",FP32,"BF16mv",BF16)
print("cand | opt_bytes | s_total | e2e@.0594 | e2e@.14 | e2e@.20 | e2e@.30 | e2e@.40")
for r in rows:
    e=r["e2e_at_p"]
    print(r["cand"],"|",r["opt_bytes_per_gauss"],"|",r["opt_traffic_reduction"],"|",
      e["0.0594"],"|",e["0.14"],"|",e["0.2"],"|",e["0.3"],"|",e["0.4"])
