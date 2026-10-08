import subprocess, re, json
so="/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs/FasterGSFusedCudaBackend/_C.cpython-310-x86_64-linux-gnu.so"
import hashlib
so_sha=hashlib.sha256(open(so,'rb').read()).hexdigest()
out=subprocess.run(["cuobjdump","-res-usage",so], capture_output=True, text=True).stdout
lines=out.splitlines()
recs=[]
i=0
while i<len(lines):
    m=re.match(r"\s*Function\s+(\S+)", lines[i])
    if m:
        name=m.group(1).rstrip(':')
        reg=stack=sh=loc=const=None
        j=i+1
        while j<len(lines) and not lines[j].strip().startswith("Function"):
            mm=re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)\s+CONSTANT\[0\]:(\d+)", lines[j])
            if mm:
                reg=int(mm.group(1)); stack=int(mm.group(2)); sh=int(mm.group(3)); loc=int(mm.group(4)); const=int(mm.group(5)); break
            j+=1
        if reg is not None:
            recs.append({"mangled":name,"reg":reg,"stack":stack,"shared":sh,"local":loc,"const":const})
        i=j
    else:
        i+=1
# block sizes from source (rasterization_config.h)
BS={"blend_backward":256,"blend_cu":256,"preprocess_cu":128,"preprocess_backward":128,
    "adam_step":256,"create_instances":256,"apply_depth_ordering":256,"extract_instance_ranges":256,"extract_bucket_counts":256}
def est_occ(reg, sh, bs):
    # A100 SM: 65536 regs, 164KB shared, 2048 threads, 32 blocks max
    if reg<=0 or bs<=0: return None
    by_regs=65536//(reg*bs)
    by_sh=(164*1024)//sh if sh>0 else 32
    by_threads=2048//bs
    blocks=max(1,min(32, by_regs, by_sh, by_threads))
    threads=blocks*bs
    warps=threads//32
    return {"blocks_per_sm":blocks, "threads_per_sm":threads, "occ_pct":round(100*warps/64,1)}
want=["blend_backward","blend_cu","preprocess_cu","preprocess_backward","adam_step","create_instances","apply_depth_ordering","extract_instance_ranges","extract_bucket_counts"]
seen=set()
result=[]
for r in recs:
    for w in want:
        if w in r["mangled"] and w not in seen:
            seen.add(w)
            bs=BS.get(w,128)
            result.append({"component_kernel":w,"registers_per_thread":r["reg"],"static_shared_bytes":r["shared"],
                           "stack_bytes":r["stack"],"local_bytes":r["local"],"const_bytes":r["const"],
                           "block_size":bs,"occupancy_est":est_occ(r["reg"], r["shared"], bs)})
result.sort(key=lambda x:x["component_kernel"])
print("SO_SHA256", so_sha)
print("N_FUSED_KERNELS_FOUND", len(result), "of", len(want))
print(json.dumps(result, indent=1))
missing=[w for w in want if w not in seen]
print("MISSING", missing)