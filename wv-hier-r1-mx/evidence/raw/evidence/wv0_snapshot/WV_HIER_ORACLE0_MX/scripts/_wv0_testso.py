import torch, importlib.util, os
so="/home/liaoyuanjun/warp_emit_a0/torch_extensions/gsplat_cuda/gsplat_cuda.so"
print("exists", os.path.exists(so), "size", os.path.getsize(so))
# check what symbols it exports vs what gsplat expects
import subprocess
r=subprocess.run(["bash","-lc",f"nm -D {so} 2>/dev/null | grep -c 'T ' ; echo '---'; nm -D {so} 2>/dev/null | grep -iE 'rasterize_to_pixels|fully_fused_projection|isect_tiles' | head"],capture_output=True,text=True)
print(r.stdout)
# Try loading it as the gsplat _C
try:
    spec=importlib.util.spec_from_file_location("gsplat_cuda_pre", so)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print("LOADED; has rasterize_to_pixels:", hasattr(m,"rasterize_to_pixels"))
except Exception as e:
    import traceback; traceback.print_exc()
