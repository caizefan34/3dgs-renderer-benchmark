GI="/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference"
p=GI+"/dpc_stage1.cu"
t=open(p).read()
if "#include <ATen/ops/arange.h>" not in t:
    t=t.replace("#include <ATen/ops/empty.h>\n",
                "#include <ATen/ops/empty.h>\n#include <ATen/ops/arange.h>\n",1)
old="means2d.const_data_ptr<float>(), conics.const_data_ptr<float>()"
new="reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()), reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>())"
assert t.count(old)==1, "anchor %d"%t.count(old)
t=t.replace(old,new,1)
open(p,"w").write(t)
print("cu fixed OK")