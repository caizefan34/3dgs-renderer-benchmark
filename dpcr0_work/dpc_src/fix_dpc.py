import io
GI="/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference"

# --- fix dpc_stage1.h ---
p=GI+"/dpc_stage1.h"
t=open(p).read()
t=t.replace("#include <ATen/core/Tensor.h>\n#include <cstdint>",
            "#include <ATen/core/Tensor.h>\n#include <c10/util/Optional.h>\n#include <cstdint>")
t=t.replace("        at::Tensor backgrounds,    // [I, CDIM] f32 (may be undefined)",
            "        const c10::optional<at::Tensor> &backgrounds, // [I, CDIM] f32 or undefined")
t=t.replace("        at::Tensor active_tiles,   // [AT] int32 or undefined",
            "        const c10::optional<at::Tensor> &active_tiles, // [AT] int32 or undefined")
assert "const c10::optional<at::Tensor> &backgrounds" in t and "const c10::optional<at::Tensor> &active_tiles" in t
open(p,"w").write(t)
print("h fixed")

# --- fix dpc_stage1.cu ---
p=GI+"/dpc_stage1.cu"
t=open(p).read()
t=t.replace("#include <ATen/ops/empty.h>\n",
            "#include <ATen/ops/empty.h>\n#include <ATen/ops/arange.h>\n",1)
old="                means2d.const_data_ptr<float>(), conics.const_data_ptr<float>(), colors_eval.const_data_ptr<float>(), opacities.const_data_ptr<float>(), bg_ptr,              \\"
new="                reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()), reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()), colors_eval.const_data_ptr<float>(), opacities.const_data_ptr<float>(), bg_ptr,              \\"
assert t.count(old)==1, "macro anchor count %d"%t.count(old)
t=t.replace(old,new,1)
assert "#include <ATen/ops/arange.h>" in t
open(p,"w").write(t)
print("cu fixed")