p = "rasterization/include/backward.h"
s = open(p).read()
s = s.replace("        unsigned int* counter_out);\nma once\n", "        unsigned int* counter_out);\n")
open(p,"w").write(s)
print("ma once removed:", "ma once" not in s)

p2 = "torch_bindings/bindings.cpp"
s2 = open(p2).read()
s2 = s2.replace("&rasterization_api::count_valid_g", "&faster_gs::rasterization::count_valid_g")
open(p2,"w").write(s2)
print("binding namespace fixed")
