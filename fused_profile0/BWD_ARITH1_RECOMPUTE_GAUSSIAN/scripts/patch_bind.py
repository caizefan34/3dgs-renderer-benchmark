p = "torch_bindings/bindings.cpp"
s = open(p).read()
add = """    m.def("count_valid_g", &rasterization_api::count_valid_g);
"""
if "count_valid_g" not in s:
    s = s.replace('    m.def("backward", &rasterization_api::backward);\n',
                  '    m.def("backward", &rasterization_api::backward);\n' + add, 1)
open(p,"w").write(s)
print("bindings.cpp patched:", "count_valid_g" in s)
