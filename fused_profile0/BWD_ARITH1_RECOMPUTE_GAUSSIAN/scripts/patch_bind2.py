p = "torch_bindings/bindings.cpp"
s = open(p).read()
add = '    m.def("count_valid_g", &rasterization_api::count_valid_g);\n'
s = s.replace('    m.def("backward", &rasterization_api::backward_wrapper);',
              '    m.def("backward", &rasterization_api::backward_wrapper);\n' + add, 1)
open(p,"w").write(s)
print("patched:", "count_valid_g" in s)
print(s)
