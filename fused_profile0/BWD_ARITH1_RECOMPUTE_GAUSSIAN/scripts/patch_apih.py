p = "rasterization/include/rasterization_api.h"
s = open(p).read()
if "#include \"backward.h\"" not in s:
    s = s.replace("#include <tuple>", "#include <tuple>\n#include \"backward.h\"", 1)
open(p,"w").write(s)
print("rasterization_api.h includes backward.h:", "#include \"backward.h\"" in s)
