p = 'clockgs/src/core.py'
s = open(p, encoding='utf-8').read()
old = '    rendered_image, radii, _depth = rasterizer(\n'
new = '    rendered_image, radii, _depth, _alpha = rasterizer(\n'
assert old in s
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(s)
import ast; ast.parse(s); print('fixed 4-tuple unpack; syntax OK')
