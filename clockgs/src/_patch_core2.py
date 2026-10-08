p = 'clockgs/src/core.py'
s = open(p, encoding='utf-8').read()
old = '    scales = rotations = cov3D_precomp = None\n    shs = colors_precomp = None\n'
new = ('    scales = None\n    rotations = None\n    cov3D_precomp = None\n'
       '    shs = colors_precomp = None\n')
assert old in s, 'scales/rotations block not found'
s = s.replace(old, new, 1)

# after the if/else that sets shs/colors_precomp/override, set scales+rotations
# when not computing cov3D in python (matches reference behavior)
old2 = '    else:\n        colors_precomp = override_color\n    rendered_image, radii = rasterizer(\n'
new2 = ('    else:\n        colors_precomp = override_color\n'
        '    if cov3D_precomp is None:\n'
        '        scales = pc.get_scaling\n'
        '        rotations = pc.get_rotation\n'
        '    rendered_image, radii = rasterizer(\n')
assert old2 in s, 'override block not found'
s = s.replace(old2, new2, 1)
open(p, 'w', encoding='utf-8').write(s)
import ast; ast.parse(s)
print('fixed scales/rotations; syntax OK')
