p = 'clockgs/src/core.py'
s = open(p, encoding='utf-8').read()
old = 'gaussians.add_densification_stats(viewpoint_cam, render_pkg["viewspace_points"], visibility_filter)'
new = 'gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)'
assert old in s, 'call not found'
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(s)
import ast; ast.parse(s); print('fixed add_densification_stats call; syntax OK')
