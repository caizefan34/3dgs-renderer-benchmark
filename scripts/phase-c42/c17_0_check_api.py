import gsplat
funcs = [x for x in dir(gsplat) if not x.startswith('_')]
print('All exports:', funcs)
print()
intersect_funcs = [x for x in funcs if 'intersect' in x.lower() or 'offset' in x.lower() or 'tile' in x.lower()]
print('Intersect/offset/tile:', intersect_funcs)
