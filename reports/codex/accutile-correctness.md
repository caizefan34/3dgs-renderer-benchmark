# AccuTile correctness

Status: **partial PASS; full workload gate not completed**.

The conservative predicate minimizes the upstream opacity-thresholded conic over
each candidate tile's continuous rectangle, with a positive boundary retention
margin.  Therefore it can only retain extra AABB tiles at boundaries; it cannot
discard a tile containing a rasterizer pixel center.  Invalid conics/opacity retain
the original AABB path.

The compiled `scripts/accutile_smoke.py` test used autograd on both paths:

| output | max abs |
| --- | ---: |
| RGB | 0 |
| alpha | 0 |
| mean grad | 4.5776367e-05 |
| quaternion grad | 6.1035156e-05 |
| scale grad | 1.0986328e-03 |
| opacity grad | 3.0517578e-05 |
| color grad | 3.8146973e-06 |

All gradient differences are normal floating-point accumulation-order effects;
NaN and Inf counts were zero.  The existing full Room checkpoint produced identical
RGB and alpha sums for OFF and ON.  Per-element Room gradients, depth mode, and
the bicycle/garden correctness runs remain unmeasured.
