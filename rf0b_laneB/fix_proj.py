p = "rf0b_fastgs_model.py"
s = open(p, encoding="utf-8").read()
old = '''    X = model.get_xyz                                  # [N,3] world
    ones = torch.ones_like(X[:, :1])
    Xh = torch.cat([X, ones], dim=1)                   # [N,4]
    wvt = cam.world_view_transform.to(X.device)        # [4,4] world->view
    Xv = (wvt @ Xh.T).T[:, :3]                         # [N,3] camera coords (z fwd)
    K = cam.K.to(X.device)                             # [3,3]
    Xh2 = torch.cat([Xv, ones], dim=1)                 # [N,4]
    P = (K @ Xh2.T).T                                  # [N,3]
    z = P[:, 2].clamp_min(1e-8)
    px = P[:, 0] / z
    py = P[:, 1] / z
    valid = (Xv[:, 2] > 0.01) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px, py, valid'''
new = '''    X = model.get_xyz                                  # [N,3] world
    ones = torch.ones_like(X[:, :1])
    Xh = torch.cat([X, ones], dim=1)                   # [N,4]
    wvt = cam.world_view_transform.to(X.device)        # [4,4] world->view
    Xv = (wvt @ Xh.T).T[:, :3]                         # [N,3] camera coords (z fwd)
    K = cam.K.to(X.device)                             # [3,3]
    P = (K @ Xv.T).T                                   # [N,3]  (K is 3x3: drop homog)
    z = P[:, 2].clamp_min(1e-8)
    px = P[:, 0] / z
    py = P[:, 1] / z
    valid = (Xv[:, 2] > 0.01) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px, py, valid'''
assert old in s, "project block not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("PROJECTION FIX APPLIED")