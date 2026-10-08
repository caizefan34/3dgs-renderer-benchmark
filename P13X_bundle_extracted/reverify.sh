#!/bin/bash
# unit-test the fixed _dct_reconstruct, then re-run ONLY the diag-on preflight leg
cd /root/epic05_p13x
cat > /tmp/dct_test.py <<'EOF'
import sys, torch
sys.path.insert(0, "/root/epic05_p13x")
import p13x_diag
torch.manual_seed(7)
x = torch.randn(1080, 1920, 3, device="cuda")
r256 = p13x_diag._dct_reconstruct(x, 256)
e256 = float((r256 - x).norm() / x.norm())
r4 = p13x_diag._dct_reconstruct(x, 4)
e4 = float((r4 - x).norm() / x.norm())
r32 = p13x_diag._dct_reconstruct(x, 32)
e32 = float((r32 - x).norm() / x.norm())
print(f"K256 rel={e256:.3e}  K32 rel={e32:.4f}  K4 rel={e4:.4f}")
assert e256 < 1e-5, "K=256 must reproduce input exactly"
assert e4 > e32 > e256, "K ordering must be monotone"
# odd-size edge case
y = torch.randn(1000, 777, 3, device="cuda")
ry = p13x_diag._dct_reconstruct(y, 256)
ey = float((ry - y).norm() / y.norm())
print(f"odd-size K256 rel={ey:.3e}")
assert ey < 1e-5
print("DCT_TEST_OK")
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/dct_test.py || exit 1
rm -rf preflight/B_diagon_300 preflight/lock artifacts/P13X_PREFLIGHT.json
export P13X_PREFLIGHT_GPU=2
/root/miniforge3/envs/gsplat/bin/python p13x_preflight.py > logs/preflight2.log 2>&1
echo "PREFLIGHT_EXIT=$?"
tail -40 logs/preflight2.log
