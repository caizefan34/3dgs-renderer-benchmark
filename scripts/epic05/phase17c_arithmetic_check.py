import torch, json, math

d = torch.load('results/phase-a100/c17_2_membership_collected.pt', map_location='cpu', weights_only=True)

for k, v in d['scenes'].items():
    nmaster = v['n_gaussians']
    nisect = v['n_isects']
    gm = v['gaussian_tile_membership']
    n_visible = gm['count']
    mean_tpg = gm['mean']
    
    out = []
    out.append("=== %s ===" % k)
    out.append("  master_gaussians = %d" % nmaster)
    out.append("  n_isects (total memberships) = %d" % nisect)
    out.append("  n_visible (Gaussians with >=1 tile) = %d" % n_visible)
    out.append("  n_invisible = %d" % (nmaster - n_visible))
    out.append("  reported mean tiles/G (over visible) = %.4f" % mean_tpg)
    out.append("  recomputed: n_isects / n_visible = %d / %d = %.4f" % (nisect, n_visible, nisect / n_visible))
    out.append("  n_visible * mean_tpg = %d * %.4f = %.1f" % (n_visible, mean_tpg, n_visible * mean_tpg))
    out.append("  DIFF = %.2f" % (nisect - n_visible * mean_tpg))
    out.append("  mean tiles/G (over ALL) = %.4f" % (nisect / nmaster))
    
    ntiles = v['n_tiles']
    w, h = v['width'], v['height']
    tw = math.ceil(w / 16); th = math.ceil(h / 16)
    out.append("  resolution=%dx%d tile_grid=%dx%d ntiles=%d computed=%d" % (w, h, tw, th, ntiles, tw*th))
    out.append("")
    
    # Also compute bincount mean manually from raw data to verify
    out.append("  RAW MEAN: n_isects / n_visible = %.6f" % (nisect / n_visible))
    out.append("")

    print('\n'.join(out))

# Camera native resolutions
with open('data/official/mipnerf360/room/cameras.json') as f:
    cams = json.load(f)
c = cams[0]
print("room native: %dx%d" % (c['width'], c['height']))

with open('data/official/mipnerf360/bicycle/cameras.json') as f:
    cams = json.load(f)
c = cams[0]
print("bicycle native: %dx%d" % (c['width'], c['height']))

with open('data/official/mipnerf360/garden/cameras.json') as f:
    cams = json.load(f)
c = cams[0]
print("garden native: %dx%d" % (c['width'], c['height']))
