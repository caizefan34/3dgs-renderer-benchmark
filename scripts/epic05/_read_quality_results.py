import json

with open('results/epic05/official/quality/quality_room_1080p_20260819_183516.json') as f:
    data = json.load(f)

print('=== Room Quality Results (tile16 vs tile32) ===')
t16 = data['tile_quality']['tile16']
t32 = data['tile_quality']['tile32']

psnr16 = t16['mean_psnr_db']
ssim16 = t16['mean_ssim']
lpips16 = t16['mean_lpips']
psnr32 = t32['mean_psnr_db']
ssim32 = t32['mean_ssim']
lpips32 = t32['mean_lpips']

print(f'Tile16: PSNR={psnr16:.3f} dB, SSIM={ssim16:.6f}, LPIPS={lpips16:.6f}')
print(f'Tile32: PSNR={psnr32:.3f} dB, SSIM={ssim32:.6f}, LPIPS={lpips32:.6f}')

gate = data['quality_gate']['tile16_vs_tile32']
print(f'Delta PSNR: {gate["delta_psnr_db"]} dB')
print(f'Delta SSIM: {gate["delta_ssim"]}')
print(f'Delta LPIPS: {gate["delta_lpips"]}')
print(f'Status: {gate["gate_status"]}')

eq = data['cross_tile_equivalence']['tile16_vs_tile32']['aggregate']
print()
print('Pixel Equivalence:')
mx = eq['max_max_abs_pixel_error']
mn = eq['mean_mean_abs_pixel_error']
rmse = eq['mean_rmse']
identical = eq['exactly_identical_all_views']
print(f'  Max pixel error: {mx}')
print(f'  Mean pixel error: {mn}')
print(f'  Mean RMSE: {rmse}')
print(f'  Exactly identical all views: {identical}')

# Performance
print()
print('Performance:')
for ts_label in ['tile16', 'tile32']:
    td = data['tile_quality'].get(ts_label, {})
    if 'num_views' in td:
        # timing from std output, not in JSON
        pass
print(f'  Tile16: {data["timing_tile16"] if "timing_tile16" in data else "N/A"}')
print(f'  Tile32: {data["timing_tile32"] if "timing_tile32" in data else "N/A"}')

# Show all keys in the result
print()
print(f'All top-level keys: {list(data.keys())}')
for k in data.keys():
    v = data[k]
    if isinstance(v, dict):
        print(f'  {k}: dict with keys {list(v.keys())[:5]}')
    elif isinstance(v, list):
        print(f'  {k}: list of {len(v)} items')
    else:
        print(f'  {k}: {v}')
