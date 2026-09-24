c = open('scripts/phase-c29/f_camera_utility.py').read()
c = c.replace('psnr=_psnr(ra.clamp(0,1).permute(2,0,1).unsqueeze(0),gt)', 'psnr=_psnr(ra[0].clamp(0,1).permute(2,0,1).unsqueeze(0),gt)')
open('scripts/phase-c29/f_camera_utility.py','w').write(c)
print('fixed')
