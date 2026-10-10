import numpy as np
from PIL import Image
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_r1_deliverables/gp0_renders"
arr=np.load(f"{D}/5K_B_cam0.npy")
gt=np.asarray(Image.open('/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_images/_DSC8679.JPG').convert('RGB').resize((396,264),Image.LANCZOS),np.float32)/255.0
print('arr',arr.shape,arr.dtype,'gt',gt.shape,gt.dtype)
mse=np.square(arr.astype(np.float64)-gt.astype(np.float64)).mean()
mae=np.abs(arr.astype(np.float64)-gt.astype(np.float64)).mean()
print('mse',mse,'mae',mae,'psnr',-10*np.log10(max(mse,1e-12)))
# also check the S1A-R1 saved render if present
import os
p=f"{D}/0005000_B.npy"
if os.path.exists(p):
    a2=np.load(p)
    print('s1ar1 5K B shape',a2.shape)
    if a2.shape==arr.shape:
        print('max diff s1ar1 vs gp0',np.abs(a2-arr).max())
