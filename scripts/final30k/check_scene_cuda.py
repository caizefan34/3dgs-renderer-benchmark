import sys, hashlib
import torch  # extension links against torch libs; must be loaded first
sys.path.insert(0, '/mnt/storage_pool/liaoyuanjun/torchext/gsplat_scene_cuda')
import gsplat_scene_cuda as s
h = hashlib.sha256(open(s.__file__, 'rb').read()).hexdigest()
print('IMPORT_OK', s.__file__)
print('SHA256', h)
print('has pack:', hasattr(s, 'pack_gaussian_inference_scene'))
