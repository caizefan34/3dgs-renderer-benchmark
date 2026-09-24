import sys, importlib.util, torch
torch.cuda.init()

def try_mod(path, name):
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        sys.modules[name] = m
        return True, None
    except Exception as e:
        return False, repr(e)[:200]

CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
EXP = "/tmp/higs_scalar_adjoint_build_20260920_r2/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
print("torch", torch.__version__, torch.version.cuda)
ok, err = try_mod(CORE, "gsplat_csrc")
print("base load:", ok, err)
sys.path.insert(0, "/tmp/higs_h3_fwd_1a_source")
if ok:
    try:
        import gsplat
        from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
        print("gsplat base wrapper OK")
    except Exception as e:
        print("gsplat base wrapper FAIL:", repr(e)[:200])
ok2, err2 = try_mod(EXP, "gsplat_scene_csrc")
print("exp load:", ok2, err2)
try:
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native
    print("experimental python OK")
except Exception as e:
    print("experimental python FAIL:", repr(e)[:200])