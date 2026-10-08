import sys
src = open("holdout8_trainer.py").read()
orig = src
src = src.replace('choices=["b1a", "b1", "b0", "c0", "a0", "a1", "a2", "gmain"])',
    'choices=["b1a", "b1", "b0", "c0", "c0fgs", "a0", "a1", "a2", "gmain"])')
src = src.replace('(0.3 if args.arm in ("c0", "a0", "a1", "a2", "gmain") else None))',
    '(0.3 if args.arm in ("c0", "c0fgs", "a0", "a1", "a2", "gmain") else None))')
src = src.replace(
    "    from gaussian_model import GaussianModel\n    from config import ReferenceV1Config\n    from colmap_reader import read_points3D_binary, sfm_to_pcd_data\n    from dataset import GTDataset\n",
    "    from gaussian_model import GaussianModel\n    from config import ReferenceV1Config\n    from colmap_reader import read_points3D_binary, sfm_to_pcd_data\n    from dataset import GTDataset\n"
    "    if args.arm == \"c0fgs\":\n"
    "        from rf0b_fastgs_model import (FastGSGaussianModel, compute_fastgs_scores,\n"
    "                                        sample_score_cameras, FGS_LOSS_THRESH,\n"
    "                                        FGS_N_SCORE_VIEWS)\n"
    "        ModelCls = FastGSGaussianModel\n"
    "    else:\n"
    "        ModelCls = GaussianModel\n")
src = src.replace("    model = GaussianModel(max_sh_degree=config.sh_degree)\n",
    "    model = ModelCls(max_sh_degree=config.sh_degree)\n")
assert src != orig
open("rf0b_trainer.py","w").write(src)
print("wrote rf0b_trainer.py", len(src), "bytes")
for tag in ["c0fgs","ModelCls","rf0b_fastgs_model","FGS_N_SCORE_VIEWS"]:
    print(tag, "count", src.count(tag))