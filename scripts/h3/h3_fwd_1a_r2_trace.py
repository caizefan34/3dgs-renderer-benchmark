#!/usr/bin/env python3
"""Capture the two pre-repair bicycle membership discrepancies."""
import argparse, importlib.util, json, math, runpy, struct, sys
from pathlib import Path
import numpy as np
import torch
from h3_fwd_0_oracle import f2_b2
from h3_fwd_1a_r_validate import cuda_lists, tile_lists

def load(source, core_so):
    spec=importlib.util.spec_from_file_location("gsplat_cuda",core_so);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);sys.modules["gsplat.csrc"]=core
    return runpy.run_path(str(Path(source)/"gsplat/experimental/render/kernels/cuda/build.py"))["build_and_load_experimental_gaussian_render_inference_scene"]()
def bits(x):
    value=float(np.float32(x)); return {"value":value,"hex":"0x%08x"%struct.unpack("<I",struct.pack("<f",value))[0]}
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--core-so",required=True);p.add_argument("--out",required=True);a=p.parse_args();x=load(a.source,a.core_so);f=f2_b2("bicycle",0,2048,"cuda")
    ins=[torch.from_numpy(f[k]).cuda() for k in ("m2d","conics","depth","opacity","radii")];o=x.higs_train_macro_f4(*ins,f["tw"],f["th"],16,True,False);torch.cuda.synchronize(); got,offs,ids,batches,masks=cuda_lists(o[:4],f["tw"],f["th"]);ref=tile_lists(f["flat"],f["offs"],f["tw"]*f["th"]);mw=math.ceil(f["tw"]/8); rows=[]
    for tile,(r,g) in enumerate(zip(ref,got)):
      for gaussian,present_b2,present_macro in [(q,True,False) for q in set(r)-set(g)]+[(q,False,True) for q in set(g)-set(r)]:
       tx,ty=tile%f["tw"],tile//f["tw"];mx,my=tx//8,ty//4;macro=my*mw+mx;bit=(ty%4)*8+(tx%8);entry=None
       for i in range(int(offs[macro]),int(offs[macro+1])):
         if int(ids[i])==gaussian: entry=i;break
       rows.append({"fine_tile_id":tile,"fine_tile_xy":[tx,ty],"macro_id":macro,"macro_xy":[mx,my],"visible_gaussian_id":gaussian,"master_gaussian_id":None,"B2_contains_pair":present_b2,"current_macro_contains_pair":present_macro,"current_32bit_mask_bit":None if entry is None else bool(int(np.uint32(masks[entry]))&(1<<bit)),"means2d":{"x":bits(f["m2d"][gaussian,0]),"y":bits(f["m2d"][gaussian,1])},"conic":{"A":bits(f["conics"][gaussian,0]),"B":bits(f["conics"][gaussian,1]),"C":bits(f["conics"][gaussian,2])},"opacity":bits(f["opacity"][gaussian]),"depth":bits(f["depth"][gaussian]),"radii":[int(f["radii"][gaussian,0]),int(f["radii"][gaussian,1])]})
    Path(a.out).write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=="__main__":main()
