#!/usr/bin/env python3
"""Fresh-process allocator accounting for B2 F4 and Macro-F4."""
import argparse, importlib.util, json, runpy, sys
from pathlib import Path
import torch
from h3_fwd_0_oracle import f2_b2

def load(source, so):
    spec=importlib.util.spec_from_file_location("gsplat_cuda",so); core=importlib.util.module_from_spec(spec); spec.loader.exec_module(core); sys.modules["gsplat.csrc"]=core
    return runpy.run_path(str(Path(source)/"gsplat/experimental/render/kernels/cuda/build.py"))["build_and_load_experimental_gaussian_render_inference_scene"]()
def size(t): return t.numel()*t.element_size()
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--core-so",required=True);p.add_argument("--out",required=True);p.add_argument("--scene",default="room");a=p.parse_args(); x=load(a.source,a.core_so);from gsplat.cuda._wrapper import isect_offset_encode;f=f2_b2(a.scene,0,2048,"cuda")
 m,c,d,o,r=(torch.from_numpy(f[k]).cuda() for k in ("m2d","conics","depth","opacity","radii")); bm,br,bd,bc=(q[None,None].contiguous() for q in (m,r,d,c));bo=o[None,None].contiguous()
 torch.cuda.empty_cache(); torch.cuda.synchronize(); before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
 try: _,isect,flat=torch.ops.gsplat.intersect_tile(bm,br,bd,bc,bo,None,None,1,16,f["tw"],f["th"],True,False,None)
 except RuntimeError: _,isect,flat=torch.ops.gsplat.intersect_tile(bm,br,bd,bc,bo,None,None,1,16,f["tw"],f["th"],True,False)
 off=isect_offset_encode(isect,1,f["tw"],f["th"]);torch.cuda.synchronize(); b2persist=size(isect)+size(flat)+size(off);b2peak=torch.cuda.max_memory_allocated()-before
 del isect,flat,off;torch.cuda.empty_cache();torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
 y=x.higs_train_macro_f4(m,c,d,o,r,f["tw"],f["th"],16,True,False);torch.cuda.synchronize(); mpersist=sum(size(q) for q in y[:4]);mpeak=torch.cuda.max_memory_allocated()-before
 entries=int(y[0][-1]);nm=y[0].numel()-1;result={a.scene: {"B2_F4":{"isect_metadata_bytes":None,"flatten_ids_bytes":None,"tile_offsets_bytes":None,"persistent_bytes":b2persist,"peak_allocator_delta_bytes":b2peak,"temporary_bytes_estimate":max(0,b2peak-b2persist),"radix_sort_workspace_bytes":max(0,b2peak-b2persist)},"Macro_F4":{"macro_offsets_bytes":size(y[0]),"macro_sorted_ids_bytes":size(y[1]),"batch_offsets_bytes":size(y[2]),"retained_mask_bytes":size(y[3]),"persistent_bytes":mpersist,"scan_workspace_bytes":3*nm*4,"segmented_sort_workspace_bytes":4*entries*4,"temporary_bytes_estimate":3*nm*4+4*entries*4,"peak_bytes_estimate":mpersist+3*nm*4+4*entries*4,"peak_allocator_delta_bytes":mpeak}}}; result[a.scene]["B2_F4"]["isect_metadata_bytes"]=len(f["flat"])*8;result[a.scene]["B2_F4"]["flatten_ids_bytes"]=len(f["flat"])*4;result[a.scene]["B2_F4"]["tile_offsets_bytes"]=f["tw"]*f["th"]*4;Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
