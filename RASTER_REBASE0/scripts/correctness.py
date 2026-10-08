import importlib.util as u, sys, math, torch
torch.manual_seed(42); DEV="cuda"; DEG=3; ASB=16; NREST=15
sys.path.insert(0, r"C:\rb\M\diff_gaussian_rasterization_faster")
import diff_gaussian_rasterization_faster as MC
sys.path.insert(0, r"C:\rb\M\diff_gaussian_rasterization_faster")
import diff_gaussian_rasterization_faster as MC
def load_HC():
    spec=u.spec_from_file_location("_C", r"C:\rb\FasterGSFusedCudaBackend\FasterGSFusedCudaBackend\_C.cp313-win_amd64.pyd")
    m=u.module_from_spec(spec); spec.loader.exec_module(m); return m
HC=load_HC()
N=50000
means=(torch.rand(N,3,device=DEV)*2-1)*2.0
scale_lin=torch.rand(N,3,device=DEV)*0.06+0.02
q=torch.randn(N,4,device=DEV); q=q/q.norm(dim=1,keepdim=True)
op=torch.rand(N,1,device=DEV)*0.4+0.4
dc=torch.randn(N,1,3,device=DEV)*0.15; sh=torch.randn(N,NREST,3,device=DEV)*0.03
scales=torch.log(scale_lin); opacity=torch.log(op/(1-op))
W=H=256; focal=0.5*W/math.tan(0.5*math.radians(60.0))
w2c=torch.eye(4,device=DEV); w2c[2,2]=-1.0; w2c[2,3]=4.0
viewmatrix=torch.eye(4,device=DEV); viewmatrix[2,3]=-4.0
proj=torch.zeros(4,4,device=DEV); proj[0,0]=2*focal/W; proj[1,1]=2*focal/H; proj[2,2]=(100.01)/(99.99); proj[2,3]=-2*100*0.01/99.99; proj[3,2]=-1.0; proj=proj@viewmatrix
st=(w2c,torch.tensor([0.,0.,4.],device=DEV),torch.zeros(3,device=DEV),ASB,W,H,focal,focal,W/2.0,H/2.0,0.01,100.0,0.0,0)
out=HC.forward(means,scales,q,opacity,dc,sh,*st)
himg=out[0].detach().cpu().float()  # (3,H,W)
buf=MC.Buffer(device=DEV,MAX_NUM_RENDERED=1<<21,MAX_NUM_TILES=1<<16)
culling=torch.zeros(N,dtype=torch.bool,device=DEV)
buf.curr_offset.fill_(0)
radii=MC._C.ops.preprocess(DEG,means,torch.empty(0,device=DEV),dc,sh,op,scale_lin,q,1.0,W,H,16,16,torch.tensor([0.,0.,4.],device=DEV),viewmatrix,proj,(0.5*W)/focal,(0.5*H)/focal,100.0,0.01,buf.gaussian_keys_unsorted,buf.gaussian_values_unsorted,buf.gaussian_values_sorted,buf.curr_offset,1.0,buf.splat_buffer,culling)
num=int(buf.curr_offset.cpu()[0])
MC._C.ops.sort_gaussian(num,W,H,16,16,buf.list_sorting_space,buf.gaussian_keys_unsorted,buf.gaussian_values_unsorted,buf.gaussian_keys_sorted,buf.gaussian_values_sorted)
last=torch.zeros(H,W,1,dtype=torch.int32,device=DEV); Ts=torch.zeros(H,W,1,device=DEV)
bs,oc,ib,sb=MC._C.ops.render_16x16(N,num,W,H,buf.splat_buffer,buf.gaussian_keys_sorted,buf.gaussian_values_sorted,buf.ranges,torch.zeros(3,device=DEV),last,Ts)
mimg=oc.detach().cpu().float()  # (3,H,W)
d=(himg-mimg).abs()
print("H img mean %.4f std %.4f"%(himg.mean(),himg.std()))
print("M img mean %.4f std %.4f"%(mimg.mean(),mimg.std()))
print("abs diff mean %.5f max %.5f"%(d.mean(),d.max()))
print("rel diff (mean/max of H): %.5f %.5f"%(d.mean()/himg.mean(), d.max()/himg.max()))
print("PSNR-ish: %.2f dB"%(10*torch.log10(1.0/(d.mean()+1e-8))))
# gradient check: compare a few grad magnitudes
g=torch.ones_like(himg)
print("OK correctness computed")