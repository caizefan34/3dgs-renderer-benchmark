import importlib.util as u, sys, json, math, time
import torch
torch.manual_seed(42)
DEV="cuda"; DEG=3; ACTIVE_SH_BASES=(DEG+1)**2; NREST=(DEG+1)**2-1

def load_HC():
    spec=u.spec_from_file_location("_C", r"C:\rb\FasterGSFusedCudaBackend\FasterGSFusedCudaBackend\_C.cp313-win_amd64.pyd")
    m=u.module_from_spec(spec); spec.loader.exec_module(m); return m
def load_MC():
    sys.path.insert(0, r"C:\rb\M\diff_gaussian_rasterization_faster")
    import diff_gaussian_rasterization_faster as M; return M
HC=load_HC()

def make_scene(N):
    means=(torch.rand(N,3,device=DEV)*2-1)*2.0
    scale_lin=torch.rand(N,3,device=DEV)*0.06+0.02
    q=torch.randn(N,4,device=DEV); q=q/q.norm(dim=1,keepdim=True)
    op=torch.rand(N,1,device=DEV)*0.4+0.4
    dc=torch.randn(N,1,3,device=DEV)*0.15
    sh=torch.randn(N,NREST,3,device=DEV)*0.03
    return dict(means=means,scale_lin=scale_lin,rot=q,op=op,dc=dc,sh=sh,N=N)

def cam(W,H,view=0):
    focal=0.5*W/math.tan(0.5*math.radians(60.0))
    ang=view*math.radians(30.0)
    R=torch.zeros(3,3,device=DEV); R[0,0]=math.cos(ang); R[0,2]=math.sin(ang); R[1,1]=1.0; R[2,0]=-math.sin(ang); R[2,2]=math.cos(ang)
    campos=R@torch.tensor([0.,0.,4.0],device=DEV)
    w2c=torch.eye(4,device=DEV); w2c[:3,:3]=R.t(); w2c[:3,3]=-R.t()@campos
    w2c[2,2]=-1.0  # H: +z forward convention handled by depth=4-z; keep simple front view
    # recompute properly: H wants depth = -z_view (positive). Use view->world then invert.
    Rinv=R.t()
    w2c=torch.eye(4,device=DEV); w2c[:3,:3]=Rinv; w2c[:3,3]=-Rinv@campos
    # flip z so depth positive in front
    Fz=torch.diag(torch.tensor([1.,1.,-1.,1.],device=DEV))
    w2c=Fz@w2c
    viewmatrix=torch.eye(4,device=DEV); viewmatrix[:3,:3]=Rinv; viewmatrix[:3,3]=-Rinv@campos
    proj=torch.zeros(4,4,device=DEV); proj[0,0]=2*focal/W; proj[1,1]=2*focal/H
    proj[2,2]=(100.01)/(99.99); proj[2,3]=-2*100*0.01/99.99; proj[3,2]=-1.0; proj=proj@viewmatrix
    return dict(focal=focal,cx=W/2.0,cy=H/2.0,campos=campos,bg=torch.zeros(3,device=DEV),
                w2c=w2c,viewmatrix=viewmatrix,proj=proj,tanfovx=(0.5*W)/focal,tanfovy=(0.5*H)/focal)

class Ev:
    def __init__(s): s.s=torch.cuda.Event(enable_timing=True); s.e=torch.cuda.Event(enable_timing=True)
    def __enter__(s): torch.cuda.synchronize(); s.s.record(); return s
    def __exit__(s,*a): s.e.record(); torch.cuda.synchronize(); return False
    def ms(s): return s.s.elapsed_time(s.e)

def h_settings(C,W,H):
    return (C["w2c"],C["campos"],C["bg"],ACTIVE_SH_BASES,W,H,C["focal"],C["focal"],C["cx"],C["cy"],0.01,100.0,0.0,0)

def h_fwd(sc,C,W,H,scales,opacity):
    st=h_settings(C,W,H)
    return HC.forward(sc["means"],scales,sc["rot"],opacity,sc["dc"],sc["sh"],*st)
def h_bwd(out,sc,C,W,H,scales,opacity):
    N=sc["N"]; st=h_settings(C,W,H); g=torch.ones_like(out[0])
    HC.backward(torch.zeros(2*N,device=DEV),sc["means"],scales,sc["rot"],opacity,sc["dc"],sc["sh"],
        torch.zeros(2*N*3,device=DEV),torch.zeros(2*N*3,device=DEV),torch.zeros(2*N*4,device=DEV),
        torch.zeros(2*N,device=DEV),torch.zeros(2*N*3,device=DEV),torch.zeros(2*N*NREST*3,device=DEV),
        g,out[0],out[1],out[2],out[3],out[4],*st,out[5],out[6],out[7])

_MB=[None]
def m_fwd(sc,C,W,H):
    N=sc["N"]; MC=load_MC()
    if _MB[0] is None: _MB[0]=MC.Buffer(device=DEV,MAX_NUM_RENDERED=1<<21,MAX_NUM_TILES=1<<16)
    buf=_MB[0]
    culling=torch.zeros(N,dtype=torch.bool,device=DEV)
    buf.curr_offset.fill_(0)
    radii=MC._C.ops.preprocess(DEG,sc["means"],torch.empty(0,device=DEV),sc["dc"],sc["sh"],sc["op"],sc["scale_lin"],sc["rot"],1.0,W,H,16,16,C["campos"],C["viewmatrix"],C["proj"],C["tanfovx"],C["tanfovy"],100.0,0.01,buf.gaussian_keys_unsorted,buf.gaussian_values_unsorted,buf.gaussian_values_sorted,buf.curr_offset,1.0,buf.splat_buffer,culling)
    num=int(buf.curr_offset.cpu()[0])
    MC._C.ops.sort_gaussian(num,W,H,16,16,buf.list_sorting_space,buf.gaussian_keys_unsorted,buf.gaussian_values_unsorted,buf.gaussian_keys_sorted,buf.gaussian_values_sorted)
    last=torch.zeros(H,W,1,dtype=torch.int32,device=DEV); Ts=torch.zeros(H,W,1,device=DEV)
    bs,out_color,img_buf,smp_buf=MC._C.ops.render_16x16(N,num,W,H,buf.splat_buffer,buf.gaussian_keys_sorted,buf.gaussian_values_sorted,buf.ranges,C["bg"],last,Ts)
    return out_color,num,bs,radii,last,Ts,img_buf,smp_buf,buf
def m_bwd(ctx,sc,C,W,H):
    out_color,num,bs,radii,last,Ts,img_buf,smp_buf,buf=ctx
    N=sc["N"]
    g=torch.ones_like(out_color)
    accum=torch.zeros(N,3,device=DEV);accum_abs=torch.zeros(N,3,device=DEV);maxr=torch.zeros(N,device=DEV);denom=torch.zeros(N,3,device=DEV)
    MC=load_MC()
    MC._C.ops.backward(DEG,bs,sc["means"],radii,sc["dc"],sc["sh"],sc["op"],sc["scale_lin"],sc["rot"],1.0,W,H,16,16,C["tanfovx"],C["tanfovy"],100.0,0.01,C["campos"],C["viewmatrix"],C["proj"],img_buf,smp_buf,buf.splat_buffer,buf.gaussian_values_sorted,buf.ranges,C["bg"],last,Ts,g,accum,accum_abs,maxr,denom)

if __name__=="__main__":
    backend=sys.argv[1]; N=int(sys.argv[2]); Wp=int(sys.argv[3]); Hp=int(sys.argv[4]); iters=int(sys.argv[5]); view=int(sys.argv[6]) if len(sys.argv)>6 else 0
    sc=make_scene(N); C=cam(Wp,Hp,view)
    for _ in range(5):
        if backend=="H": h_fwd(sc,C,Wp,Hp,torch.log(sc["scale_lin"]),torch.log(sc["op"]/(1-sc["op"])))
        else: m_fwd(sc,C,Wp,Hp)
    torch.cuda.synchronize()
    fw=[]; bw=[]; inst=0; buck=0; oc=None; img=None
    for _ in range(iters):
        if backend=="H":
            m=sc["means"].clone(); r=sc["rot"].clone(); d=sc["dc"].clone(); sh=sc["sh"].clone()
            s=torch.log(sc["scale_lin"]).clone(); o=torch.log(sc["op"]/(1-sc["op"])).clone()
            scc=dict(means=m,rot=r,dc=d,sh=sh,N=sc["N"])
            with Ev() as ev: out=h_fwd(scc,C,Wp,Hp,s,o)
            fw.append(ev.ms()); img=out[0]; inst=out[5]; buck=out[6]
            with Ev() as ev: h_bwd(out,scc,C,Wp,Hp,s,o)
            bw.append(ev.ms())
        else:
            with Ev() as ev: ctx=m_fwd(sc,C,Wp,Hp)
            fw.append(ev.ms()); oc,inst,buck=ctx[0],ctx[1],ctx[2]
            with Ev() as ev: m_bwd(ctx,sc,C,Wp,Hp)
            bw.append(ev.ms())
    fw.sort(); bw.sort()
    im=(img if backend=="H" else oc).detach().cpu().float()
    res=dict(backend=backend,N=N,W=Wp,H=Hp,iters=iters,
             fwd_ms=fw[len(fw)//2], bwd_ms=bw[len(bw)//2],
             fwd_ms_min=fw[0], bwd_ms_min=bw[0],
             instances=int(inst), buckets=int(buck), img_mean=float(im.mean()), img_std=float(im.std()))
    print("JSON_START"+json.dumps(res)+"JSON_END")
