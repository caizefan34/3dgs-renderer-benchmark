import torch
def runit():
    import gsplat
    from gsplat.cuda._wrapper import fully_fused_projection
    means = torch.zeros(3,3,device='cuda'); means[0,0]=1
    quats = torch.tensor([[1.,0,0,0]]*3,device='cuda')
    scales = torch.ones(3,3,device='cuda')*0.1
    opac = torch.ones(3,device='cuda')
    # trivial identity camera
    view = torch.eye(4,device='cuda')[None]
    K = torch.tensor([[500.,0,100],[0,500,100],[0,0,1]],device='cuda')[None]
    with torch.no_grad():
        r,m,d,c,_=fully_fused_projection(means.unsqueeze(0),None,quats.unsqueeze(0),scales.unsqueeze(0),view,K,200,200,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False,camera_model='pinhole')
    print("OK radii", r[0,0,:,0].tolist())
if __name__=="__main__":
    print("cuda", torch.cuda.is_available())
    try:
        runit()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("FAIL", repr(e)[:200])