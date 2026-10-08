import torch

from spatial_moment_predictor import spatial_nodes,allocate_nodes


def test_quadrature_preserves_arbitrary_opacity_mean_and_conic_gradients():
    torch.manual_seed(37)
    n=31
    x=torch.randn(n,2,2,dtype=torch.double)
    q=x@x.transpose(-1,-2)+.2*torch.eye(2)
    conics=torch.stack((q[:,0,0],q[:,0,1],q[:,1,1]),-1)
    opacity=torch.rand(n,dtype=torch.double)+.01
    gm=torch.randn(n,2,dtype=torch.double)
    gq=torch.randn(n,3,dtype=torch.double)
    go=torch.randn(n,dtype=torch.double)
    gc=torch.randn(n,3,dtype=torch.double)
    nodes,valid=spatial_nodes(conics,opacity,gm,gq,go,torch.ones(n,dtype=torch.bool))
    assert valid.all()
    for index,target in ((1,gm),(2,gq),(3,go)):
        torch.testing.assert_close(sum(node[index] for node in nodes),target)
    xy=torch.rand(n,2,dtype=torch.double)*1920
    labels=(torch.arange(135)%4).reshape(9,15)
    allocated=allocate_nodes(xy,nodes,gc,valid,labels,1080,1920)
    for index,target in enumerate((gm,gq,go,gc)):
        torch.testing.assert_close(sum(group[index] for group in allocated),target)


def test_zero_total_mean_gradient_retains_signed_spatial_lobes():
    q=torch.tensor([[1.,0.,1.]],dtype=torch.double)
    # Constant effective alpha adjoint: total mean derivative is zero,
    # yet left/right halves have opposite nonzero derivatives.
    nodes,_=spatial_nodes(q,torch.ones(1,dtype=torch.double),torch.zeros(1,2,dtype=torch.double),
                          torch.tensor([[-.5,0.,-.5]],dtype=torch.double),torch.ones(1,dtype=torch.double),
                          torch.ones(1,dtype=torch.bool))
    left=sum(node[1][0,0] for node in nodes if node[0][0,0]<0)
    right=sum(node[1][0,0] for node in nodes if node[0][0,0]>0)
    assert left<0 and right>0
    torch.testing.assert_close(left+right,torch.zeros_like(left))
