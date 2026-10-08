import torch

from transported_adjoint_predictor import capture_field,transported_nodes


def polynomial_integral(mu,conic,opacity,constant,linear,quadratic):
    q=torch.stack((conic[:,0],conic[:,1],conic[:,1],conic[:,2]),-1).reshape(-1,2,2)
    covariance=torch.linalg.inv(q)
    mass=2*torch.pi*opacity*torch.linalg.det(covariance).sqrt()
    mean=constant+(linear*mu).sum(-1)+torch.einsum('ni,nij,nj->n',mu,quadratic,mu)+(quadratic*covariance).sum((-1,-2))
    return (mass*mean).sum()


def test_transport_matches_frozen_quadratic_field_after_shape_and_position_changes():
    torch.manual_seed(85)
    n=7
    mu=torch.randn(n,2,dtype=torch.double,requires_grad=True)
    conic=torch.tensor([[2.,.3,1.]]*n,dtype=torch.double,requires_grad=True)
    opacity=torch.full((n,),.6,dtype=torch.double,requires_grad=True)
    constant=torch.randn(n,dtype=torch.double)
    linear=torch.randn(n,2,dtype=torch.double)
    z=torch.randn(n,2,2,dtype=torch.double)
    quadratic=(z+z.transpose(-1,-2))/2
    grads=torch.autograd.grad(polynomial_integral(mu,conic,opacity,constant,linear,quadratic),(mu,conic,opacity))
    field=capture_field(mu,conic,opacity,*grads,torch.zeros(n,3,dtype=torch.double),torch.ones(n,dtype=torch.bool))
    new_mu=(mu.detach()+.4*torch.randn_like(mu)).requires_grad_()
    new_q=torch.tensor([[1.2,-.2,1.7]]*n,dtype=torch.double,requires_grad=True)
    new_o=torch.full((n,),.4,dtype=torch.double,requires_grad=True)
    target=torch.autograd.grad(polynomial_integral(new_mu,new_q,new_o,constant,linear,quadratic),(new_mu,new_q,new_o))
    nodes,_,valid=transported_nodes(field,new_mu,new_q,new_o,torch.ones(n,dtype=torch.bool))
    assert valid.all()
    for idx,wanted in enumerate(target,1):
        torch.testing.assert_close(sum(node[idx] for node in nodes),wanted)
    # Retaining old gradient numbers cannot generally produce this transport.
    assert not torch.allclose(grads[0],target[0])
