import sys, numpy as np, torch
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\ig0")
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\ig0\remote")
import ig0_fisher as NF
import ig0_fisher_torch as TF

rng = np.random.default_rng(7)
N = 500
q = rng.standard_normal((N,4)); q = q/np.linalg.norm(q,axis=1,keepdims=True)
ls = rng.uniform(-4,1,size=(N,3))
g_mu = rng.standard_normal((N,3)); g_ls = rng.standard_normal((N,3))
g_om = rng.standard_normal((N,3))
g_q = rng.standard_normal((N,4))
dom_test = rng.standard_normal((N,3))*0.4

# natural direction: numpy per-gaussian vs torch batch
dn_mu=[];dn_ls=[];dn_om=[]
for i in range(N):
    Sigma,_,lam = NF.sigma_from_quat_logscale(q[i], ls[i])
    Sinv=np.linalg.inv(Sigma)
    a,b,c = NF.natural_direction(g_mu[i], g_ls[i], g_om[i], Sinv, lam)
    dn_mu.append(a); dn_ls.append(b); dn_om.append(c)
dn_mu=np.array(dn_mu);dn_ls=np.array(dn_ls);dn_om=np.array(dn_om)
t_mu,t_ls,t_om = TF.natural_direction_batch(torch.tensor(g_mu),torch.tensor(g_ls),torch.tensor(g_om),torch.tensor(q),torch.tensor(ls))
t_mu=t_mu.numpy();t_ls=t_ls.numpy();t_om=t_om.numpy()
print("natural dmu max abs diff", np.abs(dn_mu-t_mu).max())
print("natural dls max abs diff", np.abs(dn_ls-t_ls).max())
print("natural dom max abs diff", np.abs(dn_om-t_om).max())

# radius
def np_radius(dmu,dls,dom):
    out=np.empty(N)
    for i in range(N):
        Sigma,_,lam=NF.sigma_from_quat_logscale(q[i],ls[i]); Sinv=np.linalg.inv(Sigma)
        out[i]=NF.fisher_block_norm2_full(Sinv,lam,dmu[i],dls[i],dom[i])
    return out
rn = np_radius(dn_mu,dn_ls,dn_om)
# torch radius needs Sigma_inv, lam, G_rot
Sigma_inv_t, lam_t, _ = TF.sigma_inv_batch(torch.tensor(q), torch.tensor(ls))
G_rot_t = TF.G_rot_batch(lam_t)
rt = TF.radius_geom_batch(torch.tensor(dn_mu),torch.tensor(dn_ls),torch.tensor(dn_om),Sigma_inv_t,lam_t,G_rot_t).numpy()
print("radius max abs diff", np.abs(rn-rt).max(), "rel", np.abs(rn-rt).max()/np.abs(rn).max())

# quat_relative_log
q1 = np.array([NF.normalize_q(NF.quat_mul(q[i], NF.exp_quat(dom_test[i]))) for i in range(N)])
n_rl = np.array([NF.quat_relative_log(q[i], q1[i]) for i in range(N)])
t_rl = TF.quat_relative_log_batch(torch.tensor(q), torch.tensor(q1)).numpy()
print("quat_relative_log max abs diff", np.abs(n_rl-t_rl).max())

# quat_grad_to_tangent
n_gt = np.array([NF.quat_grad_to_tangent(g_q[i], q[i]) for i in range(N)])
t_gt = TF.quat_grad_to_tangent_batch(torch.tensor(g_q), torch.tensor(q)).numpy()
print("quat_grad_to_tangent max abs diff", np.abs(n_gt-t_gt).max())

# apply_geometry roundtrip consistency
mu=rng.standard_normal((N,3))
mu2_n=[];ls2_n=[];q2_n=[]
for i in range(N):
    a,b,c=NF.apply_geometry(mu[i],ls[i],q[i],dn_mu[i],dn_ls[i],dn_om[i])
    mu2_n.append(a);ls2_n.append(b);q2_n.append(c)
mu2_t,ls2_t,q2_t = TF.apply_geometry_batch(torch.tensor(mu),torch.tensor(ls),torch.tensor(q),torch.tensor(dn_mu),torch.tensor(dn_ls),torch.tensor(dn_om))
print("apply mu diff", np.abs(np.array(mu2_n)-mu2_t.numpy()).max())
print("apply ls diff", np.abs(np.array(ls2_n)-ls2_t.numpy()).max())
print("apply q diff", np.abs(np.array(q2_n)-q2_t.numpy()).max())
print("VALIDATE_DONE")
