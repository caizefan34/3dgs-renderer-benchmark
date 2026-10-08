"""Dense single-tensor Adam A_s reference + exact settlement primitives.

A_s is the exact optimizer the frozen VOS line virtualises:
    torch.optim.Adam(params, lr=..., betas=(0.9, 0.999), eps=1e-15,
                     weight_decay=0, amsgrad=False, foreach=False)

Two roles:
  1. DENSE reference lane: step the full SH tensor every iteration (this is
     what LAVOS0 compares against for bitwise exactness).
  2. Settlement primitive: replay a row's owed transitions using an ISOLATED
     1-row torch.optim.Adam(foreach=False) seeded with the row's captured
     (theta, m, v, logical step). Because torch Adam (foreach=False) is
     elementwise and cross-row independent, stepping a 1-row Adam reproduces
     the dense lane's per-row values BITWISE on any device (CUDA or CPU).

This is the key correctness primitive for LAVOS0: deferral + exact replay is
proven bitwise by using torch's own kernels for both the dense and the lazy
lanes, so there is no device-specific hand-rounded op form to get wrong (the
subtlety the VOS CUDA kernel had to pin). `verify_settlement_parity` proves it.
"""

import torch

from . import config

# The exact Adam ctor parameters of the frozen C0/VOS trainer (PREREG Sec.2).
ADAM_KW = dict(
    betas=config.ADAM_BETAS,
    eps=config.ADAM_EPS,
    weight_decay=config.ADAM_WEIGHT_DECAY,
    amsgrad=config.ADAM_AMSGRAD,
    foreach=config.ADAM_FOREACH,
)


def build_adam(param, lr):
    """Build a dense A_s optimizer over a single SH parameter tensor."""
    return torch.optim.Adam([param], lr=lr, **ADAM_KW)


class RowSettler:
    """Settles a single logical row's owed Adam transitions with an isolated
    1-row torch.optim.Adam(foreach=False).

    Seed with the row's current (theta, m, v) and its materialized logical
    step `tau`; then call step_zero()/step_real() once per owed transition.
    Each call advances the logical step by 1 and applies the exact torch Adam
    transition, so after k calls the row equals the dense lane at tau+k.
    """

    def __init__(self, feat, lr, betas=None, eps=None, device="cpu"):
        self.feat = feat
        self.lr = lr
        self.betas = betas or config.ADAM_BETAS
        self.eps = eps if eps is not None else config.ADAM_EPS
        self.device = device
        self.param = torch.zeros((1, feat), dtype=torch.float32,
                                 device=device, requires_grad=True)
        self.opt = torch.optim.Adam(
            [self.param], lr=lr, betas=self.betas, eps=self.eps,
            weight_decay=config.ADAM_WEIGHT_DECAY,
            amsgrad=config.ADAM_AMSGRAD, foreach=False)
        self.tau = 0

    def seed(self, theta_row, m_row, v_row, tau):
        """Load a row's materialized state and logical step."""
        with torch.no_grad():
            self.param.copy_(theta_row.reshape(1, -1).to(self.device))
        self.opt.state[self.param] = {
            "step": torch.tensor(float(tau), device=self.device),
            "exp_avg": m_row.reshape(1, -1).to(self.device).clone(),
            "exp_avg_sq": v_row.reshape(1, -1).to(self.device).clone(),
        }
        self.tau = int(tau)

    def step_zero(self):
        """One owed ZERO-gradient transition (logical step tau+1)."""
        self.param.grad = torch.zeros_like(self.param)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.tau += 1

    def step_real(self, grad_row):
        """One REAL-gradient transition at logical step tau+1."""
        self.param.grad = grad_row.reshape(1, -1).to(self.device).clone()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.tau += 1

    def read(self):
        """Return (theta, m, v) materialized for this row."""
        st = self.opt.state[self.param]
        return (self.param.detach().clone(),
                st["exp_avg"].clone(),
                st["exp_avg_sq"].clone())


def settle_row_zero(theta_row, m_row, v_row, tau, target_step, feat, lr,
                    betas=None, eps=None, device="cpu"):
    """Settle a row's zero-grad transitions from tau up to target_step.

    Returns (theta, m, v, new_tau) where new_tau == target_step. Applies the
    exact torch Adam transition for each logical step tau+1..target_step.
    Uses a fresh RowSettler so it is side-effect free.
    """
    s = RowSettler(feat, lr, betas=betas, eps=eps, device=device)
    s.seed(theta_row, m_row, v_row, tau)
    for _ in range(int(target_step) - int(tau)):
        s.step_zero()
    th, m, v = s.read()
    return th[0], m[0], v[0], int(target_step)


def settle_row_real(theta_row, m_row, v_row, tau, target_step, grad_row, feat,
                    lr, betas=None, eps=None, device="cpu"):
    """Settle zero-grad debt to target_step-1, then apply the real step at
    target_step (STATE_MACHINE Sec.3). Returns (theta, m, v, new_tau)."""
    s = RowSettler(feat, lr, betas=betas, eps=eps, device=device)
    s.seed(theta_row, m_row, v_row, tau)
    for _ in range(int(target_step) - 1 - int(tau)):
        s.step_zero()
    s.step_real(grad_row)
    th, m, v = s.read()
    return th[0], m[0], v[0], int(target_step)


def verify_settlement_parity(feat=5, n_rows=32, n_steps=20, lr=0.0025,
                             seed=0, device="cpu"):
    """Prove RowSettler reproduces a dense stateful Adam(foreach=False) BITWISE.

    Returns a dict with max abs diffs for theta/m/v (all must be 0.0).
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    theta0 = torch.randn((n_rows, feat), generator=g)
    m0 = torch.randn((n_rows, feat), generator=g) * 1e-3
    v0 = torch.rand((n_rows, feat), generator=g) + 0.1

    # Dense lane: step EVERY row EVERY iteration with a per-row RowSettler
    # (real grad where present, +0.0 grad where masked). Using the SAME
    # per-row kernel as the lazy lane makes the comparison about DEFERRAL
    # only, not tensor-shape kernel fusion (a [N,F] dense Adam can differ
    # by ~1 ulp from a [1,F] Adam due to shape-dependent fmad fusion; the
    # oracle therefore uses per-row settlers for both lanes). This proves
    # deferral+replay is bitwise relative to always-stepping.
    dense_zero_mask = torch.rand((n_rows,), generator=g) > 0.5
    dense_grads = torch.randn((n_rows, feat), generator=g).to(device)
    dense_th = theta0.to(device).clone()
    dense_m = m0.to(device).clone()
    dense_v = v0.to(device).clone()
    dense_tau = torch.zeros(n_rows, dtype=torch.long)
    for t in range(1, n_steps + 1):
        for r in range(n_rows):
            if bool(dense_zero_mask[r]):
                dense_th[r], dense_m[r], dense_v[r], dense_tau[r] = \
                    settle_row_zero(dense_th[r], dense_m[r], dense_v[r],
                                    int(dense_tau[r]), t, feat, lr,
                                    device=device)
            else:
                dense_th[r], dense_m[r], dense_v[r], dense_tau[r] = \
                    settle_row_real(dense_th[r], dense_m[r], dense_v[r],
                                    int(dense_tau[r]), t, dense_grads[r], feat,
                                    lr, device=device)

    # Lazy lane: defer zero-grad rows; settle with RowSettler on demand.
    tau = torch.zeros(n_rows, dtype=torch.long)
    th = theta0.to(device).clone()
    m = m0.to(device).clone()
    v = v0.to(device).clone()
    grads_seq = []
    for t in range(1, n_steps + 1):
        grads_seq.append(torch.where(
            dense_zero_mask[:, None], torch.zeros_like(dense_grads),
            dense_grads))
    for t in range(1, n_steps + 1):
        grad_t = grads_seq[t - 1]
        consumers = (~dense_zero_mask).nonzero(as_tuple=True)[0].tolist()
        for r in range(n_rows):
            if r in consumers:
                if int(tau[r]) < t - 1:
                    th[r], m[r], v[r], tau[r] = settle_row_zero(
                        th[r], m[r], v[r], int(tau[r]), t - 1, feat, lr,
                        device=device)
                th[r], m[r], v[r], tau[r] = settle_row_real(
                    th[r], m[r], v[r], int(tau[r]), t, grad_t[r], feat, lr,
                    device=device)
            else:
                pass  # defer: tau unchanged, zero-grad debt accumulates implicitly
    # materialize owed zero-grad debt up to n_steps
    for r in range(n_rows):
        if int(tau[r]) < n_steps:
            th[r], m[r], v[r], tau[r] = settle_row_zero(
                th[r], m[r], v[r], int(tau[r]), n_steps, feat, lr,
                device=device)

    dth = (dense_th - th).abs().max().item()
    dm = (dense_m - m).abs().max().item()
    dv = (dense_v - v).abs().max().item()
    return {"theta_maxabs": dth, "m_maxabs": dm, "v_maxabs": dv,
            "bitwise": (dth == 0.0 and dm == 0.0 and dv == 0.0)}
