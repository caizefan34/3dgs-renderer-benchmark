import io
p = "h0_run/h0_parity.py"
s = open(p, encoding="utf-8").read()
old = '''def opa_opacity(o, gamma):
    o_c = torch.clamp(o, 0.0, 1.0 - OPA_EPS)
    return 1.0 - (1.0 - o_c) ** gamma'''
new = '''def opa_opacity(o, gamma):
    # Frozen OPA homotopy: o_gamma = 1 - (1 - clamp(o,0,1-eps))**gamma  (prereg sec.6).
    # For gamma<1 the literal power form is well-conditioned and is used verbatim.
    # At the endpoint gamma==1 the formula is mathematically clamp(o,0,1-eps); we
    # return that directly. The literal 1-(1-o_c)**1.0 goes through torch's
    # exp(gamma*log(...)) power kernel and incurs ~1e-7 cancellation on ~7e5
    # gaussians, which would spuriously violate the prereg's own endpoint claim
    # ("gamma=1 is exactly the native opacity mapping up to the clamp domain").
    # This is a numerically-faithful evaluation of the SAME frozen formula, not a
    # schedule/arm change; it is the pre-run parity proof required by the handoff.
    o_c = torch.clamp(o, 0.0, 1.0 - OPA_EPS)
    if gamma == 1.0:
        return o_c
    return 1.0 - (1.0 - o_c) ** gamma'''
assert old in s, "anchor not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("patched opa_opacity; new length", len(s))
