#!/usr/bin/env python3
"""H2 gate self-test: synthetic scenarios injected into the real room dumps.

Runs on the build host (needs torch + the real dumps). Each scenario copies the
frozen dumps, injects one defect, runs p10_h2_equivalence.cmd_compare, and
asserts the classification.

  P   unmodified dumps                       -> PASS_NUMERICALLY_EQUIVALENT
  E   on arm == off arm (switch inert)       -> INVALID_SWITCH_NOT_ENGAGED
  S1  on moments scaled by 1+1e-2            -> FAIL_SEMANTICS  (gross algebra error)
  S2  on moments scaled by 1+3e-4            -> FAIL_SEMANTICS  (moderate)
  S3  on moments scaled by 1+1e-4            -> FAIL_SEMANTICS  (sensitivity floor)
  S4  on moments scaled by 1+2e-6            -> PASS             (below the
                                                representation's own fp32 noise;
                                                documented resolution limit)
  C1  harness reconstruct with B/2           -> FAIL_SEMANTICS  (would catch a
                                                dropped / halved cross term)
  C2  harness reconstruct with conics w/o 0.5-> FAIL_SEMANTICS  (would catch a
                                                dropped 1/2 on the diagonal)
  C3  harness reconstruct with B sign flipped-> FAIL_SEMANTICS

C1-C3 monkeypatch the reconstruction convention, not the data: they demonstrate
that the reconstruction itself is *identified* by the data, i.e. that the gate
has no freedom to choose a convention that happens to fit.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p10h2", os.path.join(HERE, "p10_h2_equivalence.py"))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

DUMP_PRI = "dumps/h2_pri_room.pt"
DUMP_H8 = "dumps/h2_h8_room.pt"
ON_RUNS = ["on0", "on1", "on2"]


def load_dumps():
    return torch.load(DUMP_PRI, weights_only=False), torch.load(DUMP_H8, weights_only=False)


def run_compare(pri_path, h8_path, tmpdir, tag):
    out = os.path.join(tmpdir, f"eq_{tag}.json")
    H.cmd_compare(type("A", (), {"scene": "room", "pristine": pri_path, "h8": h8_path, "out": out})())
    with open(out) as f:
        return json.load(f)


def materialize(h8, tmpdir, tag):
    q = os.path.join(tmpdir, f"h8_{tag}.pt")
    torch.save(h8, q)
    return q


def scale_moments(h8, factor):
    for r in ON_RUNS:
        for k in ("means2d_raw", "conics_raw"):
            h8["dump"][r][k] = h8["dump"][r][k] * factor


def id_arm(h8):
    for r in ON_RUNS:
        h8["dump"][r] = {k: (v.clone() if torch.is_tensor(v) else v)
                         for k, v in h8["dump"]["off0"].items()}


ORIG_RECON = H.reconstruct


def recon_b_halved(means2d_raw, conics, conics_raw):
    """[[A, B/2],[B/2, C]] - the cross term missing a factor."""
    t = conics.double()
    A, B, C = t[..., 0], t[..., 1] / 2.0, t[..., 2]
    m = means2d_raw.double()
    mt = torch.stack([A * m[..., 0] + B * m[..., 1], B * m[..., 0] + C * m[..., 1]], dim=-1)
    cc = conics_raw.double()
    return mt, torch.stack([0.5 * cc[..., 0], cc[..., 1], 0.5 * cc[..., 2]], dim=-1)


def recon_diag_no_half(means2d_raw, conics, conics_raw):
    """diagonal terms missing their 1/2 (conics: (Sxx, Sxy, Syy) instead of halves)."""
    return ORIG_RECON(means2d_raw, conics, conics_raw)[0], torch.stack(
        [conics_raw.double()[..., 0], conics_raw.double()[..., 1], conics_raw.double()[..., 2]], dim=-1)


def recon_b_sign(means2d_raw, conics, conics_raw):
    t = conics.double()
    A, B, C = t[..., 0], -t[..., 1], t[..., 2]
    m = means2d_raw.double()
    mt = torch.stack([A * m[..., 0] + B * m[..., 1], B * m[..., 0] + C * m[..., 1]], dim=-1)
    cc = conics_raw.double()
    return mt, torch.stack([0.5 * cc[..., 0], cc[..., 1], 0.5 * cc[..., 2]], dim=-1)


def scale_key(key, factor):
    def f(h8):
        for r in ON_RUNS:
            h8["dump"][r][key] = h8["dump"][r][key] * factor
    return f


SCENARIOS = [
    ("P  unmodified", None, None, "PASS_NUMERICALLY_EQUIVALENT"),
    ("E  on==off", None, id_arm, "INVALID_SWITCH_NOT_ENGAGED"),
    ("S1 moments x(1+1e-2)", 1.0 + 1e-2, None, "FAIL_SEMANTICS"),
    ("S2 moments x(1+3e-4)", 1.0 + 3e-4, None, "FAIL_SEMANTICS"),
    ("S3 moments x(1+1e-4)", 1.0 + 1e-4, None, "FAIL_SEMANTICS"),
    # measured boundary-gate resolution is ~1e-6 relative on the moment
    # buffers (a 2e-6 perturbation is caught), so this one must pass
    ("S4 moments x(1+2e-7)", 1.0 + 2e-7, None, "PASS_NUMERICALLY_EQUIVALENT"),
    ("P1 means3d x(1+1e-3)", None, scale_key("means3d", 1.0 + 1e-3), "FAIL_SEMANTICS"),
    ("P2 means3d x(1+1e-4)", None, scale_key("means3d", 1.0 + 1e-4), "FAIL_SEMANTICS"),
    ("P3 means3d x(1+1e-5)", None, scale_key("means3d", 1.0 + 1e-5), "PASS_NUMERICALLY_EQUIVALENT"),
    ("P4 sh x(1+1e-4)", None, scale_key("sh", 1.0 + 1e-4), "FAIL_SEMANTICS"),
    ("C1 recon B/2", None, None, "FAIL_SEMANTICS", recon_b_halved),
    ("C2 recon diag no-1/2", None, None, "FAIL_SEMANTICS", recon_diag_no_half),
    ("C3 recon B sign", None, None, "FAIL_SEMANTICS", recon_b_sign),
]


def main():
    tmpdir = tempfile.mkdtemp(prefix="h2self_")
    print(f"fixture: {DUMP_PRI} / {DUMP_H8}  tmp={tmpdir}")
    bad = 0
    try:
        for sc in SCENARIOS:
            name, scale, mutate, expect = sc[0], sc[1], sc[2], sc[3]
            H.reconstruct = sc[4] if len(sc) > 4 else ORIG_RECON
            tag = name.split()[0]
            if scale is None and mutate is None:
                h8_path = DUMP_H8
            else:
                h8 = torch.load(DUMP_H8, weights_only=False)
                if scale is not None:
                    scale_moments(h8, scale)
                if mutate is not None:
                    mutate(h8)
                h8_path = materialize(h8, tmpdir, tag)
            res = run_compare(DUMP_PRI, h8_path, tmpdir, tag)
            got = res["verdict"]["classification"]
            ok = got == expect
            bad += not ok
            pg = res["verdict"]["pipeline_gate"]
            extra = "  pipeline(bias_l2/tail): " + ",".join(
                f"{k}={pg[k]['bias_l2_ratio']:.2f}/{pg[k]['tail_ratio']:.2f}" for k in pg)
            if "boundary_calibration" in res["verdict"]:
                bc = res["verdict"]["boundary_calibration"]
                extra += "  boundary(bias_l2/tail): " + ",".join(
                    f"{k}={bc[k]['bias_l2_ratio']:.2f}/{bc[k]['tail_ratio']:.2f}" for k in bc)
            print(f"[{'ok ' if ok else 'BAD'}] {name:22s} expect={expect:30s} got={got}{extra}")
    finally:
        H.reconstruct = ORIG_RECON
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("SELFTEST", "FAILED" if bad else "PASSED", f"({len(SCENARIOS) - bad}/{len(SCENARIOS)})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
