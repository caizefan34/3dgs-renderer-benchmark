#!/usr/bin/env python3
"""Toy ONLY; no 3DGS or MX data. Check analytic Bernoulli HT maths."""
import math
import random
from variance_oracle import probabilities, analytic_variance, analyze, read_data


def test_probability_budget():
    c = [1.0, 2.0, 0.5, 4.0, 0.7]
    s = [4.0, 1.0, 0.5, 2.0, 1.0]
    B = 0.25 * sum(c)
    pp = probabilities(s, c, B, 0.05)
    assert all(0.05 <= p <= 1.0 for p in pp)
    assert math.isclose(sum(p * cc for p, cc in zip(pp, c)), B, rel_tol=1e-9)
    for invalid in [0.0, -1.0]:
        try:
            probabilities(s, c, invalid, 0.05)
        except ValueError:
            pass
        else:
            raise AssertionError("expected budget floor failure")


def test_oracle_optimal_vs_uniform():
    ids = [str(x) for x in range(6)]
    g = [3, 9, 1, 0.4, 2, 6]
    c = [4, 1, 3, 2, 4, 1]
    err = [1, 1, 1, 1, 1, 1]
    d = analyze(ids, g, c, err, 0.25, 0.05, "UNIT_TEST_NOT_MX")
    assert d["results"]["O_unavailable_true_gradient_oracle"]["analytic_variance"] <= d["results"]["U_uniform_Bernoulli"]["analytic_variance"] + 1e-7
    assert d["evidence_class"] == "DERIVED_OFFLINE_PROXY_COST_NOT_RUNTIME"


def test_ht_unbiased_and_variance():
    rng = random.Random(20261008)
    grads = [[2.0, -1.0], [1.0, 3.0], [-2.0, 0.5], [1.5, -0.5]]
    probs = [0.25, 0.6, 0.4, 0.8]
    ref = [sum(g[k] for g in grads) for k in range(2)]
    accum = [0.0, 0.0]
    err_sq = 0.0
    n = 90000
    for _ in range(n):
        est = [0.0, 0.0]
        for g, p in zip(grads, probs):
            if rng.random() < p:
                for k in range(2):
                    est[k] += g[k] / p
        for k in range(2):
            accum[k] += est[k]
        err_sq += sum((x-y)**2 for x, y in zip(est, ref))
    mean = [x/n for x in accum]
    assert all(abs(a-b) < 0.065 for a, b in zip(mean, ref)), (mean, ref)
    ana = analytic_variance([math.sqrt(sum(x*x for x in g)) for g in grads], probs)
    emp = err_sq / n
    assert abs(emp-ana)/ana < 0.04, (ana, emp)


def test_header_only_is_not_data():
    from pathlib import Path
    try:
        read_data(Path(__file__).parent.parent / "schema" / "tile_measurements_example.csv")
    except ValueError:
        return
    raise AssertionError("header-only CSV must be rejected")


if __name__ == "__main__":
    test_probability_budget()
    print("PASS probability budget and floor")
    test_oracle_optimal_vs_uniform()
    print("PASS analytic oracle <= uniform proxy variance")
    test_ht_unbiased_and_variance()
    print("PASS toy HT unbiasedness and analytic variance")
    test_header_only_is_not_data()
    print("PASS no placeholder data accepted")
    print("ALL 4 TOY MATH TESTS PASSED; REAL MX EXPERIMENTS NOT RUN")
