p = "cvt0_REPORT.md"  # placeholder, not used
# patch the report GENERATOR narrative
gp = "cvt0_report.py"
s = open(gp, encoding="utf-8").read()

old1 = '''The `G_hard` gate is mandatory: transfer-aware selection must materially beat
the highest-loss camera. It does not \u2014 `G_hard_median` exceeds the 0.20 promote
floor in only a few panels and is frequently ~0 or negative, and `T_oracle_median`
is at/below zero in most cells.'''
new1 = '''The mandatory `G_hard` gate is the one CVT0 exists to test (TurboGS already uses
an error-aware anchor). Here `G_hard_median` clears the 0.35 strong floor in every
cell, so the transfer oracle does beat the highest-current-loss source on the
normalized peer metric. The DROP is therefore NOT from `G_hard`; it comes from the
gates that require the advantage to be *consistent and specific*:
- `F_pos_median` (fraction of peers the oracle source actually improves) never
  reaches the 0.65 promote floor in any cell (0.29\u20130.57) \u2014 the oracle's
  advantage is diffuse, not a targeted improvement.
- `H_oracle_positive_count` (true-holdout panels where the oracle helps) is 0\u20131
  in 5/6 cells; only bicycle@200K reaches 3.
- `T_oracle_median > 0` fails for room@15000/20000; `G_true_median` is
  \u22640 for room@15000 and bicycle@15000/20000; `rho_TH_median` is negative in
  4/6 cells \u2014 the train-panel transfer signal does not generalize to held-out
  views.'''
assert old1 in s, "narrative 1 not found"
s = s.replace(old1, new1)

old2 = '''Cross-view transfer is dominated by geometry proximity and the highest-current-loss
camera; a peer-transfer oracle provides no consistent, material advantage over the
error-aware source that TurboGS already selects, and it does not generalize to the
true held-out views (`G_true_median` frequently \u2264 0, `H_oracle_positive_count` 0\u20131).
Per prereg \u00a79 and DSH_HANDOFF item 8, **{verd['verdict']} closes this mechanism**.
No rescue: panels/gates/holdout/checkpoints are unchanged.'''
new2 = '''The transfer oracle does beat the highest-current-loss source on the normalized
peer metric (`G_hard` gate passes everywhere), so cross-view transfer is *not*
purely the hard-view signal. But the advantage is not actionable: it is diffuse
(`F_pos` 0.29\u20130.57, far below 0.65), it rarely helps the true held-out views
(`H_oracle_positive_count` 0\u20131 in 5/6 cells), and the train-panel transfer signal
does not generalize to held-out views (`G_true_median` \u22640 in 3/6 cells,
`rho_TH_median` negative in 4/6). Geometry proximity is a weak correlate
(pose-vs-|M| Pearson \u2248 \u22120.16), so transfer is not merely viewpoint distance
either. Per prereg \u00a79 and DSH_HANDOFF item 8, **{verd['verdict']} closes this
mechanism**. No rescue: panels/gates/holdout/checkpoints are unchanged.'''
assert old2 in s, "narrative 2 not found"
s = s.replace(old2, new2)

open(gp, "w", encoding="utf-8", newline="\n").write(s)
print("patched narrative")
