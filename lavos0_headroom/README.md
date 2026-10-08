# LAVOS0 — Lineage-Aware Exact VOS P0 Harness

Fail-closed experiment harness for the frozen LAVOS0 preregistration
(`LAVOS0_PREREG.md`, `LAVOS0_STATE_MACHINE.md`). It measures whether exact VOS
can be extended into the `0..15000` dynamic-topology phase, and only a
`LAVOS0_HEADROOM_PROMOTE` / `LAVOS0_STRONG_HEADROOM_PROMOTE` verdict authorizes a
later full LAVOS1 implementation.

This directory is an INDEPENDENT output root. It does **not** modify VOS1 or
ReproTopo0 (both frozen), and it fabricates no scientific verdict: the GPU
measurement phases require the EPIC-05 A100 and are provided as drivers + runbook.

## What is implemented here

| Layer | Module | Status |
|-------|--------|--------|
| Config / constants | `lavos0/config.py` | frozen from prereg + source lock |
| Canonical hashing / JSON IO | `lavos0/hashing.py` | done, tested |
| Dense A_s reference + settlement primitive | `lavos0/adam_ref.py` | done, bitwise-tested |
| Phase A event ordering | `lavos0/event_order.py` | encoded from frozen C0 source |
| Phase B consumer/debt trace | `lavos0/consumer_trace.py`, `lavos0/debt_replay.py` | done, tested |
| Phase C structural headroom gates | `lavos0/headroom_gates.py` | done, tested |
| Phase D exact topology replay oracle | `lavos0/topology_oracle.py` | done, bitwise-tested |
| Phase E perf / Amdahl oracle | `lavos0/perf_oracle.py` | done, tested |
| Source-identity freeze/verify | `lavos0/source_identity.py` | done, hashes 7 frozen files |
| Summary assembly | `lavos0/summary.py` | done |
| Mechanical verdict fold | `lavos0/verdict.py` | mirrors frozen analyzer |
| Frozen analyzer | `lavos0_analyze.py` | byte-identical to packet (SHA256 verified) |
| Offline test suite | `tests/` | 42 tests, all passing |
| Synthetic end-to-end pipeline | `drivers/lavos0_orchestrate_synthetic.py` | runs, manifest OK |
| A100 scientific driver | `drivers/lavos0_run_a100.py` | fails closed without CUDA/A100 |
| MANIFEST builder/verifier | `drivers/lavos0_make_manifest.py` | sha256sum -c compatible |

## Offline verification (no GPU)

```bash
python tests/run_all.py                          # 42 deterministic tests
python drivers/lavos0_orchestrate_synthetic.py --outdir a100_out/synthetic
python drivers/lavos0_make_manifest.py a100_out/synthetic --verify
```

The synthetic run exercises the FULL artifact pipeline (Phase A->E, summary,
frozen analyzer, MANIFEST) and writes `synthetic=true` on every artifact. It is
a plumbing proof, NOT a scientific result.

## Frozen event ordering (Phase A, observed from source)

Per iteration `t` in the frozen C0 trainer (`publication_trainer.py`):

```
update_learning_rate -> [oneup_sh_degree] -> render+loss -> backward
  -> [update_max_radii2D, add_densification_stats]           (densify region)
  -> [densify_and_prune]   == clone -> split -> prune          (topology)
  -> [reset_opacity]                                            (opacity reset)
  -> optimizer.step()   (dense A_s; NO-OP on topology iterations: params just
                         replaced with grad=None, state["step"] does not advance)
  -> zero_grad
```

Topology and opacity reset execute BEFORE the optimizer step. LAVOS0 mirrors
this: a topology event is observed, then the (possibly no-op) step runs.

## Running the real scientific experiment (EPIC-05 A100)

```bash
python drivers/lavos0_run_a100.py --all --outdir /root/lavos0_headroom
```

The driver executes the prereg order and stops fail-closed on:
`LAVOS0_SOURCE_IDENTITY_FAIL`, `LAVOS0_EVENT_SEMANTICS_UNRESOLVED`,
`LAVOS0_NO_STRUCTURAL_HEADROOM`, `LAVOS0_TOPOLOGY_EXACTNESS_FAIL`,
`LAVOS0_PERFORMANCE_HEADROOM_FAIL`.

Phase B (the real 0..15K dense A_s trace per scene, seed 42, corrected holdout
`i % 8 == 0` excluded) runs through the frozen `vos/scripts/vos_train_driver.py`
visibility wrapper. It records, per iteration: nonzero SH-grad rows, clone/split
parent ids, prune ids, fence rows, and topology/optimizer/iteration walls. The
harness consumes exactly that schema (`consumer_trace.py`). The Phase D oracle
then replays the captured events through clone/split/prune/compaction and
requires SHA256 equality of SH theta / exp_avg / exp_avg_sq / global step /
Gaussian count — no tolerance rescue.

## No-rescue rules (enforced structurally)

SH DC + SH rest only; A_s = `torch.optim.Adam(foreach=False)`; seed 42; scenes
room/garden/bicycle; 0..15000 phase; no opacity/xyz/scale/quat virtualization;
no threshold tuning; no Adam epsilon approximation; no local GPU for science.
See `LAVOS0_PREREG.md` Sec.11.

## Outputs

`LAVOS0_{ENVIRONMENT,SOURCE_IDENTITY,EVENT_ORDER,CONSUMER_TRACE,HEADROOM,
TOPOLOGY_ORACLE,PERF_ORACLE,SUMMARY,VERDICT}.json`, `LAVOS0_REPORT.md`,
`MANIFEST.sha256`.
