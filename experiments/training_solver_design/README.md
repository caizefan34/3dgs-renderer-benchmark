# Training solver design experiments

This directory is an isolated research artifact. It does not change the frozen
P12 renderer, trainer, public baselines, or benchmark leaderboards.

Current strong-method literature and a bounded spatial group-gradient screen:
[RECENT_TRAINING_FRONTIER_AUDIT.md](RECENT_TRAINING_FRONTIER_AUDIT.md).
ACE-GS, Vol3DGS, 3DGART and GRay cover several previously plausible ideas;
simple shared voxel motion lacked a broad multi-view gradient signal on room.
The pinned Vol3DGS source and a stable, but unbenchmarked, center-ray
calculation are screened in [VOLUMETRIC_FACTOR_SCREEN.md](VOLUMETRIC_FACTOR_SCREEN.md).

Latest bounded prior-art screen: [DEPTH_ORDER_PRIOR_ART_SCREEN.md](DEPTH_ORDER_PRIOR_ART_SCREEN.md).
Per-ray depth/local sorting is already covered by StopThePop, and tile-local
depth binning/repair by TileGS. The narrower color/transmittance-aware conflict
idea received a bounded real-room pixel screen: direct per-ray reordering
worsened sampled GT L1 on average and is not advancing to training CUDA.
See the screen document for coverage and limitations.

Recent representation hypothesis: [TRANSMITTANCE_SPLIT_DESIGN.md](TRANSMITTANCE_SPLIT_DESIGN.md).
The isolated CUDA pair compositor passed independent-child VJP checks and
three full-resolution room birth audits, including all 260,530 parents paired.
Read [SURVIVAL_PAIR_RESULTS.md](SURVIVAL_PAIR_RESULTS.md) for native gate
counterexamples, the preserved initial full-image failure, and its stable
compositing correction. Motion coverage/order, training speed, final quality,
production cost and novelty remain unverified. The prior
full-resolution prune-repair direction is recorded in
[FULL_RESOLUTION_PARETO_DIRECTION.md](FULL_RESOLUTION_PARETO_DIRECTION.md), but
its deletion-scoring opportunity gate did not generalize across diagnostic
views; read [PRUNING_DELETION_RESULTS.md](PRUNING_DELETION_RESULTS.md).

The follow-on [SURVIVAL_MOTION_RESULTS.md](SURVIVAL_MOTION_RESULTS.md) tested
real 3D movement, union tile coverage, parent-group versus physical child
ordering, and A/B view loss. The fixed-parent-depth pair configuration did not
earn promotion to full training. Run
`python experiments/training_solver_design/analyze_survival_motion.py` to verify the
executed source identities and bounded summary. The broader objective remains
faster full-resolution training with higher official unseen-view quality.

Recheck the executed sources and bounded result summary with
`python experiments/training_solver_design/analyze_survival_pair.py`.

Historical full-resolution audit: [FULL_RESOLUTION_PARETO_DIRECTION.md](FULL_RESOLUTION_PARETO_DIRECTION.md).
The user requires faster training and higher unseen-view quality. The earlier
half-resolution training plus full-resolution topology scouts is retained only
as a historical tradeoff hypothesis; it is not promoted as the current answer.
The existing 132-job full-resolution fused/prune/radius-clip arm is faster and
passes non-inferiority gates, but mean LPIPS worsens, so a quality improvement
has not been established. Per-scene raw results and exact environment/data
identity must be recovered before attributing that regression or new runs.

Historical proposal: [RESOLUTION_TOPOLOGY_DESIGN.md](RESOLUTION_TOPOLOGY_DESIGN.md).
The audited 30K `higs_proposed` arm uses half-resolution until step 24K and
visible Adam; default densification ends at 15K. The earlier tile-sampling
explanation in CAPACITY_COVERAGE_DESIGN.md was a protocol mismatch and has
been corrected. Fixed P11 room/garden/bicycle cross-resolution probes find candidate
ranking changes, but neither training quality nor a new method is verified.
Recheck `training_frontier_audit.json` with `analyze_training_frontier.py`.

Latest evidence: [SPATIAL_PREDICTOR_RESULTS.md](SPATIAL_PREDICTOR_RESULTS.md).
Free coarse sorting still did not beat native. A spatial-moment predictor has a
fresh-reference noise-reduction signal, but old caches fail geometry noise checks;
screen-space field transport also has an unexplained numerical mean discrepancy.
See [SPATIAL_PREDICTOR_DESIGN.md](SPATIAL_PREDICTOR_DESIGN.md). No complete-training
or novelty claim. Run `python experiments/training_solver_design/analyze_spatial_predictor.py`
to verify six executed source identities and regenerate the summary.

Earlier evidence: [MULTILEVEL_RESULTS.md](MULTILEVEL_RESULTS.md). Half-resolution
control variates reduce sampled-gradient noise on all six fixed views, but the
two-render prototypes remain slower on garden even after shared projection/SH,
selected-core loss, and cheaper conic scaling. The room SH boundary discrepancy
was fixed in a versioned helper without changing the gate or frozen sources.
No end-to-end training or SOTA advantage is established. Read
[MULTILEVEL_DESIGN.md](MULTILEVEL_DESIGN.md) for the exact estimator and prior art.
Run `python experiments/training_solver_design/analyze_multilevel.py` to verify
executed source identities and regenerate the diagnostic summary.

Earlier evidence: [TRANSIENT_RESULTS.md](TRANSIENT_RESULTS.md) and
[SIBLING_RESULTS.md](SIBLING_RESULTS.md): 52 permanent-coordinate and 40
transient-coordinate GPU arms did not establish a stable final advantage.
Raw optimizer-state transport under an explicit zero-extended prehistory was
implemented and independently tested; this is not a complete-training speedup.
[TRANSIENT_COORDINATES_DESIGN.md](TRANSIENT_COORDINATES_DESIGN.md) preserves the
design and its limitations. Current configurations are not promoted to production.

Earlier [NEWBORN_RESULTS.md](NEWBORN_RESULTS.md) covers 34 known newborn-age and
moment-inheritance controls, with no stable standalone optimization.

Read [REAL_TOPOLOGY_RESULTS.md](REAL_TOPOLOGY_RESULTS.md) first: real room/garden
GPU probes did not establish a useful selection advantage for the current
conditional scores. [TOPOLOGY_DESIGN.md](TOPOLOGY_DESIGN.md) retains the finite-edit capacity
allocation proposal, prior-art distinctions, costs and gates. [DESIGN.md](DESIGN.md)
retains the previous solver proposal and negative results; its broad innovation
positioning is closed. Neither supports an end-to-end speedup or SOTA claim.

From the repository root, using the existing Python environment:

```powershell
python -m pytest experiments/training_solver_design/test_appearance_oracle.py -q
python experiments/training_solver_design/topology_oracle.py
python experiments/training_solver_design/analyze_real_topology.py
python experiments/training_solver_design/newborn_adam_audit.py
python -m pytest experiments/training_solver_design/test_newborn_updates.py -q
python experiments/training_solver_design/analyze_newborn.py
python -m pytest experiments/training_solver_design/test_sibling_coordinates.py experiments/training_solver_design/test_sibling_transport.py -q
python experiments/training_solver_design/analyze_sibling.py
python experiments/training_solver_design/analyze_transient.py
python experiments/training_solver_design/real_crop_probe.py --scene room --output experiments/training_solver_design/room_crop_evidence.json
python experiments/training_solver_design/real_crop_probe.py --scene bicycle --output experiments/training_solver_design/bicycle_crop_evidence.json
python experiments/training_solver_design/real_crop_probe.py --scene garden --output experiments/training_solver_design/garden_crop_evidence.json
python experiments/training_solver_design/subspace_probe.py
```

Dependencies already available locally: NumPy, SciPy, PyTorch, Pillow, plyfile,
threadpoolctl, pytest. The crop probe requires local official PLY/cameras,
extracted dataset images, and `public_baselines/fastgs/utils/sh_utils.py`.
It hashes checkpoint, camera, image and SH-source inputs. It does not install
dependencies, compile CUDA, download a dataset, or modify an upstream checkout.

The crop probe is a small independent renderer, not certified gsplat output.
Its camera split follows a conventional sorted every-eighth rule and is not
independently certified. Results are restricted to fixed pretrained geometry
and selected DC coefficients. Crop SSIM and PSNR must not enter a leaderboard.

`subspace_probe.py` uses synthetic 3DGS with a complete L1+valid-window SSIM
gradient. Curvature uses a uniformly sampled RGB-response Jacobian, optionally
with IRLS weights. Its SciPy bounded solve and CPU autograd Jacobian are
correctness oracles, not the planned GPU implementation. Validation cameras
are never used to choose candidate steps.

Historical development configurations are retained:

```powershell
python experiments/training_solver_design/subspace_probe.py --mode absolute_mse --constraint ray --output experiments/training_solver_design/reproduced_absolute_mse.json
python experiments/training_solver_design/subspace_probe.py --mode scaled_irls --constraint ray --output experiments/training_solver_design/reproduced_scaled_ray.json
```

Current scripts add stronger controls and diagnostics, so historical JSON
schemas differ; the old core candidate settings remain reproducible.

The literature artifact records versioned article URLs, retrieval times, HTML
hashes and relevant excerpts. It is a bounded prior-art check, not an exhaustive
novelty certification.
