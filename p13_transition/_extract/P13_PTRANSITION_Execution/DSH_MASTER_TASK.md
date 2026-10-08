# DSH master task — P13-X mainline + P-TRANSITION side probe

Do not ask for further scientific-design clarification. Preserve raw failures and contamination.

## Priority
1. P13-X X1/X2 room instrumentation pilot (mainline).
2. P-TRANSITION new-seed 5K/14K precheck.
3. Inventory exact 8K/11K snapshots. If they exist with verifiable provenance, run them. If not, stop the transition study at TWO_POINT_PRECHECK.
4. Do not start full 30K training from either branch.

## A. P13-X room pilot
Use the included `p13x/` files.
Follow `P13X_X12_INTEGRATION.md` exactly.
Do not modify `/root/epic05_p12/publication_trainer_p12.py`; copy it to `/root/epic05_p13x/`.
C0/fused hashes must match preregistration.
Pilot diagnostics: room at 15000, 15250, 15500 on 8 deterministic true COLMAP anchors.
Diagnostics are outside timing and RNG state must be restored exactly.
Run `p13x_analyze_x12.py`.
Pilot success only validates instrumentation. It is not X1/X2 scientific PASS.

Required room-pilot outputs:
- source identity JSON
- 24 snapshot NPZs (8 anchors × 3 iterations)
- room pilot analysis JSON
- paired short replay proof that diagnostics do not alter trajectory
- full log + GPU occupancy/contamination record

## B. P-TRANSITION
Base tree: Section-7 PRISM-GS v0.1.2. Do not modify old evidence.
Create `/root/epic05_ptransition/`, copy the v0.1.2 PRISM tree there, then place `run_transition.py` beside the `prism/` package.

New seeds ONLY:
271828 314159 161803

Primary fixed LR grids are embedded in the runner and preregistration:
screen = 0.01,0.03,0.1
geometry = 0.001,0.003,0.01

First run 5K and 14K for bicycle/garden, all 3 seeds.
Use the exact same snapshot and camera-sequence provenance rules as Section-7.

Example:
```bash
python run_transition.py \
  --gx-path /root/epic05_grc3d/remote_scripts \
  --gpu 2 --scene bicycle --iteration 5000 --seed 271828 \
  --snapshot /ABSOLUTE/VERIFIED/snap_bicycle_005000.pt \
  --camera-seq /ABSOLUTE/VERIFIED/p12a_gmainf_s42_bicycle/camera_sequence.npy \
  --out results/transition_bicycle_005000_s271828.json
```

Do not guess snapshot paths. Resolve them from existing manifests and save sha256s.

After the 12 two-point units:
```bash
python analyze_transition.py \
  --inputs results \
  --out P_TRANSITION_AGG.json
```

## C. 8K / 11K inventory
Search existing frozen manifests/checkpoints for exact bicycle/garden iteration 8000 and 11000 states.
Admissible only if provenance identifies the exact training recipe/source and the snapshot includes the same raw model parameter semantics needed by Section-7.

If unavailable:
- do not interpolate
- do not resume from 5K without the original optimizer state
- do not create substitute states
- report `INTERMEDIATE_SNAPSHOTS_UNAVAILABLE`
- final transition status remains a two-point precheck

If available, run the same 3 new seeds and fixed LR grids, then rerun the aggregator.

## D. Contamination / order
Use only one clean physical GPU cohort for a compared batch.
Record GPU UUID and foreign processes before/after each unit.
A contaminated unit is retained verbatim under a `.contaminated_*` name and rerun unchanged.
Candidate order is deterministic-shuffled by runner; do not reorder manually.

## E. Stop rules
P13-X:
- no production X1/X2 multi-scene campaign until room instrumentation/trajectory non-contamination passes.

P-TRANSITION:
- no full training.
- `TWO_POINT_STAGE_CROSSOVER_SIGNAL` only permits collecting real 8K/11K if available.
- `MECHANISM_SIGNAL_STAGE_DEPENDENT` only permits designing the next equal-wall-time trajectory experiment.
- any result remains non-comparable to frozen original-training Adam until original optimizer-state control is implemented.

## F. Return package
Return:
- P13-X room pilot raw data + analysis
- P-TRANSITION every raw JSON, not only best LRs
- P_TRANSITION_AGG.json
- snapshot inventory and hashes
- logs / contamination records
- source hashes and patches
- one report with OBSERVED / EVIDENCE / HYPOTHESIS / TEST / CONCLUSION
- no git commit
