# DSH TASK — P17-P0 Multirate Gradient Oracle

Read `P17_P0_PREREG.md` completely before execution.

## Role

DSH is an EPIC-05 execution harness only.

Do not:
- redesign P17,
- tune thresholds,
- add a new loss,
- implement a production multirate renderer,
- run full training,
- claim speedup,
- change the camera set after seeing results.

## Execution sequence

### 1. Identity

Verify the four frozen P12/C0 hashes in the preregistration.

Locate certified 16K C0 model+optimizer states for:
- room
- bicycle.

If multiple candidate checkpoints exist, choose the one whose lineage has the strongest existing identity evidence; record its exact sha256.

Do not mix parameters from one run with optimizer state from another.

### 2. Freeze cameras

Build `P17_CAMERA_SET.json` using the exact sorted-ID quantile algorithm.

Freeze it before gradient probes.

### 3. d=1 parity

For every selected probe camera:

- run an ordinary frozen C0 reference render/backward;
- run the P17 oracle code at d=1;
- compare:
  - rendered RGB
  - loss
  - raw parameter gradients
  - visible IDs / intersections.

Primary parity:
- RGB max abs <= 1e-6
- fused-loss rel error <= 1e-7
- every finite raw gradient tensor cosine >= 0.999999
- exact visible/intersection ID equality

Any systematic d=1 mismatch => `P17_EXPORT_INVALID`.

### 4. Coarse probes

For d=2,4,8:

- area-downsample GT deterministically;
- scale intrinsics;
- use eps2d=0.3/d^2;
- reset the dynamic scene before every independent render/backward;
- save per-Gaussian raw gradients;
- save conic/covariance footprint;
- save exact intersection counts.

### 5. Primary tau=2 assignment

Construct `d_i` exactly as preregistered.

Do not pick the best d per Gaussian using gradient fidelity.

The assignment is footprint-only.

### 6. Hybrid step

Use cloned 16K parameter+optimizer state.

For each camera:
- one dense full-res reference Adam step;
- one hybrid-gradient Adam step;
- exact full-res post-step evaluation on probe camera and all 4 audit cameras.

No densification event.

### 7. Aggregate Stage A

Run the supplied analyzer.

If hard drop:
- STOP.
- Do not run garden.

If HOLD:
- STOP.
- Do not run garden.

If Stage-A strong:
- prepare an exact C0 16K garden state if necessary;
- run identical garden protocol;
- aggregate final verdict.

### 8. Contamination

Use one clean A100.
Record:
- GPU UUID
- foreign compute PIDs
- start/end timestamps
- peak memory.

Quarantine contaminated units and rerun unchanged.

## Implementation notes

The C0/HIGS dynamic renderer has a known state-management constraint from P13-Y:

- independent oracle renders must begin from a fresh `_HIGS_DYNAMIC_SCENE.reset()` state;
- do not use `mark_dirty()` as a substitute when a previous backward may be pending.

Use the existing P13-X/P14 exact metadata path for:
- conics / covariance,
- flatten IDs,
- intersection counts.

## Return

Return the exact artifact tree required by the preregistration plus:

- oracle source files
- source diffs
- exact command lines
- environment manifest
- raw gradient NPZs
- raw stdout/stderr
- manifest sha256s.

No next experiment after verdict.
