# AtomBirth-GS — Prior-Art Guard (2026-09-30)

This is a limited current audit, not a novelty proof.

## SteepGS — CVPR 2025
SteepGS provides an optimization-theoretic densification framework:
- identifies saddle/stationary Gaussians;
- derives splitting conditions;
- chooses steepest split/update directions;
- analytically normalizes offspring opacity.

Boundary:
AtomBirth cannot claim principled density control in general.
Its candidate distinction is that a new atom need not descend from an existing Gaussian parent.

## ResGS — ICCV 2025
ResGS introduces residual split and Gaussian-image-pyramid supervision.

Boundary:
Do not use the phrase “residual densification” as the core claim.
AB0 is not residual splitting of an existing parent and does not use a Gaussian image pyramid.

## EDGS — CVPR 2026
EDGS eliminates densification using dense pixel correspondences / triangulated initialization.

Boundary:
Parent-free primitive placement is not novel by itself.
AtomBirth must distinguish online objective-driven atom insertion DURING optimization from dense one-shot correspondence initialization.

## VAD-GS — CVPR 2026
VAD-GS can add new Gaussians using multi-view stereo / visibility reasoning in sparse dynamic urban scenes.

Boundary:
Do not claim first new Gaussian creation away from existing points.
AB0 explicitly forbids external MVS/depth and uses only the current native photometric objective/render state.

## GaussianSpa — CVPR 2025
Optimization/sparsification framework; discusses Matching Pursuit as classical sparsity context.

Boundary:
Do not claim “first Matching Pursuit for sparsity” broadly.
The potential contribution, if validated, is an online function-space BIRTH operator for new 3D Gaussian atoms selected by native multiview photometric descent, rather than an opacity sparsification method.

## CVPR 2026 densification improvements and other recent work
Edge-aware scores, long-axis splits, multi-view update, uncertainty, visibility and other density-control heuristics are already crowded.

Boundary:
A new score for selecting an existing parent is NOT enough.

## Candidate narrow claim if later validated

An online parent-free Gaussian birth operator that proposes a new 3D primitive directly from the current multiview full-resolution photometric objective and accepts it based on native-loss descent, rather than deriving offspring from an existing Gaussian.

Do not claim:
- first densification optimization;
- first residual-guided densification;
- first new Gaussian away from an existing point;
- first matching pursuit in graphics;
- first multi-view densification.
