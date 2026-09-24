# Independent Candidate Discovery for 3D Gaussian Splatting Training

## 1. Executive summary

This audit finds one direction with a credible path to a main-paper mechanism, two directions worth retaining only under strict gates, and one direction that should be dropped.

The strongest candidate is **function-preserving Gaussian fission**: replace the current discontinuous `parent -> parent + two children` split with a topology morphism that replaces the parent by children while approximately preserving the rendered function at the moment of the split. This is not a new densification threshold. It changes the semantics of representation growth. In the audited implementation, split children copy the parent's opacity and SH coefficients, their scales are halved, two children are appended, and the parent remains. Thus a split adds two Gaussians and changes optical mass immediately. Across the canonical 30k traces, split is invoked 426,139 times for room, 3,933,293 times for garden, and 2,844,388 times for bicycle. Before accounting for downstream dynamics, avoiding one redundant family member per split corresponds to 19.8%, 34.5%, and 25.7% of the respective final populations. Those percentages are an opportunity envelope, not a speedup claim.

The second candidate is a **topology-transactional optimizer**. Every densification/pruning event creates new `Parameter` objects, and the training scripts rebuild Adam. This discards the first and second moments of every surviving Gaussian, not just state belonging to births/deaths. A stable-ID transaction that transports survivor state and defines state inheritance for clone/split children could expose a previously hidden coupling between representation topology and optimization dynamics. Its direct per-iteration systems saving is small; its research value depends on a reproducible improvement in time-to-quality or population stability.

The third candidate, an **exact fan-out-separated backward**, is technically plausible but has a low main-paper ceiling. Corrected footprint data show a heavy tail, and rasterization backward performs warp-reduced atomic additions to Gaussian gradients. Routing only high-fan-out Gaussians through a partial-reduction path could remove contention without skipping gradients. However, the measured Amdahl ceiling on the available room profile is only 11.7% end to end even if rasterization backward vanished entirely.

The fourth candidate, **certified delta maintenance of camera-local intersection topology**, is rejected. Although only about 0.196% of intersection count changes per adjacent fixed-camera step, exact global reuse was 0/499 pairs, 98.53% of tiles were dirty, camera sampling changes views, and the whole forward renderer was only 6.1% of profiled GPU time. A sophisticated patchable data structure could be built, but current evidence does not justify it as the next top-conference direction.

**Answer to the central question:** if only one new direction can be tested, test function-preserving Gaussian fission. It has the best combination of a non-trivial representation mechanism, direct source evidence, potentially large downstream compute consequences, and a cheap experiment that can kill it before a training rewrite.

## 2. Current pipeline bottleneck audit

### 2.1 Audited training path

The canonical experiment path activates model parameters, renders one sampled camera, computes L1 plus SSIM, backpropagates, performs density control at 100-step intervals, and runs Adam over five parameter groups. The relevant orchestration is in `scripts/phase-c51-stage4b/canonical_training.py`; the Gaussian topology operations are in `scripts/epic05/phase7/gaussian_model.py`; the gsplat wrapper and CUDA snapshots are in `tmp_gsplat_src/` and `patches/gsplat_orig/`.

| Stage | Main inputs | Main outputs | Change frequency and lifetime | Reuse / exactness finding |
| --- | --- | --- | --- | --- |
| Parameter activation | master xyz, log-scale, raw quaternion, opacity logit, SH | xyz, normalized rotations, `exp(scale)`, `sigmoid(opacity)`, SH | Every iteration; N-sized activated tensors live through render/backward | Values change after every optimizer step. Some elementwise fusion is possible, but generic mixed precision and simple cadence changes are ruled out. |
| Fused projection | activated geometry, view matrix, intrinsics, image bounds | radii, means2d, depths, conics, optional compensation | Every sampled view and iteration; radii/conics plus master inputs are retained for projection VJP | Projection forward is 0.2% in the room profile. Backward deliberately recomputes 3D intermediates rather than retaining roughly 27 floats per Gaussian-view; the current recompute/storage trade-off is reasonable. |
| SH evaluation | directions, SH coefficients, active degree | RGB per projected Gaussian | Every view; directions and coefficients are saved for SH backward | Camera-dependent and parameter-dependent. It is not reusable exactly across views or updates. |
| Tile-count pass | means2d, integer radii, tile grid | `tiles_per_gauss` | Every forward; temporary | One thread per projected Gaussian computes its rectangular tile range. This output drives the prefix sum and allocation. |
| Prefix sum | flattened `tiles_per_gauss` | cumulative placement offsets and scalar intersection count | Every forward; cumulative offsets are temporary | `at::cumsum` is run before the second intersection pass. Topology and projected bounds make the result iteration-specific. |
| Intersection emission | projected bounds/depth plus cumulative offsets | int64 `(image,tile,depth)` keys and int32 Gaussian IDs | Every forward; emitted arrays survive through sort; sorted IDs survive through backward | One record is emitted per Gaussian-tile incidence. Exact membership changes in every audited adjacent step. |
| Radix sort | incidence keys and Gaussian IDs | depth-ordered keys/IDs | Every forward; sorted `flatten_ids` survives through backward | Global sort was 2.3% of the room profile and prior local/segmented variants were not competitive. It is not a default opportunity. |
| Tile offsets | sorted keys | int32 start offset per image tile | Every forward; saved through backward | `intersect_offset` scans key transitions. Offsets and sorted IDs jointly define the exact backward traversal. |
| Forward rasterization | means2d, conics, RGB, opacity, offsets, sorted IDs | rendered color/alpha and `last_ids` | Every view; alpha and last IDs are saved | Front-to-back compositing and early termination require exact current attributes/order. Forward rasterization was 1.8% of the room profile. |
| Backward rasterization | saved attributes/topology, alpha, last IDs, image gradients | gradients for means2d, conic, color, opacity | Every training iteration | Reverse traversal reloads the forward attributes. Per-warp reductions are followed by global atomic additions into Gaussian gradients. This was 11.7% of profiled GPU time. |
| Projection and SH VJPs | saved master inputs, conics/radii/directions, upstream 2D/RGB gradients | gradients for xyz, rotation, scale, SH | Every training iteration | Projection backward was 0.6%; SH backward consumes the rasterizer color gradient. |
| Density control | accumulated xyz-gradient norm, activated scale, opacity | appended clone/split tensors, compacted survivor tensors | Every 100 steps until 15k; opacity reset/prune every 3k | This is a discontinuous topology transaction. New `Parameter` objects replace all five arrays. Current split retains the parent and adds two children. |
| Optimizer update | five parameter arrays, gradients, Adam moments | updated parameters and moments | Every iteration; moments should be persistent but are rebuilt after topology events | At SH degree 3 there are 59 float parameters/Gaussian. Parameter + gradient + two FP32 Adam moments are about 944 bytes/Gaussian before renderer temporaries: roughly 4.72 GB at 5M and 11.33 GB at 12M. |

### 2.2 Measured bottlenecks and Amdahl limits

The clean room checkpoint profile at about 1.0M Gaussians reports 88.10 ms/iteration of GPU kernel time:

- SSIM loss: 40.7%.
- Adam: 19.8%.
- miscellaneous elementwise kernels: 17.0%.
- rasterization backward: 11.7%.
- projection + intersection + sort + rasterization forward: 6.1%.
- projection backward: 0.6%.

This profile is not universal: canonical 30k averages show forward+backward/optimizer of 49.49/7.31 ms for room, 84.64/33.63 ms for garden, and 80.79/38.28 ms for bicycle. The large-scene optimizer is therefore structurally important even though generic fused Adam would be an incremental engineering contribution.

For any candidate affecting a fraction `f` of iteration time and accelerating it by `s`, the bound used here is `speedup = 1 / ((1-f) + f/s)`. Consequently:

- eliminating the entire room forward renderer can save at most 6.1% of GPU time;
- making rasterization backward infinitely fast can save at most 11.7%; a 2x kernel improvement gives about 6.2% overall speedup;
- making Adam infinitely fast can save at most 19.8% in this profile, although its fraction is larger in the outdoor canonical averages;
- changing future population can affect several terms simultaneously and is not bounded by one current kernel fraction, but it must be evaluated causally because C51 showed that population changes dominate apparent kernel gains.

### 2.3 Persistent and non-persistent state

The rasterizer already preserves the state backward genuinely needs: `means2d`, conics, evaluated colors, opacities, tile offsets, sorted Gaussian IDs, rendered alpha, and `last_ids`. It does not preserve full projection intermediates; projection backward recomputes them. This is a sensible semantic separation rather than obvious waste.

The state that should persist across *training* iterations is optimizer history and Gaussian identity/lineage. The current tensor-replacement topology operations destroy that persistence at exactly the moments when the representation changes most. By contrast, camera-local projected state is not stable: fixed-camera C38 found no exact-valid adjacent pair in 499 comparisons.

## 3. What C49–C53 have already ruled out

- **C49–C51 sparse backward:** gradient mass is concentrated and previous gradients predict future gradients, but preserved-densification sparse backward produced only about +0.5% intrinsic end-to-end benefit. Most practical gain was mediated by a smaller future population. None of the candidates below uses low gradient as a work mask.
- **C52 gradient-ranked densification:** at matched density budgets, predictive ranking beat uniform in one condition and lost in two; the cross-scene net was approximately zero. No candidate below ranks birth candidates by historical gradient.
- **C53 visibility prediction:** current visibility's apparent predictive strength was target leakage. Its correlation with residual future tile work was approximately zero. No candidate below skips Gaussians based on predicted visibility.
- **C53 footprint prediction:** screen radius predicts future tile work, but this is physical geometry and a crowded scheduling direction. The candidate set does not filter by radius or duplicate the separate workload-persistence track.
- **C42 loss engineering:** separable/downsampled/infrequent SSIM remains composable, but it is not a main candidate here.
- **C38/C39 reuse:** exact global and tile-local reuse failed even for a fixed camera. The rejected Candidate D explicitly incorporates rather than ignores this evidence.

## 4. Structural opportunities

### 4.1 Topology change is not function preserving

`GaussianModel.densification()` copies a split parent's rotation, opacity, and SH, halves its log-scale, creates two displaced copies, appends both, and leaves the parent in place. There is no constraint that the pre- and post-split transmittance or radiance agree. Thus density control is simultaneously (1) an optimization decision, (2) an uncalibrated jump in the represented function, and (3) a population multiplier. These roles should be separated.

### 4.2 Topology change is not optimizer-state preserving

Appending or pruning replaces each `nn.Parameter`. The canonical loop then calls `get_optimizer(model)`, creating fresh Adam state for all survivors. Densification occurs every 100 steps from 500 to 14,900 (145 opportunities), with additional rebuilds after opacity-reset pruning. Therefore the nominally continuous optimizer is repeatedly restarted by representation bookkeeping.

### 4.3 The useful phase boundary is event-defined, not iteration-defined

The canonical traces do not support one universal iteration schedule. From steps 500–2,999, room has net negative topology flow while garden is strongly positive; from 10k–14,999, room adds about 0.40M net Gaussians while garden adds 2.66M and bicycle 1.63M. Gradient concentration in the C49 trace is comparatively stable after the first measurement. A paper mechanism should therefore key off topology transactions, reproduction/stability, or residual response—not a fixed iteration number.

### 4.4 Backward is an incidence-graph reduction

Forward compositing is tile/pixel ordered. The final gradient destination is Gaussian ordered. The current backward stays tile ordered and resolves this mismatch with global atomics. Corrected activated-scale measurements report median tiles/Gaussian of 39/56/21 and P99 of 25,350/38,979/4,166 for room/bicycle/garden at the audited real cameras. This suggests a small high-fan-out subset may have qualitatively different reduction behavior, but the actual atomic-stall concentration has not been measured.

### 4.5 Training-state audit limitations

The audit reused raw JSON, not report prose alone:

- canonical 30k trajectories and 145 density-event records per scene provide Gaussian counts, clone/split/prune flows, quality, and component timing;
- C49 supplies 45 gradient-distribution checkpoints and density events;
- C53 supplies cross-scene tile-work, gradient, visibility, update, and age analyses.

Two requested observables are not validly available. C49's `lifespans` array is empty, so newborn survival and population turnover by stable identity are unmeasured. The corrected heavy-tail audit supplies per-Gaussian tile percentiles but not per-Gaussian backward atomic transactions. These gaps are treated as missing evidence, not filled by inference.

## 5. Candidate A — Function-preserving Gaussian fission

**Core mechanism.** Replace the current append-only split with a local representation morphism:

`one parent -> two children`, not `one parent -> parent + two children`.

Choose child means, covariances, opacities, and optionally low-order color corrections so that, over a small certification set of rays/views, post-split alpha and color match the parent's pre-split contribution to first order. A transaction is accepted only if the measured render discrepancy is below a strict bound. Optimization can then refine the children. The key object is a certified, approximately function-preserving topology operator, not a different rule for deciding *which* Gaussian to split.

**Specific structural inefficiency.** The current operation changes both parameterization and represented optical mass. It retains the parent and appends two children with copied opacity/SH. Every split therefore adds two full parameter/gradient/Adam records and can introduce an immediate loss shock. At 59 FP32 parameters plus gradients and Adam moments, each avoidable resident Gaussian is roughly 944 bytes before renderer state.

**Why potentially important.** It attacks the causal chain `density event -> population jump -> future rendering + optimizer + backward`, while attempting to preserve the current rendered function. It could turn densification from a heuristic population explosion into a controlled model morphism. The claim, if supported, is about representation dynamics and function-preserving growth, not simply fewer Gaussians.

**Evidence already available.** Source lines 218–260 of `scripts/epic05/phase7/gaussian_model.py` show copied opacity/SH, half scale, two appended children, and no parent removal. Canonical totals are:

| Scene | Final Gaussians | Clone | Split | Prune | One-per-split population envelope |
| --- | ---: | ---: | ---: | ---: | ---: |
| room | 2,155,755 | 469,627 | 426,139 | 759,526 | 19.8% of final N |
| garden | 11,401,827 | 2,339,648 | 3,933,293 | 643,643 | 34.5% of final N |
| bicycle | 11,049,676 | 2,253,884 | 2,844,388 | 3,024,938 | 25.7% of final N |

The last column is not the expected reduction because a changed trajectory changes later births/prunes. It is the static magnitude of the current `+2` versus replacement `+1` split semantics.

**Why C49–C53 did not solve it.** Sparse backward changed which gradients were computed; C52 changed candidate allocation; C53 predicted work. None changed the algebra of a split or required the pre/post topology functions to agree. This mechanism uses neither gradient top-K, visibility, footprint thresholds, nor loss approximation.

**Missing evidence.** It is unknown whether two anisotropic children can match a parent sufficiently well under alpha compositing across several views, and whether removing the parent damages later optimization. The current split displacement/noise may also be serving exploration rather than approximation.

**Cheapest falsification.** At one real room checkpoint and one actual density event, snapshot the model and render eight fixed training views in three states: before split, current append split, and a replacement split whose two child opacities are initialized to conserve parent transmittance and whose covariance/means match the parent's first two moments. Do no optimizer steps. Record delta L1/SSIM, maximum and mean alpha error, and intersection count.

**DROP if:** no replacement initialization keeps all eight views within 0.05 dB PSNR-equivalent render change and mean alpha error below `1e-3`, or if it reduces intersections/population by less than 10% at that event. Passing this gate only establishes a morphism; it does not establish training quality.

**Expected maximum impact.** Directly affected components are population growth, future optimizer traffic, projection/intersection volume, rasterization, and backward. A defensible pre-experiment expectation is *unknown to 10–20% time-to-quality*, not the 20–35% population envelope. The maximum is governed by how much of the avoided population would remain visible and how later density events respond.

**Why this might fail.** Alpha compositing is nonlinear and view dependent; conserving a parent's moments in 3D may not conserve per-ray transmittance. Keeping the parent may be essential to cover views not in the certification set. The observed implementation may also be a simplified benchmark variant rather than the most faithful 3DGS split. Finally, function-preserving network morphisms are a general idea, so the 3DGS-specific formulation and empirical phenomenon must be strong.

**Novelty risk:** **MEDIUM.** Model morphisms are established broadly, but a certified radiance/transmittance-preserving topology operator coupled to 3DGS training dynamics is more specific than ordinary densification tuning. No novelty claim is made without literature review.

**Decision: HIGH PRIORITY**

## 6. Candidate B — Topology-transactional optimizer state

**Core mechanism.** Give each Gaussian a stable logical identity and make clone/split/prune an optimizer transaction. Surviving rows retain their exact Adam `exp_avg`, `exp_avg_sq`, and step. Clones inherit a defined parent state; split-child moments are transported through the split's parameter transformation or deliberately initialized under an explicit rule. Pruning compacts parameter and state arrays with the same permutation. No unrelated survivor is restarted.

**Specific structural inefficiency.** Current topology methods allocate new `Parameter` objects for all five arrays. The training loop then reconstructs Adam after every density event and opacity reset. This converts a local birth/death edit into a global optimizer reset and global state reallocation.

**Why potentially important.** The mechanism separates representation turnover from optimization memory. If moment resets explain loss shocks, delayed convergence, or repeated redensification, state-preserving transactions could improve quality/time-to-quality while also enabling later lifecycle mechanisms. A negative result would still establish that frequent global Adam resets are benign regularization, which is a useful training-system insight.

**Evidence already available.** The source contract is unambiguous: topology methods replace all five `Parameter` objects and the loop calls `get_optimizer(model)` after densification/pruning. Density events occur at 100-step cadence through 15k. C30 observed a newborn opacity transient, but tested only 500 steps and did not compare exact survivor-state preservation against reset. Earlier reports recognized the discarded state; the decisive training experiment remains absent.

**Why C49–C53 did not solve it.** Those phases manipulated gradient computation or density selection while retaining the same optimizer reconstruction. This candidate does not skip low-gradient updates and does not change density thresholds.

**Missing evidence.** The sign of the effect is unknown. Resetting moments may stabilize the model after a discontinuous split; preserving them blindly may amplify stale directions. Children also need a principled state-transport rule, particularly for displaced means and rescaled covariance.

**Cheapest falsification.** Add a Python state-transplant helper for survivors only, leave child moments zero, and compare current-reset versus survivor-preserving Adam on room for 5k iterations with the same camera order and random seed. Log the loss/PSNR immediately before and for 100 steps after each topology event, final population, and wall-clock time-to-fixed PSNR.

**DROP if:** post-event recovery, final PSNR, population, and time-to-quality differ by less than 0.1 dB and 5% respectively in two seeds, or preservation causes instability. Do not proceed to transformed child-state inheritance unless survivor preservation first shows a signal.

**Expected maximum impact.** Direct event overhead is small: existing audits put optimizer/mask rebuild operations below about 1 ms per event, and density-control timing averages 0.11–0.90 ms/iteration in the three canonical runs. The plausible effect is therefore convergence-mediated. Before evidence, cap the expectation at 5–15% time-to-quality and approximately 0% steady-state per-iteration kernel saving.

**Why this might fail.** Fresh Adam may be an implicit and beneficial restart after the objective dimension changes. Stable IDs and state compaction add bookkeeping. A survivor's old moment can be invalid because neighboring births change residual attribution. Optimizer-state transport is also common in dynamic-model systems, creating substantial prior-art risk.

**Novelty risk:** **HIGH.** The 3DGS topology/optimizer coupling is concrete, but state transplant by itself may be judged engineering unless it reveals a reproducible optimization phenomenon or is unified with a principled topology morphism.

**Decision: KEEP AS BACKUP**

## 7. Candidate C — Exact fan-out-separated backward aggregation

**Core mechanism.** Interpret rasterization backward as reduction on a tile–Gaussian incidence graph. Keep the existing warp-atomic path for ordinary-degree Gaussians. For the high-fan-out class, write tile-local partial gradients to a compact buffer and reduce them by Gaussian in a second pass. The classification is execution routing based on exact current incidence degree; it does not omit any Gaussian, pixel, or gradient.

**Specific structural inefficiency.** Forward requires tile/depth order, but gradient destinations are Gaussian ordered. The current reverse traversal performs warp reductions and then atomic additions for color, conic, mean, and opacity. A Gaussian intersecting thousands of tiles receives many concurrent atomic updates, while using the same kernel/data path as a Gaussian intersecting tens of tiles.

**Why potentially important.** It targets a structural mismatch rather than a missed vectorization: tile-major compositing followed by Gaussian-major accumulation. It is especially relevant at 5–12M Gaussians if a small incidence tail dominates atomic serialization.

**Evidence already available.** CUDA source confirms the atomic destination pattern. Corrected activated-scale measurements show P50 tiles/Gaussian of 21–56 but P99 of 4,166–38,979, so fan-out spans orders of magnitude. `tiles_per_gauss` correlates strongly with measured forward/backward CUDA time at aggregate level. However, the corrected audit does not report what fraction of atomic work the top 1% causes.

**Why C49–C53 did not solve it.** Sparse backward suppressed selected Gaussian gradients; this proposal computes every gradient exactly. It neither predicts footprint nor filters large-radius Gaussians. Footprint/fan-out is used only to select an exact reduction implementation for work already present.

**Missing evidence.** Atomic contention may not be the bottleneck: warp aggregation already reduces write count by up to 32x, and memory loads/compositing arithmetic may dominate. A second reduction buffer may cost more bandwidth than it saves.

**Cheapest falsification.** On one large garden or bicycle checkpoint, collect `tiles_per_gauss`, construct degree percentiles, and use Nsight Compute on rasterization backward to measure atomic throughput/stall reasons. Estimate atomic writes attributable to the top 0.1%, 1%, and 5% from the saved incidence list and `last_ids`; no kernel rewrite is needed.

**DROP if:** atomic serialization/stalls account for less than 15% of backward cycles, the top 1% accounts for less than 20% of executed atomic updates, or rasterization backward is below 8% of end-to-end time on the target 5–12M scene.

**Expected maximum impact.** On the available room profile, total elimination of rasterization backward is capped at 11.7% E2E; a realistic 2x improvement is about 6.2%. On outdoor scenes the bound must be reprofiled. This is more likely a strong systems component than a paper's sole contribution.

**Why this might fail.** The tail figures come from corrected checkpoint geometry, not executed backward contributions after alpha rejection and early termination. High tile incidence can therefore overestimate atomic work. Temporary partials can be very large, exact summation order changes floating-point results, and degree-aware graph reductions are a crowded systems pattern.

**Novelty risk:** **HIGH.** Specialized reduction paths are common; the work needs a distinct 3DGS incidence phenomenon and a broadly useful exact algorithm to rise above kernel engineering.

**Decision: LOW PRIORITY**

## 8. Candidate D — Certified delta maintenance of camera-local topology

**Core mechanism.** Maintain per-camera tile lists in stable-capacity slabs. At the next occurrence of a camera, use conservative bounds from maximum parameter deltas to certify unchanged Gaussian tile membership and unchanged pairwise depth order; patch only failed certificates. This is validity-driven reuse, not unconditional caching.

**Specific structural inefficiency.** Full count, cumsum, emission, sort, and offset encoding rebuild an incidence structure even when many Gaussian memberships may be unchanged. Stable slabs could avoid global array shifts caused by a small set difference.

**Why C49–C53 did not solve it.** It is unrelated to gradients, density ranking, visibility prediction, loss, or footprint filtering. It attempts exact maintenance of the data structure, not skipping render work.

**Evidence already available.** The only favorable number is that adjacent fixed-camera changes altered about 0.196% of total intersections by count. The decisive contrary evidence is stronger: no exact-valid pair in 499 steps, about 5,978 Gaussians changed tile assignment per step, 98.53% of tiles were dirty, and the current training loop rotates cameras. The full forward pipeline is only 6.1% of the room GPU profile.

**Missing evidence.** Set symmetric difference and within-tile depth inversion counts were not measured with stable Gaussian IDs, especially between repeated occurrences of the same training camera. Those are the correct quantities for a slab/patch design; simple tensor equality is too strict because one insertion shifts a flat array.

**Cheapest falsification.** Instrument 100 training steps to store stable-ID tile memberships and within-tile order for each revisited camera, then compute membership symmetric difference, inversion fraction, certificate pass rate, and bytes that a patcher would touch.

**DROP if:** certified unchanged incidences are below 70%, within-tile order certificates pass below 90%, or estimated patch traffic exceeds 30% of a full rebuild. Current fixed-camera dirty-tile evidence already makes these gates unlikely.

**Expected maximum impact.** Even an impossible perfect removal of projection/intersection/sort/rasterization forward saves at most 6.1% on the room profile; a topology-only implementation saves less. It may matter on a different forward-dominated deployment, but that is not the audited training bottleneck.

**Why this might fail.** Camera recurrence intervals magnify parameter deltas; topology edits invalidate stable arrays; depth-order certificates may fail even when membership is stable; bookkeeping requires persistent memory per camera; and touching nearly every tile defeats locality. The mechanism also overlaps generic incremental rendering/data-structure literature.

**Novelty risk:** **HIGH.** Certified incremental geometry is generic, while current measurements give a poor 3DGS-specific payoff.

**Decision: DROP**

## 9. Comparative ranking

The ranking prioritizes potential novelty before raw speed.

1. **Candidate A — function-preserving Gaussian fission.** Best mechanism strength and top-conference potential. It changes what a topology event *means* and has a zero-training falsification test. Implementation risk becomes high only after the gate passes.
2. **Candidate B — topology-transactional optimizer state.** Strong source-grounded coupling and easy to falsify. It is less novel alone and has no credible direct per-step speed claim.
3. **Candidate C — fan-out-separated backward.** Exact, composable, and technically clear, but Amdahl and prior-art risks make it a component rather than the main story.
4. **Candidate D — certified delta topology.** Good validity logic but contradicted by measured invalidation and a small forward ceiling.

| Candidate | Mechanism | Expected E2E Potential | Novelty Risk | Experiment Cost | Implementation Risk | Recommendation |
| --------- | --------- | ---------------------: | ------------ | --------------- | ------------------- | -------------- |
| A — Function-preserving fission | Replace a parent by optically/moment-matched children under a render-error certificate | Unknown; plausibly 10–20% time-to-quality if the static 20–35% split-population envelope survives dynamics; zero benefit if conservation fails | MEDIUM: general morphisms exist, but 3DGS transmittance-preserving topology is specific | 1–3 h: one checkpoint, one event, eight views, no training | HIGH after gate: multi-view conservation and stable training | **HIGH PRIORITY** |
| B — Transactional optimizer | Preserve survivor Adam state and define lineage-aware child state across topology edits | Direct steady-state ≈0%; capped at 5–15% time-to-quality until a convergence effect is measured | HIGH: state transplant is generic unless it reveals a new dynamic | 2–3 h code plus one 5k room run per seed | MEDIUM: tensor/state permutation is straightforward; semantics are not | **KEEP AS BACKUP** |
| C — Fan-out-separated backward | Atomic path for ordinary degree, partial/reduce path for high incidence degree; exact gradients | Room ceiling 11.7%; about 6.2% E2E for a 2x backward kernel, target-scene profile required | HIGH: workload-specialized reductions are crowded | 1–2 h Nsight + incidence attribution | HIGH: extra buffers, reduction ordering, CUDA tuning | **LOW PRIORITY** |
| D — Certified delta topology | Stable per-camera slabs patched only when membership/order certificates fail | Less than the 6.1% full-forward ceiling on the audited room profile | HIGH: generic incremental rendering; weak measured payoff | 1–3 h trace-only stable-ID audit | VERY HIGH: camera cache, topology edits, fragmentation | **DROP** |

## 10. Recommended first experiment

Run exactly one **zero-step split conservation test**: at a room checkpoint, apply one real densification event and compare eight fixed-view renders for (i) pre-split, (ii) the current parent-plus-two append split, and (iii) a parent-to-two replacement initialized to conserve parent transmittance and first two spatial moments. Measure mean/max alpha error, delta L1/SSIM (or PSNR-equivalent render error), Gaussian count, and total intersections. Do not train or tune thresholds in this experiment.

This is decisive because it tests the prerequisite structural claim: a lower-growth topology event can preserve the represented function. If it fails, the highest-novelty candidate dies before any optimizer, lifecycle, or CUDA implementation work.

## 11. Risks

1. **Implementation fidelity.** The audited `GaussianModel` is the project's canonical experimental model, but its split semantics should be compared with the exact production/original-3DGS path before a paper claim. A benchmark-specific bug is not a general contribution.
2. **Trace comparability.** C49 used a configuration with only 1.3% aggregate prune/create ratio, whereas canonical C51 traces have much larger pruning. The empty `lifespans` array prevents a stable-ID survival claim.
3. **Population counterfactuals.** Static split counts cannot predict final population or speed; all end-to-end claims require matched-quality full training and causal decomposition like C51 Stage 5.
4. **Function conservation is view sampled.** Passing eight views does not prove global equivalence. Later work would need geometric bounds or broader certification.
5. **Floating-point exactness.** Candidates A and C may change summation/compositing order. Bit exactness should not be promised unless demonstrated; quality-equivalent and numerically bounded are different claims.
6. **Profile dependence.** The detailed kernel breakdown is one room checkpoint. Outdoor 5–12M scenes need the same profiler taxonomy before systems implementation.
7. **Prior-art uncertainty.** No external literature search was performed by design. All novelty-risk labels are screening judgments, not novelty claims.

## 12. Final recommendation

Pursue **Candidate A — function-preserving Gaussian fission** first. Its paper-level hypothesis is:

> 3DGS representation growth is unnecessarily expensive and unstable because densification is a discontinuous, non-conservative topology edit; making topology edits approximately function preserving can decouple representational refinement from population explosion.

That hypothesis is structurally different from C49–C53, can be falsified without training, and—if true—supports a mechanism deeper than another threshold or mask. Keep Candidate B only as the next independent causal probe into topology/optimizer coupling. Candidate C is a possible exact systems component after profiling. Candidate D should not consume implementation effort under the present evidence.

If Candidate A fails its conservation gate, the honest conclusion from this audit is **no sufficiently novel mechanism found among the remaining candidates to justify a new main paper direction without additional empirical discovery**.
