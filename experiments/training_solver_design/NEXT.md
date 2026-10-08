# Next evidence required for the active goal

## Current (2026-09-28): correlated-direction birth bounded screen fails

Read `OBSERVABLE_BIRTH_P0_20260928/room_stage0/JOINT_GATE_RECHECK.md`, the raw NPZ, Luna's report, and Astra's appended verdict in `ASTRA_OBSERVABLE_BIRTH_AUDIT_20260928.md`. The room@1500 twelve-training-view geometry-only gate is **NO-GO for this version**. A-only proxy reduction of 42.55% was dominated by sparsely seen families. The preregistered ≥3 A plus ≥2 B active-camera coverage retained 233/34,495 three-axis families, only 0.671% of their historical children; within that subset the proxy saved 5.85%. Root independently checked the prespecified every-active-view ≤1% discarded-response criterion from the SHA-verified raw matrices: only 7 families/130 children passed (0.02614% of three-axis children; 0.01308% of all 993,750 birth children). This is geometry only and 12-view limited, not accepted-work or image-quality evidence. Stop this version without lowering thresholds, adding garden, or launching new-layout CUDA/continuation. The full original goal remains open; find a materially different mechanism.

## Current (2026-09-28): correlated-direction birth is a concrete untested design

Read `CORRELATED_DIRECTION_BIRTH_20260928.md`, `ASTRA_OBSERVABLE_BIRTH_AUDIT_20260928.md`, and the frozen 1500-step child-count evidence in `OBSERVABLE_BIRTH_P0_20260928/`. Root proposes replacing SAD's per-axis tensor-product birth lattice with a lattice and child covariance jointly rotated into the eigenframe of a parent-whitened, multiview observable-frequency matrix. This aims to avoid multiplying children across parent axes that encode one image-frequency direction, while retaining the original full-resolution loss. The three-axis native groups produce 497,246/993,750 room children and 1,315,435/4,157,368 healed-garden children at the captured 1500 events. **These are only child counts**, not spectral rank, accepted hits, quality, cost, novelty, or full-training benefit. Astra found SAD already uses camera-projected per-axis frequency and reports a negative LPIPS directional ablation; only the full cross-axis frame/layout distinction survives. Its audit corrected the candidate math: native takes historical per-axis max eta and a fourth-root mapping relative to a quadratic form, so fresh-view averaged matrices are not native parity. The first gate is single-view formula parity, then fresh multiview spectrum, every-view geometry preservation and native-work share; no new-method GPU or full-training claim follows from the present algebra.

## Current (2026-09-28): joint SAD births are an untested, bounded hypothesis

Read `ASTRA_JOINT_BIRTH_P0_REVIEW_20260928.md` and `ASTRA_JOINT_BIRTH_PRIOR_ART_20260928.md`. SAD's paper already identifies overlapping primitives' shared frequency capacity as a limitation, so generic overlap-aware splitting is not original. The narrower proposal tests whether two simultaneously selected overlapping parents have antagonistic births by evaluating all four legal split-family states, then comparing a globally assembled decision with same-count independent selection on internal-check training cameras. A strict two-scene original-loss and native-work P0 is specified, but neither benefit nor deployable scoring overhead nor novelty has been established. Astra's expanded prior-art review judges **four-state enumeration plus spatial pairing and independent-score control NO-GO as an original algorithm**; it can only diagnose an opportunity for a further distinct mechanism. Luna's feasibility audit found no restorable optimizer-complete 1500 state, confirmed the author accepted-work counter and healthy official inputs, and found GPU 3 idle. Root approved **only P0 stage 1**: fresh instrumented 1500-step room/garden prefixes, split-entry capture/row mapping, candidate census and child-work screen. The expensive four-state oracle is not yet launched; the screen is a heuristic, not a net-work bound. Do not call this a new-method training arm or full-event result.

The source/CPU row-identity audit also confirms that SAD's old `custom_prune_mask` is padded but not remapped after split deletes parents, so a true bit can target another survivor. Real-scene incidence and impact are unknown. A split-entry/exit P0 must bypass later pruning and opacity cap; any full-event or continuation comparison requires a separately validated mask transport fix. This is baseline correctness work, not an original mechanism.

## Current (2026-09-28): P16 curves and same-loss SAD triad are complete, but no original method has passed

Read `SAD_GS_SAME_LOSS_P16_TRIAD_20260928/RESULTS.md` and the returned P16 v2 curves. The live EPIC-05 status now reports 12/12 curve and 12/12 profile markers, 24/24 profile JSONs, zero failures, both chains done, and GPU 2 idle; do not describe that chain as still running. All 12 curves have five successful, monotonic checkpoints, but their calibrated x axis matches native CUDA-event training time rather than complete command wall. All 12 time-to-quality results for the registered C0 quality box are null. The SAD replay endpoints remain below the room/bicycle/garden box PSNR edges by 0.345/2.574/1.750 dB.

The separate, original-loss SAD triad uses `lambda_l2=0`, full 1920×1080 inputs and all 39/25/24 common held-out views. Its room/bicycle/garden PSNR/SSIM/LPIPS are 31.1373/.923742/.275955, 24.6237/.751522/.249722, and 26.5097/.841572/.145261. Against the P16 FastGS common evaluator, PSNR changes are −0.664/+0.211/−0.189 dB, while SAD improves SSIM and LPIPS in all three. SAD train-phase walls are 137.204/146.996/168.794 s, with room time estimated from logs after a later evaluation-driver failure; different GPU/environment/concurrency and incomplete end-to-end aggregates prevent a strict full-wall ranking. This is an existing-method baseline and one seed, not an original intervention or the required multiseed three-metric win. The earlier independent room 5K original-loss duration arm remained about 0.99 dB below its FastGS reference, so simple duration extension is not a supported PSNR repair.

## Current (2026-09-28): SAD low-frequency parent-retention idea fails source/moment gate

Pinned SAD `scene/gaussian_model.py:638–831` replaces each structure-split parent with a kx·ky·kz child grid and copies opacity and SH; it does not retain the parent. But with default `scale_power=1`, each child's axis scale is σ/k and grid-center spacing is √12·σ/k, so an equally weighted *normalized additive* mixture has parent mean and covariance exactly: center variance `(k²−1)σ²/k²` plus child variance `σ²/k² = σ²`. This does not establish optical-depth or image equivalence under alpha compositing, but refutes the direct claim that deleting the parent necessarily loses low-frequency first/second moments. SAD also invokes absgrad splits and opacity/size/low-eta pruning in the same density event; independent room 3K/5K PLYs cannot isolate structural-split damage. Generic parent retention adds native hits, and opacity conservation has existing prior art/negative local screens (`TRANSMITTANCE_SPLIT_DESIGN.md`). Do not send Luna a parent-retention or simple opacity-scaling arm. A genuine causal screen would need matched same-event training-view pre/post/no-structural-split original loss and native work.

The later projected-alpha check rejects a global child-opacity correction for this SAD grid. With k=2 on one axis, centers are ±√3σ/2 and child width σ/2. For parent center opacity .5 in a side-on view, copied child opacity .5 yields center alpha ≈.2107, already below the parent's .5; any downscale deepens that hole. In an axial view the children overlap and matching .5 requires each child opacity ≈.292893 (scale .5858), whereas the side-on parent center would require scale ≈2.625, beyond valid opacity. One global scale cannot preserve both views. Smaller/broader children become a different geometry rule near BOGausS/SteepGS; no event-level quality and wall opportunity is shown. Stop the simple optical-depth calibration arm.

## Current (2026-09-28): direction and experiment ownership

Per the user's division of work, root and Astra develop and audit candidate mechanisms; Luna executes experiments. A proposed **new-method training arm** needs a distinct intervention, a training-view quality opportunity, a plausible reduction in dominant native work, and a frozen stop rule. Baseline calibration may be run separately and labeled as such. Astra's review of the healed garden P16 results found no new-method candidate meeting that gate. The P16 profile/curve chain is active on GPU 2; do not start a competing job on that card. Use the healthy garden training-view profile when it is complete; native-loss baseline differences and prior corrupt-garden diagnostics cannot select a new mechanism by themselves.

The later read-only check found the live P16 chain at 5/12 profile markers with a FasterGSFused garden training process; do not infer a stalled job from a momentary low GPU-utilization sample. In healthy FasterGS final-window profiles, raster backward remains large (room/bicycle/garden 4.5221/7.2229/8.7393 ms), while optimizer is 1.3065/4.6889/4.6637 ms. The author already provides FusedAdam and a fused method; its room final-window median total was 9.8925 ms versus FasterGS main's 9.6105 ms, and its garden native training timer was 528.2 s versus 485.7 s. These independent 20-iteration phase medians are diagnostic, not full-wall causal estimates. Merely repackaging optimizer fusion lacks both an original mechanism and a route to three-metric quality preservation.

## Current (2026-09-28): P16 garden baselines recovered from official sparse data

Read `P16_GARDEN_POINT_ORDER_AUDIT_20260928.md` and the authoritative remote `/root/epic05_p16rb/artifacts/P16_DATA_HEAL_garden.json`. The old remote garden sparse model was a September-16 synthetic/corrupt flat gray cloud of 1,839,236 untracked points, and its `images.bin` had lost 2D keypoints. Root's initial isolated byte-order conversion correctly exposed the wrong format but **did not repair the data**; never use that converted cloud as a baseline. The P16 operator independently restored the official archive's 138,766-point sparse reconstruction, verified provenance, and reran all five garden methods from initialization. Their common-evaluator garden PSNR is now FastGS 26.699, FasterGS main/fused 26.859/26.849, SADGS 26.647 and P12_C0 28.451, versus roughly 12–13 for the four corrupted native runs; all five have 24/24 images at 1920×1080, identical GT hash manifests, and clean records. The final P16 chain is running curves/profiles on GPU 2; avoid duplicate jobs. Quarantine prior EPIC-05 garden results from the corrupt shared input, including P11 garden@2000, FastGS garden stage profile, and residual–hit overlap, while retaining independent room evidence. The local Windows garden sparse files match official sizes/hashes. This is essential baseline validity progress, **not** an original quality-speed mechanism or the user's multi-seed same-protocol result. P16 native training timers and native author losses differ, so do not claim a complete-wall or original-loss leaderboard from this table. ACE-GS source remains unavailable in the bounded search.

## Current (2026-09-28): camera-calibration and shared radial-color route has no ready mechanism

Root decoded local Mip-NeRF360 `cameras.bin` for bicycle, bonsai, counter, garden, kitchen and room: each has one `PINHOLE` camera with `cx=w/2` and `cy=h/2` exactly. FastGS's loader only accepts undistorted PINHOLE/SIMPLE_PINHOLE and discards principal-point fields; that omission creates no demonstrated error on these six scenes. PINHOLE metadata does not prove residual lens error is zero. The prior global RGB-bias oracle only rejects a constant-color correction, while the `>64px` block residual is not a camera-error diagnosis. PPISP (arXiv:2601.18336v1 §4.2–5) already learns camera-shared per-channel radial attenuation, vignette and response with new-view use; 3DGUT covers nonlinear camera models and COLMAP-Free 3DGS addresses joint pose/scene estimation. A camera postprocess does not itself remove Gaussian–pixel hits and would add fitting/inference cost. No observed miscalibration, distinct intervention, or quality-and-wall causal path justifies a Luna training arm. If a genuinely distinct rule appears, first use prespecified training-only views to test transfer of a frozen shared correction, then independently establish lower native work and full-wall repayment; do not fit on formal held-out GT.

## Current (2026-09-28): pre-exponential alpha rejection is not a 35K quality-and-wall mechanism

Root and Astra audited pinned FastGS `PerGaussianRenderCUDA`: it calls `exp(power)` before discarding `alpha<1/255`; for finite positive opacity a per-lane log-domain cutoff could reject some pixels before that exponential. Direct FP32 threshold replacement needs conservative boundary handling, and the native backward still does pixel/warp state transfers, power math, accepted gradients and atomics. Critically, the previously quoted 45–63% accepted/visited ratio is a **forward replay** denominator; backward already excludes rows beyond saved `n_contrib`, so it is not the fraction of backward exponentials this gate could save. gsplat's experimental MacroTileRasterize already compares opacity in the log/sigma domain while evaluating exp2 in parallel; Faster-GS/TruncGradGS address adjacent support/truncation behavior. This is a possible arithmetic microoptimization, not a distinct image-quality mechanism or established original method. The measured room 4099→35K complete continuation costs 291.46 s versus 262.40 s for an independent 30K reference; a combined exact-kernel-plus-extra-steps plan must save over 29.06 s (9.97% of its 35K wall) before claiming faster training. No measured backward saving supports that gate, and the 35K quality gain is only one developed room trajectory. Do not hand Luna a 35K or CUDA arm on this rationale alone.

## Current (2026-09-28): camera-order speed margin does not support a cancellation-based topology fix

Read the 2026-09-28 supplement to `FASTGS_CAMERA_GRADIENT_ORDER_SCREEN.md`. One full-resolution, original-loss FastGS room 4099→30K pose-balanced order did lower complete continuation wall from 262.40 to 217.85 s and final Gaussians from 566,856 to 357,515, but worsened all three development-view metrics (PSNR −0.115685 dB, SSIM −0.003455, LPIPS +0.010373). The tempting explanation that cross-camera *signed* gradient cancellation suppressed births is incompatible with pinned FastGS source: it norms each view's screen gradient before accumulating nonnegative densification statistics, uses a separate absgrad statistic for splits and intersects with multiview importance. The earlier xyz-gradient prefix oracle is not that statistic. The causal source of the point-count difference remains unknown; lowering thresholds or enforcing a point floor is not a new quality/cost mechanism. GraB already covers inexpensive online gradient-balanced ordering and BOGausS already adapts position/scale steps from camera-projection uncertainty. Do not assign Luna a renamed camera-order or simple preconditioner arm. Seek a distinct rule with evidence for preserving the 17% wall margin while closing the observed triple-metric deficit; none is ready now.

## Current (2026-09-28): shared broad-splat parameters and weaker SAD filtering do not supply the missing mechanism

Root and Astra source-checked two more quality/cost hypotheses before GPU work. Sharing parameters among broad Gaussians leaves their per-pixel alpha hits and native raster backward intact; the valid room late windows in `FASTGS_STAGE_PROFILE_RESULTS.md` show raster backward 52.7–52.9% of measured CUDA kernel time and optimizer 0.6–1.3%. Its old garden rows came from corrupt initialization and are quarantined. Healthy P16 garden final-window phase medians separately show FastGS raster backward 5.0056 ms out of 8.6649 ms total and optimizer 0.0119 ms; this 20-step instrumented profile has a different timing definition and is not a full-wall estimate. Replacing many hits by one spatially expressive splat overlaps Textured-GS/SVGS/LiteTex-GS/RBS and still needs a distinct representation plus measured hit/cost reduction; generic sharing is not such a proposal. For the original-loss SAD room 5K deficit, pinned author `arguments/__init__.py` defaults `compute_3d_filter=False`, and the frozen training command did not enable it (`SAD_GS_ROOM_L2ZERO_5K_20260927/FROZEN_PROTOCOL.md`); author train code sets `filter_3D` to all zeros in that branch. There is therefore no enabled 3D filter to weaken in this measured trajectory. SAD's separate 2D low-pass adds 0.1 to projected covariance diagonals and compensates opacity by an area ratio, whereas FastGS adds 0.3 without that compensation. Merely changing SAD's 0.1 is an antialiasing configuration ablation without a demonstrated original-loss quality or dominant-work mechanism. The prior `>64px` block-residual share is descriptive, not a Fourier proof that filtering is irrelevant. No Luna arm follows; continue searching for a distinct causal intervention.

## Current (2026-09-28): learned sharp splat shapes already have a direct prior-art and quality boundary

Root and Astra checked [GES (CVPR 2024, arXiv v2)](https://arxiv.org/html/2402.10128v2), rather than sending Luna a generic “sharper splats need fewer splits” experiment. GES §4.2 learns a shape parameter but implements its effect by rescaling Gaussian covariance via an effective-variance approximation; it does not evaluate a generalized-exponential kernel at each hit and notes view-dependent boundary error. Its default objective adds a DoG frequency-mask term (`λ_frequency=.5`, `λ_L1=.3`, `λ_SSIM=.2`), outside this goal's original-loss protocol. Table 2 already tests removal of that term on Mip-NeRF 360: original GS `27.21/.815/.214` versus GES without frequency loss `27.07/.800/.250` (PSNR/SSIM/LPIPS); full GES is `26.91/.794/.250`. The smaller model (`411 MB` versus `734 MB` without the term) is not a three-metric quality win. Table 1 warns that reported training times can use different GPUs, and its rendering speedups cannot substitute for same-protocol full-process training wall against FastGS-Big or ACE-GS. This does not rule out every non-Gaussian compact kernel, but a new proposal needs a distinct per-hit/representation mechanism, an original-loss quality cause and a measured dominant-work saving. No Luna arm follows from GES's shape/scale idea alone.

## Current (2026-09-28): cost-negative native split is not an evidenced next experiment

Root and Astra reviewed the proposed high-cost-parent split before assigning GPU work. Native FastGS divides each child scale by 1.6, so an ideal continuous 2D area calculation gives two children 2/1.6² = 0.78125 of a parent's area. But the saved room@4100 native event on one 1600×1066 training camera (`fastgs_event_split_cost_result.json`) replaces 9,345 selected parents' 21,173 Gaussian–tile entries with 27,658 child entries: **+6,485 (+30.63%)** for those families, or **+0.75%** of the 866,228-entry whole scene. Paired forward+original-loss+backward CUDA medians are 6.1578→6.2170 ms. This is one fixed event/camera and not full training, but directly defeats the assumed aggregate cost saving there; tile entries are not accepted-pixel work. An individual cost-negative split may still exist, without established frequency, useful quality effect, or total-wall repayment. The old `TOPOLOGY_DESIGN.md` already allows negative incremental footprint cost, while `REAL_TOPOLOGY_RESULTS.md` and `FASTGS_EVENT_GATE_RESULTS.md` find unstable or tiny long-horizon quality signals from finite-edit and one-shot split scoring. EDC Long-Axis Split and ShorterSplatting are close geometry/footprint-cost prior work. **Do not send Luna a renamed native split score, simple shrink, or training arm on this basis.** The full original multi-scene objective remains open.

## Current (2026-09-28): continuous split gate alone already fails its quality opportunity screen

Read `FASTGS_VECTOR_GATE_RESULTS.md` before reviving topology homotopy. At one room@4100 event, 9,345 family gates trained for 12 Adam updates on ten original-loss full-resolution training views with child/model parameters frozen: train loss improved, but versus no split the 39 developed view metrics moved PSNR −0.002344 dB, SSIM −0.000165 and LPIPS +0.000470. This does not test joint gate+child adaptation, yet it removes the claim that continuous gates themselves are untested or quality-positive. The 1.2913 s offline gate work excludes complete trainer costs. Astra's source/cost review found no frozen exploration/probation/commit rule or credible repayment mechanism for native integration; do not launch Luna on a renamed one-shot veto, gate-only repeat or CUDA fusion without a new causal reason. The full original multi-scene objective remains open.

## Current (2026-09-28): tile-prefix ideal opportunity gate fails; stop joint-transmittance culling

Read `TILE_PREFIX_TRANSMITTANCE_GATE_20260928/RESULTS.md` and its frozen `ideal_gate_results.json`. Luna reconstructed exact native FastGS-Big room 30K tile lists on 8 training cameras × 256 selected tiles; one tile per camera passed CPU/native `n_contrib`, RGB and final-T replay checks. Across 240,560 sorted entries the **ideal** all-pixels-done removable suffix is 27,777 (11.5468%); 0/8 cameras reach the frozen 20% per-view threshold. Root recomputed raw tile/view sums, the 960 all-done/937 positive-suffix counts, every parity threshold and the script hash. Both frozen gates fail (needed 40% pooled and four cameras at 20%), so no conservative certificate, production CUDA, training or held-out evaluation was run. This is a sampled single-room fixed-state upper bound, not a general impossibility result. Do not revive this culling direction by adjusting bins/thresholds on the same data; look for a different representation or training mechanism with native-work and quality causality.

## Current (2026-09-28): joint foreground tile transmittance certificate has a frozen Luna opportunity gate

Read `TILE_PREFIX_TRANSMITTANCE_GATE_20260928.md` and `TILE_PREFIX_PRIOR_ART_20260928.md`. A conservative tile-wide alpha lower bound from each foreground Gaussian's four pixel-rectangle corners can upper-bound transmittance; after a completed depth bin is proven to take every pixel below native early-stop threshold, strictly deeper Gaussian–tile entries could be omitted before sort. Preserve the triggering prefix: FastGS marks done before writing its `test_T`. AccuTile addresses single-Gaussian ellipse support, while OccluGaussian uses region visibility and Proxy-GS a mesh-depth prior; a full novelty audit would be required if the idea were revived. The frozen opportunity screen has now failed at its ideal upper bound; see the newer result above. The older Gap-A recommendation was marked obsolete because AccuTile already exists.

## Current (2026-09-28): covariance projection-nullspace edit loses freedom across views

Read `COVARIANCE_PROJECTION_NULLSPACE_GATE_20260928.md`. A rank-two camera has a three-dimensional symmetric covariance edit nullspace `ΔΣ=n vᵀ+v nᵀ`, but two general views leave one dimension and three general views none; a seeded NumPy example returned ranks 3/5/6. A second-view original-loss descent in the first-view nullspace does not imply lower native work or preserved third-view quality. The simple all-view ray-thickness shrink overlaps surface-style priors and has unknown loss effect. Rotation/log-scale/Adam conversion and possible opacity compensation add cost/semantic issues. This distinct mathematical idea does not yet justify Luna training; do not mistake one-view footprint invariance for a multiview quality-and-cost mechanism.

## Current (2026-09-28): two-lobe sort-entry packing has no demonstrated full-training opportunity

Read `COMPOSITE_SIBLING_SORT_ENTRY_GATE_20260928.md`. Packing two independent Gaussian lobes into one sort entry preserves rendering only where the lobes are consecutive in the original order; an intervening effective splat changes alpha compositing. At most the overlapping tile-support count is removed from entry generation/sorting; both lobes' projection, per-pixel evaluation, gradients and optimizer state remain. Existing moved-child screens show frequent interleaving, and no eligible-overlap/cost bound or quality mechanism exists. Do not launch Luna on this version. The old Splatshop characterization as a 2024 merging trainer was incorrect and has been fixed using the later source audit; this correction does not establish packing originality.

## Current (2026-09-28): P17 coarse-loss gradient route is outside this goal's per-step loss protocol

Read `P17_PROTOCOL_BOUNDARY_20260928.md` and `P17_P0_Multirate_Gradient_Oracle_Bundle/REPORT.md`. P17-P0's frozen oracle area-downsamples GT and evaluates d=2/4/8 losses; directly routing those gradients into training would change the required original full-resolution loss per optimization step. The P17 execution has now ended with preregistered `P17_P0_HARD_DROP`, independently checked in remote `/root/epic05_p17/P17_VERDICT.json` (SHA256 `7b3b52f5dfbe4f40b5115ca8777f9bfecb4b6d7c3fe100c8d04c1e4221cbb7bf`): room/bicycle median theoretical interaction-cost ratios were .09065/.03215, but one-step full-resolution loss-improvement ratios were -.01532/-.00245, and per-Gaussian mean-gradient cosine medians were .000539/.000373. These are frozen-state oracle results, not realized wall savings. The frozen gate forbids garden or P17-P1 continuation. The full-image-adjoint variant was separately stopped at the native state/cost gate. P16-RB has healthy baseline results but its profile/curve chain remains live; neither project establishes the requested original multi-scene result.

## Current (2026-09-28): expensive primitives also carry training utility; frozen-fragment cache lacks reuse

Read `HIGH_UTILITY_COST_AND_FRAGMENT_CACHE_GATE_20260928.md` and the existing P15-P0 report. Its six-cell fractional-knapsack upper bound retained the predefined minimum gradient-energy viability level in 0/6 cells at 50% of exact Gaussian–tile intersection budget; this is not a full-training impossibility proof. Caching ordered color/transmittance fragments is algebraically possible only while their parameters and order stay fixed, but the high-work population lacks a justified freeze rule, active/frozen fragments interleave, and per-view storage/refresh costs remain. Existing room 20K→30K improves all three metrics; LFH and Flash Cache are close prior work. Do not ask Luna to rerun primitive selection/freeze. Focus on a representation or render-work change with an identified quality mechanism.

## Current (2026-09-28): full-image adjoint plus per-Gaussian sparse backward lacks a native-work and originality path

Read `FULL_RES_ADJOINT_MULTIRATE_SOURCE_GATE_20260928.md`. The candidate keeps full-resolution forward/original loss but samples broad Gaussian parameter gradients. FastGS/SAD bucket backward must still visit every accepted hit to propagate alpha, transmittance and accumulated color for later Gaussians; directly skipping hits corrupts narrow-point gradients. Keeping the recurrence saves only some derivative accumulation, not the whole interaction. This shares P17's footprint-based per-primitive gradient-rate core, while full-image adjoints only change the estimator relative to coarse-image P17 and the old 75% random mask. Wide footprints can carry high-frequency occlusion and pixel-adjoint structure; no causal quality repair or material measured instruction saving is established. No Luna GPU arm follows. Seek a different mechanism with a direct cost path and an identified multiview failure.

## Current (2026-09-28): FastGS → SAD-GS PLY handoff stopped at source/cost gate

Read `FASTGS_SAD_HANDOFF_SOURCE_GATE_20260928.md`. The two-stage recipe is a combination of existing methods, not a new mechanism. Its unchanged SAD `--ply_path` path calls optimizer setup while `spatial_lr_scale` is still zero, so xyz Adam LR and its scheduler are zero even after the extent field is later assigned. PLY fields are compatible, but optimizer/RNG/camera/topology state is absent. The existing 10K/20K FastGS tqdm times cannot substitute for complete prefix process wall; after counting any non-loop cost, the SAD budget is less than 194.27/116.27 s against a 287.27 s from-zero FastGS target. The measured 30K loop-external overhead is 58.852 s; a rough 10K-prefix-plus-independent-SAD-3K estimate is already 292.8 s before transfer. A separate same-protocol FastGS parallel-loader run has 248.50 s process wall. These are estimates/reference runs, not handoff timings or an impossibility proof, but leave no demonstrated repayment margin. No GPU handoff arm is authorized by this evidence. Fix and verify the loader order only if the recipe is later needed as a reference baseline; keep searching for a distinct quality-and-native-cost mechanism.

## Current (2026-09-28): no-bbox SAD initialization is published and lacks a traceable cost/quality cause

Read the 2026-09-28 supplement in `SAD_GS_PRIOR_ART_20260927.md`. The SAD paper already reports a no-bbox ablation: quality nearly identical, while bbox initialization slightly improves peripheral appearance and consistently yields fewer final Gaussians. The local author code adds 1,536 bbox-face primitives once before training but stores no ancestry in the final PLY; subsequent births/prunes make endpoint IDs insufficient to attribute the broad PSNR deficit or work to bbox descendants. Disabling the flag is an existing-method configuration, not a novel algorithm, and the published point-count direction undercuts the assumed cost saving. No new Luna training arm follows. Continue searching for a distinct intervention that ties a demonstrated multiview quality failure to net native-work or convergence reduction.

## Current (2026-09-28): top-32 high-cost SAD primitives fail the frozen work opportunity gate

Read `SAD_HIGH_COST_CAUSAL_GATE_20260928.md` and `SAD_HIGH_COST_CAUSAL_GATE_20260928/RESULTS.md`. Luna used the pinned SAD original-loss room 5K model and the eight pre-registered full-resolution training cameras at `mult=.25`, ranking Gaussian IDs by summed native accepted-pair counts without test data. The 32 costliest IDs account for only **2,654,114 / 547,515,685 = 0.4848%** of those pairs, far below the frozen 5% opportunity filter. Root recomputed the sums and verified every local evidence SHA256. No held-out view, counterfactual edit, or training arm was run. This rejects a *few-most-expensive-splats* removal/replacement route to a material wall-time gain on this state; it does not prove that broader groups, other stages or scenes lack opportunity, nor does accepted-pair share directly measure backward milliseconds. Astra's independent review notes that top projected-footprint groups carry cost but their identity is not tied to the broad held-out residual, and deleting an occluder can expose deeper work. Ordinary shrink/prune/split and perspective-correct/culling have direct prior-art overlap. No credible original quality-and-cost edit passes this gate; continue mechanism search before new GPU training.

## Current (2026-09-28): SAD hybrid SH mask incidence is real; union correction loses room 3K metrics

Read `sad_hybrid_last_mask_gate_20260927/REAL_VIEW_AND_UNION_3K_RESULT.md`. In four real full-resolution room training-camera pairs at the SAD original-loss 5K PLY, the final-camera SH sparse mask skipped 6.65%–85.45% of the accumulated higher-order SH gradient squared norm; this is fixed-state incidence, not a trajectory-wide rate. A single isolated 3K union-of-batch visibility correction kept the original loss/protocol and clean author source, trained in 140.158 s versus unmodified SAD 140.937 s, but all 39 official held-out metrics worsened: PSNR 31.175938 versus 31.360905, SSIM .929890 versus .930564, LPIPS .180953 versus .180577. GTs and frozen inputs match. This rejects the correction alone as a quality-and-cost candidate on the developed room gate; one run cannot prove a general negative effect. It is a baseline implementation issue, not an original method. Stop this arm, return to a distinct mechanism with a direct cost path and multiview quality rationale.

## Current (2026-09-27): isotropic covariance gauge fails sustained-cost gate

Read `EIGEN_GAUGE_PRIOR_AND_SOURCE_GATE_20260927.md`. An exactly isotropic Gaussian can change its rotation without changing its covariance or current image, suggesting a gradient-aligned scale basis. Pinned FastGS/SAD code guarantees widespread isotropy at initialization, but split/clone inherit anisotropic parent axis ratios and are not guaranteed to provide repeated gauge opportunities. The basis change adds work and saves no current native interactions; a speed claim would rest entirely on fewer later iterations, with no link to the measured broad SAD PSNR deficit or three-metric noninferiority. A one-step mathematical descent probe would be too weak. Reject this candidate at the source/opportunity gate without a Luna experiment.

## Current (2026-09-27): SAD hybrid SH last-camera mask has a conditional lost-gradient path

Read `SAD_HYBRID_MASK_SOURCE_AUDIT_20260927.md` and `sad_hybrid_last_mask_gate_20260927/RESULTS.md`. Astra traced the frozen author commit: with `batch_size=2`, both camera backward calls accumulate gradients, but `last_radii` is overwritten by the second camera and its mask alone gates `hybrid` sparse higher-order SH Adam. The native optimizer skips parameter and moment updates when this mask is false, then the accumulated gradient is cleared. Luna's independent pure-Python CPU gate and root's rerun pass the minimal condition: a row visible only in camera 1 with nonzero SH gradient is skipped; a last-camera-visible control updates; a union-mask counterfactual updates the first row. The source-level conditional is confirmed, **not** its incidence in real room training or its quality/wall impact. Geometry/DC dense Adam remains unaffected. A union mask is an ordinary correctness fix, not an original quality-and-cost mechanism; it may increase optimizer work and cannot be claimed to repair SAD's PSNR gap. No GPU or full-training arm follows from this CPU gate alone.

## Current (2026-09-27): HTGS closes the proposed front-core/tail mechanism

Read `HTGS_PRIOR_ART_20260927.md`. HTGS already depth-sorts and alpha-blends the first `K` per-pixel fragments, aggregates the remaining tail with alpha-weighted color and product transmittance, and precomputes shared tail derivatives. Its perspective-correct 3D Gaussian evaluation plus tighter culling also already gives a published speed and three-metric gain over original 3DGS in one five-outdoor-scene ablation. Default HTGS loses some PSNR against 3DGS in both Table 1 dataset aggregates, and no published or local result yet establishes the required same-protocol FastGS-Big/ACE-GS complete-wall and triple-quality win. Its official code is available for a later comparator. The proposed front-core/tail approximation and perspective-correct-plus-culling combination are not original candidates; no Luna training arm follows. Root and Astra continue searching for a distinct, falsifiable 3D-state quality/cost intervention.

## Current (2026-09-27): accepted-mask transpose native performance gate failed

Read `fastgs_mask_transpose_native_20260927/CUDA_GATE_RESULT.md` and its raw `parity.json` plus GPU3 timing JSONs. The isolated native candidate passed author FastGS RGB/loss and all six parameter plus four screen-gradient fields on the four pre-registered full-resolution room training views. On a clean GPU3, the three paired medians for 128 fixed-state forward+original-loss+backward calls gave candidate/native wall ratios **1.244, 1.291, 1.238**; root independently recomputed these from the raw files. Forward-only ratios were 1.047, 1.051, 1.071 with mask clear/write included. This fails the frozen <=0.95 performance gate despite numerical parity. The earlier GPU2 timing was invalid due to concurrent load and is excluded. Stop this particular mask/transposition implementation: no production integration, 30K run or claim of training acceleration. The fixed-state blocks exclude optimizer, data loading, topology and convergence, so they do not measure complete-training wall or quality. Astra still found no original quality-and-cost mechanism; the full multi-scene/multi-seed, unseen-view, three-metric strong-baseline goal remains open.

## Current (2026-09-27): phased split/clone plus low-resolution acceleration is occupied

Read `DECOMPOSING_DENSIFICATION_PRIOR_ART_20260927.md`. Huang et al. already schedule early split-driven spatial spread versus later clone refinement, with energy-guided downsampled early training and adaptive opacity pruning. Their own Mip-NeRF 360 G2L-only ablation is faster but worse in all three image metrics; the full method is faster and higher in PSNR/SSIM than its 3DGS-accel baseline but worse in LPIPS. Downsampled stages also violate this project's full-effective-resolution protocol. This closes a tempting relabeling of phase-separated split/clone as an original solution; it is not a same-protocol FastGS-Big/ACE-GS comparison. No new Luna training arm follows.

## Current (2026-09-27): synthetic 32×64 mask-transpose gate passed, full goal still open

Read `fastgs_mask_transpose_gate_20260927/RESULTS.md`. Luna's corrected NumPy FP32 reference exercised one fixed 64-pixel × 70-Gaussian synthetic tile, with explicit 32×64 `w=Tα`/`h=G·∂L/∂α` arrays, accepted masks, per-Gaussian cross-pixel reduction, early stop, background-only pixels, saturated alpha and a partial bucket. All 950 accepted hits matched mask popcount; reconstructed mean/absgrad, conic, color and opacity had maximum absolute errors of 9.54e-7, 5.96e-8, 5.96e-8 and 1.42e-7 against a separate native-equation reference (fixed tolerance 2e-5). Root reran the final script successfully. This is **one synthetic CPU case**, not real-view CUDA parity or performance. The first draft of the script incorrectly zeroed saturated-alpha spatial gradients and reused full adjoints in the candidate path; both were corrected before the reported result and must not be cited as evidence. The next systems gate, if this component remains relevant, is isolated native CUDA parity plus complete-step timing including forward mask writes, occupancy, barriers and atomics. Do not promote it to a production change or 30K run from this CPU result. Astra's independent direction review still found no specific original 3D-state intervention connecting improved original-loss multi-view quality to lower dominant complete-training cost; the exact-backward component alone does not satisfy that missing requirement.

## Current (2026-09-27): bounded backward-layout gate pending; no quality mechanism

Read `FASTGS_BACKWARD_BUCKET_OPPORTUNITY.md` final section. Astra retained only a tentative accepted-mask plus shared-memory two-scalar transpose for FastGS's 32-Gaussian backward buckets. It keeps native T/RGB bucket boundaries and proposes an extra actual-acceptance bitmask; mathematical parity must include saturated-alpha custom VJP, background, partial buckets, early stop, all parameter gradients and per-contribution absgrad. A full 32×64×2 FP32 shared array is 16 KiB and mask storage is nominally 25% of the existing four-FP32 boundary storage, before other costs. Neither speed nor novelty is established. Luna's two attempts at the CPU/reference and resource gate were interrupted by model-service rate limits, so **no experiment ran**. Do not implement a production kernel or launch 30K until that bounded gate passes, then require complete-step timing including forward mask writes. This is at most a systems component; no original mechanism yet connects SAD's broad PSNR error to lower full-training cost. Astra also rejected a training-view broad-residual comparison because either result would not decide a specific cost-saving intervention.

## Current (2026-09-27): COLMAP tracks cannot supply a causal broad-error label

Read `SAD_GS_ROOM_L2ZERO_5K_20260927/diagnostic/sfm_coverage/RESULTS.md`. Eight pre-registered room training views had a median 216/425 64px blocks with at least three linked COLMAP tracks, but coverage ranged 53–288/425 and only a median 122.5 blocks had ten tracks. The tracks are sparse and both SAD-GS and FastGS initialize from the same COLMAP point cloud; they are not independent dense geometry ground truth or evidence that correcting a depth discrepancy would reduce accepted training work. Existing final PLYs also lack matched topology/history for clean geometry/opacity/appearance swapping. Stop a broad depth-discrepancy repair arm on this basis. The unresolved mechanism question is which 3D state causes the large-scale held-out errors **and** why a legal correction would lower dominant complete-training cost without sacrificing SSIM/LPIPS. No candidate passes that gate yet.

## Current (2026-09-27): SAD excess room error is broad-scale, source unknown

Read `SAD_GS_ROOM_L2ZERO_5K_20260927/diagnostic/multiscale_error/RESULTS.md`. Exact block-projection MSE decomposition of the frozen 39-view SAD 5K and timed FastGS-Big 30K PNG residuals assigns 79.45% of SAD's extra mean MSE to `>64px` block scales; in a purely descriptive 38-view calculation excluding the known outlier, that share is still 64.26%, and `<4px` excess is negative. Thus fine-detail/frequency reinforcement is not the direct explanation for the present PSNR gap. A block-scale residual does not identify a 3D cause or a dominant-cost opportunity; previous high-residual accepted-work counters found no hotspot. Root and Astra still have no original intervention with a credible full-training speed and triple-quality path. Before GPU work, a candidate must tie a specific 3D state to the broad residual and show why correcting it reduces expensive training work or iteration count. Keep all 39 views in official metrics.

## Current (2026-09-27): global RGB-bias oracle cannot close SAD PSNR gap

Read `SAD_GS_ROOM_L2ZERO_5K_20260927/diagnostic/color_bias_oracle/RESULTS.md`. Frozen 39-view room render PNGs from SAD-GS original-loss 5K and timed from-zero FastGS-Big 30K share byte-identical GTs. A privileged per-view, per-method additive RGB offset fitted using each test GT raises mean PSNR from 31.218957 to 31.403328 for SAD and 32.205460 to 32.399964 for FastGS. The FastGS-SAD gap therefore grows from .986503 to .996636 dB; even this non-deployable oracle does not close it. The isolated `00021.png` veil gains 1.197 dB but remains more than 9 dB below FastGS on that view. Stop simple global exposure/color-bias repair as an explanation or original-method direction. This is CPU error attribution, not training speed, multi-scene quality or a method. The full goal remains active.

## Current (2026-09-27): paired held-out mult counterfactual closes support-mismatch explanation

Read `SAD_GS_ROOM_L2ZERO_5K_20260927/diagnostic/test_mult025/RESULTS.md`. The same independent original-loss 5K SAD checkpoint was rendered on all 39 official room test views at the training tile-support setting `mult=0.25`, isolated from the author-default `mult=0.5` official output. The author metrics changed PSNR 31.218960→31.303608 (+.084648 dB), SSIM .930894→.930532 (worse .000362), LPIPS .177528→.176773 (better .000755). Against the timed FastGS-Big room reference, PSNR still trails .901852 dB. The `00021.png` veil view gained only .097504 dB PSNR. This rules out the train/render support mismatch as the main explanation for the official .986500 dB PSNR deficit; it does not make `mult=0.25` an official result or an original method. Do not run threshold scans or a new training arm based on this observation. The full multi-scene, multi-seed original-method goal remains unmet.

## Current (2026-09-27): SAD train/render support differs; not a new method

Source inspection of pinned SAD-GS `auxiliary.h` found that `duplicateToTilesTouched` multiplies the opacity-threshold ellipse's squared radius by `mult`. The frozen room training used `mult=0.25`, while author `render.py` defaults to `0.5`; this changes tile support and can change images, not just timing. The eight pre-registered training views in `SAD_GS_ROOM_L2ZERO_5K_20260927/diagnostic/DIAGNOSTIC.md` already provide same-checkpoint comparisons: mean RGB MAE increases .013158→.013512 for 3K and .012164→.012529 for 5K when `mult` grows .25→.5, while the top-10%-footprint accepted-work share rises from about 50% to 56%. This is a training/render semantic difference, not proof that it caused the official held-out PSNR deficit; official 39-view evaluation remains unchanged. Simply matching support or sweeping `mult` is not a credible original algorithm, and adjacent truncation/support-gradient ideas have prior art. Do not promote it as success or start another full training arm on this observation alone.

## Current (2026-09-27): frozen SAD training-view fog/work diagnostic closed

Read `SAD_GS_ROOM_L2ZERO_5K_20260927/diagnostic/DIAGNOSTIC.md`. Luna compared the independent original-loss 3K and 5K room checkpoints on eight pre-registered, evenly spaced training cameras at the original 1600x1066 resolution. At the training rasterizer setting `mult=0.25`, mean RGB MAE improved .013158 to .012164 on 7/8 views; the top 10% residual pixels accounted for only 10.51%/10.58% of accepted Gaussian-pixel pairs, and semitransparent output pixels accounted for only 1.02%/1.10% of those pairs. The author render setting `mult=0.5` showed the same qualitative pattern. These are forward accepted counts, not backward time or a causal duration comparison. The isolated held-out yellow veil was not reproduced as a broad training-view regression or a dominant accepted-work hotspot. Stop the fog/high-residual-cost direction; do not run pruning, threshold sweeps, adjacent-view mining or new training on this basis. Root and Astra still have no credible original mechanism simultaneously predicting lower dominant full-train cost and three-metric noninferiority. Keep the complete original goal active and the official 39-view metrics unchanged.

## Current (2026-09-27): SAD-GS original-loss 5K frontier gate failed PSNR

Read `SAD_GS_ROOM_L2ZERO_5K_20260927/RESULTS.md`. One frozen, independently launched room 5K run retained original `0.8 L1+0.2(1−fusedSSIM)`, full effective 1600-wide input and the same 272/39 split. Train process wall was 188.996 s versus timed FastGS-Big from-zero 30K 287.270 s; SSIM/LPIPS were .930894/.177528 versus .930243/.188115, but PSNR 31.218960 versus 32.205460, a **.986500 dB deficit**. All 39 test GT PNGs match. Twenty-five views improved PSNR relative to the independent SAD 3K arm, but one view lost 9.6179 dB and shows a broad translucent veil; the 5K and 3K topology counts already differ at 3K, so this is not an identical-state continuation experiment. Excluding that view still leaves a descriptive −.734594 dB 5K-vs-FastGS gap, and removal might increase native accepted work. Do not remove the outlier from the official mean, claim a causal source or run fog-specific pruning without a training-view cost/quality mechanism. Stop duration-only sweeps, including 7K. The full original-method, multiscene/multiseed triple-quality and training-wall goal remains open.

## Current (2026-09-27): SAD-GS original-loss room arm completed

Read `SAD_GS_ROOM_L2ZERO_3K_20260927/RESULTS.md` and its frozen protocol. Setting only `lambda_l2=0` makes the SAD-GS scalar image loss match FastGS's `0.8 L1+0.2(1−fusedSSIM)` expression. On the same 272/39 room split, effective 1600-wide images and A100 GPU3, the complete author train/render/metrics commands all passed: training process wall 140.937 s, test PSNR 31.360905, SSIM .930564, LPIPS .180577. Versus the 3K author L2=2 arm, this is −.074223 dB PSNR, +.000216 SSIM, and −.000732 LPIPS, with 70,407 fewer final Gaussians. GT PNG content and test names match. Against the already timed from-zero FastGS-Big room 30K run with identical test GT PNGs, SAD trains 2.038× faster and has slightly better SSIM/LPIPS, but **PSNR is .844555 dB lower**. One run/scene is not a quality or time variance estimate; changing the loss also changes topology. The sole subsequent 5K arm is recorded above. This calibrates an existing method but does not provide an original mechanism or required multiscene, multiseed strong-method win. The goal remains open.

## Current (2026-09-27): SAD-GS author room 3K baseline completed

Read `SAD_GS_ROOM_AUTHOR_3K_20260927/RESULTS.md` and its frozen protocol. On idle A100 GPU3, the pinned author indoor Config 2 ran 3K training, rendered all 39 official room holdout views, and completed author metrics: PSNR 31.435127 dB, SSIM .930348, LPIPS .181309. Training **process** wall was 147.618 s, versus the author's 76.751 s summed CUDA-event training intervals; full train→render→metrics wall was 314.032 s. Peak sampled GPU memory was 31,729 MiB. This uses the author's extra L2 weight 2.0, so it is a source-available author baseline calibration, **not** an original-loss same-protocol quality/speed comparison or a new-method result. The three author commands exited 0 and 39/39 render/GT and per-view metric entries were independently mapped to sorted test cameras. The outer supervisor's retained nonzero exit came only from its incorrect expectation that render files use source image names rather than numeric indices; attempt 001 was separately stopped during initialization by a host/container PID mismatch. Do not rerun training to repair either monitor error. The author checkout is clean, GPU3 idle.

## Current (2026-09-27): scalar-adjoint engineering gate stopped

Read `FASTGS_SCALAR_ADJOINT_RESULTS_20260927.md` and its frozen gate. Isolated native CUDA passed room@4100 four-view full-resolution RGB/loss and all-gradient parity, but two paired five-by-128-step block wall ratios were 0.962458 and 0.968246. Their best possible three-pair median is already above the required 0.95, so pair 3 was stopped under the frozen rule. This is a 3–4% fixed-block opportunity, not a complete-training speed or quality result and not an original method. Do not extend it to garden or full training on this evidence.

Root and Astra found no credible distinct quality-preserving, dominant-cost-reducing original mechanism to hand Luna after this gate. A proposed raster truncation/dead-zone repair also overlaps [Faster-GS Appendix C.4](https://arxiv.org/html/2602.09999v1) and [TruncGradGS §3](https://arxiv.org/html/2609.03534v1); it has no demonstrated net native-work saving. Keep the ambitious full goal open. Luna completed the already-prepared SAD-GS author room baseline gate above. Do not launch an original-method arm merely to maintain activity.

## Current (2026-09-27): SAD-GS bounded native CUDA smoke completed

See `SAD_GS_CUDA_SMOKE_20260927.md`. A room two-optimizer-step author-code smoke on EPIC-05 A100 GPU2 passed at the author's effective 1600×1066 loader resolution, with 272 training cameras, native renderer/backward/optimizer completion and saved iteration-2 PLY. The first 20 GiB allocator-cap attempt OOMed **before training** during 100-image structure-tensor precomputation; a single 60 GiB cap retry succeeded (peak allocated 29.77 GB decimal). Retry process wall was 57.839 s while the author's loop event sum was .693 s; the short-process ratio cannot predict 3K behavior. No short-run loss value or image metrics were printed, so neither finite-loss numerical parity nor quality was verified. Both attempts ended, GPU2 returned idle, author repo was restored clean, and old/new environments still import their native modules. This closes CUDA execution feasibility only. The next baseline gate is a frozen full author room run with process-wall wrapper and official held-out eval, followed by separately labeled original-loss common-protocol arms if a genuine new candidate emerges; a two-step smoke is not a method result. No new original mechanism was selected by root/Astra this turn.

## Current (2026-09-27): SAD-GS is a newly checked strong, source-available prior

Read `SAD_GS_PRIOR_ART_20260927.md`. SIGGRAPH 2026 SAD-GS already uses multiscale image structure/frequency versus projected Gaussian axes, multiview consistency, and anisotropic one-shot splitting to speed convergence; its author implementation is public at `LinjieLyu/SADGS@029814fb`. On the paper's H100 Table 1, it trains faster and improves SSIM/LPIPS over FastGS-Big in all three dataset aggregates, but loses PSNR by .71/.28/.81 dB and has far more Gaussians. It adds a nonzero L2 loss and reports summed per-iteration CUDA-event optimization time that excludes some setup and saving, not the full process wall demanded here. Add it to the eventual matched strong-baseline set, but do not relabel its core as original or run a generic frequency-split Luna arm. The author explicitly notes overlapping Gaussians can provide collective frequency support that its per-Gaussian decision ignores; that is a gap to investigate, not a validated new algorithm or reason to spend GPU before defining a distinct intervention. The full goal remains open.

Luna subsequently pinned the SAD-GS author checkout on EPIC-05 and built/imported its rasterizer, simple-knn and fused-ssim extensions in an isolated prefix; see `SAD_GS_BASELINE_PREP_20260927.md`. One pip command unintentionally removed `simple_knn` from the pre-existing Torch environment because it was visible on `PYTHONPATH`; Luna restored the exact cached wheel and reverified native import before continuing with clone-only installs. This establishes build feasibility, **not** native CUDA correctness, image/split parity, an author-result reproduction, or original-loss complete-wall comparison. No GPU training/rendering was run. Astra checked the collective-support opening and found that overlapping contributions have frequency-domain cross terms of either sign; overlap counts cannot establish avoidable splits. It has no distinct cheap intervention yet.

Root's proposed tangent-space orthogonal-residual birth criterion was rejected before another experiment: `TOPOLOGY_DESIGN.md` already defines the analogous regularized old-parameter projection and conditional finite-edit score; `REAL_TOPOLOGY_RESULTS.md` records room/garden@2K and room 512-step negative screens with substantial candidate-scoring cost. Calling a faster JVP the same mechanism would not fix its observed inability to predict lasting multiview benefit. SteepGS also separates mean shift from true split degrees of freedom. Do not send this relabeled candidate to Luna.

## Current (2026-09-27): ACE-GS ablations constrain claims, not a new experiment

The pinned ACE-GS v1 §5.1 states original image resolution and initialization-to-convergence wall time on one RTX 4090, but no verifiable author implementation or stage logs were found to align its protocol with the local A100 FastGS-Big runs. Its Table 2 is Bicycle only: full ACE is 5m44s / 25.47 / .753 / .243 (time / PSNR / SSIM / LPIPS); removing sparsification C yields higher SSIM (.757), removing densification E yields slightly better LPIPS (.242), and removing frequency compensation A is 8 s faster while worsening all three quality metrics. These are component tradeoffs, not evidence for a new method or a common-protocol FastGS-Big win. See `RECENT_TRAINING_FRONTIER_AUDIT.md` for the full source check. Astra proposed tracing whether newly born Gaussians incur substantial accepted work before later removal, but removal alone does not prove wasted work or a quality-preserving intervention. Luna's blockwise 30K gradient diagnostic also lacks a distinct causal intervention and cannot replace missing 20K Adam states. Neither proposal meets a GPU experiment gate; keep Luna available for a later frozen, mechanism-specific test. No new GPU arm was launched.

## Current (2026-09-27): room late phase still improves all three held-out metrics

The existing unchanged-author room trajectory in `FASTGS_LATE_PHASE_ATTRIBUTION_20260927.md` improves PSNR +.209511 dB, SSIM +.0011383, LPIPS −.0024882 from 20K to 30K on 39 already-developed retained views. This rejects simple 20K truncation for that run, not all earlier stopping schedules. Native backward cost share cannot say whether these gains came from geometry or appearance. Luna's read-only remote check found no 20K optimizer-complete room/garden checkpoint: three room 20K PLY-only files exist, garden has no 20K PLY. Root verified the author `--checkpoint_iterations` argument is inert for saving; model/Adam capture also omits RNG/camera-stack state. A matched-history selective-update diagnostic would need an explicit fuller capture during a rerun and would still not itself be a novel method. No GPU arm was launched or untouched confirmation scene consumed.

## Current (2026-09-27): shorter Gaussian lists already have a direct prior-art mechanism

Read `SHORTER_LIST_PRIOR_ART_20260927.md`. The ShorterSplatting paper already periodically shrinks Gaussian scales and penalizes alpha-weight entropy to shorten per-pixel Gaussian lists, reducing forward and backward work. It also uses a coarse-to-fine resolution scheduler and an extra loss, outside our original full-resolution step protocol. In its author Table 3, default FastGS 0.4M versus the proposed 0.4M setting goes 88.55→66.32 s but PSNR 27.43→26.50, SSIM .797→.754, LPIPS .261→.314. It does not beat FastGS-Big or meet the triple-quality goal. Do not rerun simple footprint shrinking or entropy polarization as a novel Luna arm. The missing link remains a distinct quality-repair mechanism that demonstrably reduces dominant native backward work under the original protocol.

## Current (2026-09-27): published compact-appearance timing omits enabled posttraining

The pinned Compact Neural Appearance implementation enables 5,000-step PPISP controller distillation but explicitly excludes posttraining from `Train` timing; see the source-level check in `SEPTEMBER_2026_FREQUENCY_APPEARANCE_PRIOR_ART.md`. Its 1.3× optimization claim is not complete process wall time. Five-run scene means show no strict PSNR/SSIM/LPIPS triple improvement in the nine Mip-NeRF 360 scenes against SH; one of four Tanks & Temples/Deep Blending scenes has a rounded triple improvement, without a significance claim. The authors also document non-static capture effects. This narrows, rather than supplies, a quality mechanism for us. LFH's pinned release supplies only 1K scripts and its room FastGS threshold differs from our pinned baseline. No new Luna GPU arm follows from either paper.

## Current (2026-09-27): polarized-opacity quality/cost chain is occupied

The additional full-paper check in `SEPTEMBER_2026_FREQUENCY_APPEARANCE_PRIOR_ART.md` found that the Polarized Opacity Prior already argues high opacity triggers earlier ray termination and less forward/backward work while an L2 loss plus blur densification repairs opacity-induced artifacts. Its author gsplat comparisons are faster but substantially worse in all three quality metrics, and replacing L1 violates our original-loss protocol. Do not rename this chain as a new mechanism or launch it with Luna. The remaining need is a distinct representation error and intervention with evidence for both original-loss quality and dominant native cost reduction.

## Current (2026-09-27): LFH and compact-appearance papers narrow the search

Read `SEPTEMBER_2026_FREQUENCY_APPEARANCE_PRIOR_ART.md`. LFH already trains a frequency hierarchy while archiving learned fields; its author FastGS-Big comparisons are faster but lose PSNR, SSIM and LPIPS in all three 1K aggregate datasets and the 4K Mip-NeRF 360 aggregate. Its staged resolution/loss is not the original full-resolution loss each step, and reported time excludes archive disk writes. Compact Neural Appearance already fuses a shared tiny-MLP appearance decoder into Faster-GS; its Mip-NeRF 360 SH comparison is faster and higher PSNR but lower SSIM and worse LPIPS. Faster-GS is not FastGS-Big, and neither paper establishes the required strong-baseline triple win. Author implementation links are recorded in the report. Root and Astra have no evidence-backed original causal intervention to test, so no Luna GPU arm is launched. The full goal remains open.

## Current (2026-09-27): direction-dependent opacity is not a new core

The pinned [Augmented Radiance Field](https://arxiv.org/html/2602.19916v1)
§3.1 already models Gaussian opacity with a trainable angular lobe. It uses
that for specular post-enhancement, not our complete-training speed/quality
protocol; nonetheless simply suppressing a Gaussian in unhelpful view
directions is not an original representation claim. See the appended source
check in `RECENT_TRAINING_FRONTIER_AUDIT.md`. Together with ACE-GS and RBS,
this rules out a renamed combination of sensitivity pruning, internal color
modes and view-opacity gating as a credible new core. No GPU arm was launched.

Astra and root found no intervention worth testing next. A potentially useful
missing observation is how much native work comes from *distinct Gaussian
sets redundantly representing the same physical surface across views* rather
than fine geometry, camera coverage or reflection. Current shared-ID hit
fractions cannot establish that correspondence or a causal quality/work
mechanism. Do not label this question an algorithm or rerun deletion, merging
or relocation without a specific different intervention and original-loss
benefit. The full goal remains active.

## Current (2026-09-27): expressive footprint color modes collide with prior art

Read `FOOTPRINT_MODES_PRIOR_ART_SCREEN.md` before proposing local
Gaussian/Hermite/affine appearance modes as the original quality-and-cost
mechanism. RBS already combines a Gaussian envelope with internal Hermite
RGB modes and explicitly argues that one expressive footprint can replace
overlapping flat splats and accelerate rasterization. Its evidence is for 2D
super-resolution, not 3D multiview training, but the proposed mathematical
core/rationale is occupied. Textured-GS/SVGS/Textured Gaussians/LiteTex-GS
and 2DGH impose further existing boundaries. No renderer, GPU screen, or
training arm was built for this rejected candidate. A richer splat has
higher per-hit cost; fewer primitives alone would not prove lower accepted
work or complete wall. The full original goal remains open.

## Current (2026-09-27): residual–native-hit overlap gate failed

Read `fastgs_residual_cost_overlap_20260927/REPORT.md` and its raw `result.json`.
At room/garden 4100 and 30K, four frozen full-resolution training views per
state gave pooled top-25%-L1-residual accepted-hit shares of 0.26065, 0.25659,
0.24943 and 0.25822, respectively. All miss the preregistered 0.40 gate.
The same pixels held 54.64%–63.91% of L1 residual, but roughly their area
fraction of native accepted-hit work. The 16 counter renders had exact RGB
parity and no failed view; the author source, PLYs, training-camera split,
protocol and probe identities were checked. Stop the **high-residual-
concentrated-work same-region** direction. These hit counts are not backward
milliseconds or an original-loss decomposition. The result does not reject
all cross-region quality/work mechanisms. There is still no credible original
candidate that meets the full-resolution, multi-scene, multi-seed complete
wall-time and formal unseen-view quality goal against strong baselines.
Do not interpret `f25≈0.25` as spatial disjointness or as evidence that a
cross-region transfer works. A fixed-state edit could affect a different
view/region through shared Gaussian parameters, but the existing measurement
has not established a beneficial intervention distinct from pruning,
split/relocation, or MCMC. Astra's direction reassessment found no candidate
worth a new GPU experiment on this evidence.

ACE-GS remains a required strong comparison, but a bounded source search
found no author implementation link in its paper/arXiv page or matching public
GitHub repository on 2026-09-27; read `ACE_GS_SOURCE_AVAILABILITY_20260927.md`.
This is an implementation-discovery limit, not a claim that no code exists.
Do not compare its author-reported RTX 4090 paper minutes directly with the
A100 FastGS-Big baseline or call an unvalidated rewrite a reproduction.

## Current (2026-09-27): native stage costs completed; next quality mechanism still open

Read `FASTGS_STAGE_PROFILE_RESULTS.md`, `FASTGS_DEFERRED_PULLBACK_RESULTS.md`
and `ASTRA_DIRECTION_REASSESSMENT_20260927.md` before new implementation.
Two unchanged author trajectories from initialization completed 30K on room
and garden; each has four 128-step Kineto windows. Only room retains valid
input provenance: its two late native raster backward shares are 52.7–52.9%
of active CUDA time, and optimizer shares are 1.3% and 0.6%. The old garden
rows were initialized from a corrupt sparse model and cannot support an
official-garden cost claim. Healthy P16 garden has a separate 20-step phase
profile (5.0056 ms raster backward / 8.6649 ms total); its timing denominator
differs from this Kineto series. Early room windows
contain scoring/topology events and cannot be called ordinary steps. The old
multi-stage `mechanism_discovery_v2` tables include gsplat and estimates, so
they are not a FastGS denominator. Exclude CUDA user annotations from kernel
sums; use native autograd operator attribution across backward threads. These
profile-affected runs are diagnostics, not uninstrumented speed baselines or
quality results. Raw data and final PLY hashes are retained in
`fastgs_stage_profile_20260927/` and the report.

Current-state covariance/opacity pullback reuse was tested via the author's
existing Python `cov3D_precomp` interface. On four full-resolution room@30K
training views, native replay RGB was identical, but the deferred implementation
had max RGB error 0.00319773 > the fixed 1e-4 gate. All six parameter-gradient
groups passed their gate; this does not rescue renderer parity. Stop this
implementation: no timing, garden, CUDA or continuation was run. The image
discrepancy's cause is not isolated. The chain-rule identity is not an original
quality mechanism; do not call this a failed timing result or disprove all
possible exact-native deferred implementations.

Two preceding negative screens must also remain excluded: high-cost pruning
(`fastgs_pruning_ab_20260927/A_REPORT.md`) qualified only 13 of 64 candidates
against the required 16, with just 0.4173% additive accepted-hit coverage;
near-color runs (`FASTGS_NEAR_COLOR_RUN_SCREEN.md`) put 93.16% of accepted hits
in runs of length <=2 and only 1.64% in runs >=8 at 4/255. Neither route gets
CUDA or a renamed retry. Covariance–opacity linked shrinkage was already
rejected as close to known regularizers without an independent quality lever;
the repeated Astra suggestion was withdrawn before any experiment.

User-directed working split: Luna executes bounded experiments and organizes
evidence; root and Astra design and screen directions. Root verifies returned
results. This is a workflow preference, not a guaranteed token saving.
All experiments launched in this turn have ended; GPU 2/3 were empty at the
last check, but recheck before any new run. Astra encountered service rate
limits; its written reassessment is retained, not a new validated method.

No new original candidate is selected. More timing subdivision alone will not
establish the missing quality mechanism. A different **unmeasured** diagnostic
question is whether substantial original-loss residual lies outside native
densification evidence (no accepted contributors or no gradient-qualified
parents). This must distinguish visibility, data noise and view dependence
from true missing support, and cannot establish useful births or novelty;
residual-guided birth/relocation already has MCMC prior art. Only execute a
bounded, preregistered quality-attribution test if it can inform a materially
different mechanism. Do not create an arbitrary regularizer merely to keep
experiments running. The full-resolution multi-scene/multi-seed complete-wall
and formal unseen-view strong-baseline goal remains active and unmet.

## Current (2026-09-27): 75% full-resolution sparse adjoint pilot

Read `fastgs_sparse_adjoint_pilot_20260927/REPORT.md` and its `summary.json` before reviving pixel-sampled backward. A matched author FastGS room 4099→10K pilot used the full 1600×1066 image and original loss, changing only a tile-stratified 75% image VJP mask. The corrected `--eval` pair trained on 272 cameras and evaluated 39 already-developed retain views. Mean p75−p1 changes were +0.112961 dB PSNR, +0.000132 SSIM and +0.000454 LPIPS (worse); all descriptive paired intervals cross zero. One view supplied +6.26484 dB PSNR, larger than the total net PSNR gain; the other 38 views averaged −0.04893 dB. Python masking skipped no CUDA backward work, and full process times 126.81/127.20 s (p75/p1) do not establish algorithmic speed. The first pair omitted `--eval`, leaked the 39 views into training, and was explicitly invalidated; do not cite its metrics. Do not promote this candidate to a production sparse CUDA implementation on current evidence. The joint full-training speed and untouched-scene quality goal remains open.

## Current (2026-09-27): recent strong frontier and naive group-motion screen

Read `RECENT_TRAINING_FRONTIER_AUDIT.md`. ACE-GS already combines
momentum-gradient consistency densification, opacity-gradient sensitivity
pruning, and FFT residual compensation, with strong author-reported
quality/time. Vol3DGS already integrates per-ray Gaussian transmittance and
reports better SSIM/LPIPS but longer training than its 3DGS control. 3DGART
already uses primitive-centric tile backward for ray tracing; GRay exploits
dense tiny Gaussians but reports somewhat lower quality than raster 3DGS.
These are bounded full-text checks, not source reproductions or directly
comparable time baselines. The first distinct group-motion hypothesis also
received a real full-resolution room gradient screen: simple 16/32/64 voxel
groups have median cross-view group-gradient cosine near zero and at most
1.27% of *eligible* gradient norm in groups with coherence >=0.7 and positive
mean cross-view cosine (64 bins/axis). Do not write a production group optimizer
from that signal; more sophisticated grouping is untested. The actual joint
goal remains open and requires a new mechanism with material full-image and
full-cost opportunity before any 30K campaign.
The next bounded opening now has a source and numerical screen; read
`VOLUMETRIC_FACTOR_SCREEN.md`. Pinned Vol3DGS Slang uses per-pixel ray maximum,
3D inverse-covariance quadratic, sqrt, and two exponentials. A tempting global
`c-b²/a` simplification is numerically catastrophic in FP32 on a 20K-case
synthetic screen with normalized rays and the pinned alpha constants (0.888 p99
alpha error). A center-ray residual identity is stable at roughly the pinned
residual implementation's accuracy (5.80e-5 versus 5.96e-5 p99 alpha error), but has **no
GPU speed or real reconstruction evidence**. Only an actual parity-checked
CUDA/Slang forward/backward and real full-image cost/quality screen could
promote this narrower implementation opportunity. The elementary identity is
not an original method; keep matched ACE-GS/other strong-system validation as
the final requirement.

2026-09-27 follow-up supersedes that implementation opening: an exact
projective-screen quadratic reformulation passed synthetic FP64 identity and
had FP32 alpha accuracy comparable to the pinned residual computation, but
its symmetric-parameter FP32 center-gradient discrepancy reached 0.00289 p99
relative norm and 0.0622 maximum. An isolated RTX 5070 Laptop flat forward
kernel showed **no stable paired CUDA-graph speed win** at 16/64/256 pixels per
Gaussian; paired median ratios crossed 1 across runs, before B precompute or
any rasterizer/backward cost. Read the appended section of
`VOLUMETRIC_FACTOR_SCREEN.md` and its three scripts/JSONs. Do not write the
production Vol3DGS shader or launch 30K on this evidence. The full-resolution
joint quality/time goal remains open; the next candidate must first show a
material quality mechanism and an integrated GPU cost opportunity on real
scenes, with original strong-baseline protocol intact.

2026-09-27 FastGS real-backward opportunity screen: read
`FASTGS_BACKWARD_BUCKET_OPPORTUNITY.md`. The archived 44-step room profile puts
only 2.09% of self CUDA time in CUB sorting but 43.13% in raster backward.
On four real full-resolution room@4100 views, a new diagnostic exactly matched
an independent accepted-interaction count. Within existing 32-Gaussian
backward buckets, only 0.33%-3.04% of eligible bucket-pixel pairs contain no
accepted Gaussian; whole-bucket-pixel skipping is not a material lever.
Changing group width 32->4 on one view raises empty eligible pairs to 20.05%
but raises an equivalent FP32 T+RGB boundary-state layout from 118.3 to
855.9 MiB. These are work/memory proxies, not timings or a new algorithm.
Do not implement fixed-width small buckets on this evidence. A future
single-primitive sparse adjoint would need a different exact state layout,
native forward/VJP and measured integrated savings; it still needs a separate
reproducible quality mechanism and untouched-scene strong-baseline validation.

## Current (2026-09-27): per-ray/local depth sorting has direct prior art

Read `DEPTH_ORDER_PRIOR_ART_SCREEN.md`. StopThePop already derives the per-ray
maximum-contribution depth of a 3D Gaussian and performs tile/per-ray local
resorting; 2DGS uses perspective-correct ray–planar-splat intersection; TileGS
uses tile-local depth bins with selective baseline-order repair. Thus a generic
per-pixel depth/local sort is not an original main line. RoofGS is a strong 4K
rendering reference, but its author-reported 10.1x frame speedup and -0.028 dB
PSNR are not complete training results. A narrower color/transmittance-aware
conflict-repair hypothesis has an exact adjacent-swap RGB identity; its first
bounded real-scene pixel screen is summarized immediately below.
Do not promote the identity or a keyword-search miss to originality or SOTA.
The first actual screen is now complete: `fastgs_order_conflict_result.json` sampled
256 pixels on each of four room@4100 full-resolution training cameras. Among
425 parity-matched rays without native early stopping, direct per-ray-depth
resorting worsened sampled GT L1 by +0.0005836 on average, with 179/425 pixels
improving; 599/1024 sampled rays stopped early and were excluded. Risk is
concentrated (top 10% adjacent inverted pairs contribute 78.7%-82.4% of
summed pair-risk), but this does not predict GT improvement. Retire **direct
ray-depth resorting** as the next training implementation. This is a bounded
room pixel diagnostic, not proof against every ordering method or a whole-image
loss result. Any reopening requires a different mechanism and full-image
evidence that covers early stopping before CUDA/30K work.

## Current (2026-09-27): first real capacity exchange fails room opportunity gate

Read `FASTGS_CAPACITY_EXCHANGE_SCREEN.md`. A native split child was re-fitted
after seven nearby deletion proposals, selected using six full-resolution
training views. The fit-selected pair changed the original full-image loss
by only -5.29e-7 versus unedited, barely beyond the birth-only -5.22e-7;
on four internal-check training views exchange was marginally worse than
birth alone. Across ten cameras, author rasterizer Gaussian-tile instances
fell by only 23 out of 7,615,103 (0.000302%). This is not GPU time, but it
leaves no large structural speed opportunity for this pair. Stop the native-
child/nearby-delete version before CUDA or continuation. A garden author-
protocol 4100 checkpoint was generated for potential future development
(4,111,216 Gaussians; hash in the screen doc) but **no garden exchange was
tested**. A later full-text check found direct prior art in 3DGS-MCMC and
Metropolis-Hastings Sampling for residual-guided Gaussian birth/relocation.
The broad capacity-exchange framing is therefore retired as an original
strong-method main line. The ambitious user goal remains open; do not simply
change geometry and relabel this idea as a breakthrough.

## Current (2026-09-27): fixed-child birth fit is locally real but globally tiny

Read `FASTGS_BIRTH_FIT_RESULTS.md`. A visibility-eligible single native child
chosen by a crude center-residual score nearly vanished under analytic
opacity/RGB fitting. Screening all 24 multi-view-eligible children with only
six fit training views found one with 19.01% sampled SSE reduction; one
internal-check training view had 17.41% sampled reduction. Author CUDA versus
the fixed-support formula agreed within 3.67e-7 on sampled pixels.

The same child, inserted at the fitted opacity and constant RGB into an
isolated FastGS model, reduced original full-image L1+fused-SSIM loss by only
5.22e-7 averaged over six fit views and 1.21e-7 averaged over four internal
checks. Mean PSNR gains were below 0.0001 dB. A phase-timed repeat spent
4.365 s on the 24-candidate six-view CPU fit alone (plus 54.48 s scene
loading), and no integrated trainer or continuation exists. This tests one
born primitive at one developed room state; it neither
rules out a joint many-child mechanism nor supports one. Any next birth route
must show joint full-image benefit per *measured future GPU cost*, avoid a
separate CPU candidate sweep, and pass matched continuation before 30K work.
Cross-view residual selection alone is already prior art.

Jointly inserting all 14 fit-positive candidates with fitted opacity/RGB
did not rescue this fixed-state opportunity: author full-image original loss
changed by -7.13e-7 over fit views and +1.91e-7 (worse) over the four internal
checks. One check view lost 2.01e-6 in original loss. See the joint section of
`FASTGS_BIRTH_FIT_RESULTS.md`. Retire this native-child CPU scan and do not
reinterpret local MSE percentage as reconstruction quality. The next route
needs a material whole-image effect and a measured integrated cost path.

`OCCLUSION_AWARE_CAPACITY_EXCHANGE_DESIGN.md` records a now-retired broad
route: delete a costly redundant primitive, fit a 3D residual-based birth in
the **post-deletion** transmittance/background, and commit only if native
full-image quality and measured future pixel work jointly improve. A
10,000-case FP64 oracle checks the ideal single-exchange formula to 3.41e-16
and shows independent overlapping-birth responses can err by 0.441 RGB;
joint native rerender validation is mandatory. There is no real exchange
experiment, GPU implementation, complete-training gain, or originality
certification. Its first real room gate failed, and direct relocation prior
art closes the broad originality position. Do not repeat it on garden without
a substantively different mechanism and a new opportunity case.

## Current (2026-09-27): density-H4 split proxy fails its alpha semantics gate

Read `MOMENT_FOURTH_SPLIT_SCREEN.md`. A moment-compensated *additive density*
split has a leading H4 response, but the actual opacity compositor changes
the order: even a zero-shift-exact, adjacent equal-color child kernel yields
an O(h²) positive opacity term, verified by a FP64 1-D oracle. Ordinary
half-opacity children also have a nonzero error at zero shift. Thus a cheap
H4 density tangent is not the leading derivative of a legal alpha split.
Even with matched total parent/child transmittance, an interleaved Gaussian
changes RGB by `(1-a1)*a2*b*(d-c)` for parent-before-B versus child1-B-child2;
1,000 FP64 random cases confirmed the identity within 2.78e-16.
At .5 sigma the density finite-edit approximation already has 31.07% L2
error; only 47.81% of H4 response norm remains after removing old tangent
directions under image L2. SteepGS already computes split curvature and
2DGH already trains H4 kernels. Retire this H4 proxy as a strong GPU/30K
candidate. Reopen only if a physically sorted alpha-aware derivative has a
credible cost and cross-view long-term opportunity. The joint goal remains
open.

## Current (2026-09-27): vector split gate quality screen is weak

Read `FASTGS_VECTOR_GATE_RESULTS.md`. A full-resolution, frozen-state
9,345-family gate diagnostic matched native pre/post RGB endpoints within
2.384e-7 and one selected FP64 pixel-family VJP within 9.14e-9. Twelve
gate-only updates on ten train views lowered their mean loss to .0327590
versus .0329433 unsplit and .0344714 native all-split. But on 39 already
consulted room development views, the continuous gate was slightly worse
than unsplit (PSNR −.002344 dB, SSIM −.000165, LPIPS +.000470), and the
1,922-family binary commit was weaker. No model continuation or net wall
benefit has been measured. Deprioritize native fused-gate implementation
until there is a compelling long-term quality mechanism; if investigated,
first use a matched full-resolution short continuation with all offline gate
costs counted. A binary-mask continuation does not test continuously learned
gates. The joint goal remains open and requires untouched scenes/seeds.

## Current (2026-09-27): bounded fused-gate cost opportunity, quality still unproven

Read `FASTGS_FUSED_GATE_OPPORTUNITY.md`. On four real full-resolution room
training views at the native 9,345-family split event, the half-open shadow
support increases scalar visited pixel–tile-list rows by only 0.99–2.43%
versus native post-split. Endpoint nonzero-alpha work counts match the
native pre/post states exactly. A same-buffer kernel toggle found that family
atomic accumulation itself adds 0.018–0.107 ms, whereas recomputing a whole
VJP traversal adds roughly 0.96–2.15 ms beyond RGB-only in the isolated
diagnostic. This supports investigating **fusion into the existing native
backward traversal** before retiring the continuous-gate idea on cost grounds.
It is not a prediction of fused wall time. The one-shot veto and child-choice
pilots remain weak on quality. Next implementation gate: preserve ungated
candidate support in native forward, compute per-family gate gradients in
native backward, verify physical-depth endpoint/gradient parity and bounded
probation optimizer-state lifecycle, then measure complete per-step and
full-training wall including all gate work. Stop if actual cost is not repaid
or quality does not improve on untouched scenes/seeds. Do not claim originality
without source-level prior-work comparison.

## Current (2026-09-27): two-proposal child placement is also a weak pilot

Read `FASTGS_CHILD_PROPOSAL_CHOICE_RESULTS.md`. At the same full-resolution
room@4100 event, the frozen ten-train-view VJP rule chose one of two entire
child-position proposals for each of all 9,345 native split parents. The
matched 512-step branch had equal surviving Gaussian count and improved the
39 development-view means by only +.013601 dB PSNR, +.000054 SSIM and
−.000069 LPIPS; paired-view intervals for all three cross zero. Relative to
the prior seed-27 control the PSNR gain shrinks to +.006384 dB, and relative
to the prior seed-28 control LPIPS worsens. At least 323.15 ms of measured
scoring components is excluded from the branch wall. No net speedup or robust
quality gain is demonstrated. Retire this one-shot child-choice policy as a
strong candidate; do not launch 30K work on it without a new cost/quality
mechanism. The original joint full-resolution goal remains open. Future
candidate must be independently testable on untouched scenes/seeds with all
construction/scoring work in complete process wall and latest strong methods
under the same image-resolution and metric protocol.

## Current (2026-09-27): all-family VJP passes, one-shot veto has only a weak room signal

Read `FASTGS_EVENT_GATE_RESULTS.md`. One real 9,345-family room@4100 event
now has an isolated full-resolution CUDA fixed-support VJP: native pre/post
endpoint RGB within 2.384e-7, three pixel-family VJPs within 1.50e-14 of
FP64. The separate all-family image pass costs 3.686–4.411 ms and candidate
tiles cover 64.6% of image pixel slots, so a distinct pass every probation
step is a poor speed path without a fused, measured alternative. Ten fixed
training views have 4,926/7,079 meaningfully supported families with mixed
gradient signs; one-view decisions are unsafe.

A frozen exploratory policy vetoed 1,831 bootstrap-stable positive-gradient
splits in one matched room event. With identical full child proposals and no
later densification, 64/512-step development-view mean PSNR/SSIM/LPIPS moved
slightly favorably, but 512-step PSNR improved in only 18/39 views and the
paired-view bootstrap interval crosses zero. Complete 512-step wall was
68.59/67.85 s (control/veto), with scoring cost excluded; the veto training
timer was slower. Thus neither faster training nor robust quality is shown.
Next: freeze the policy, test another child seed and a distinct scene with
the entire on-event score cost, then retire it if the quality/time signal
fails. Do not promote a single room pilot to originality or SOTA.

Two-seed follow-up: with child seed 20260928 the frozen rule vetoed 1,855
families. At 64 steps PSNR/SSIM/LPIPS means again moved favorably; at 512
steps gains shrank to +.00347 dB/+0.000064/−0.000054, all descriptive
paired-view bootstrap intervals crossed zero. Its 512-step process wall was
**slower** (70.83 vs 64.58 s), with screening still excluded; the earlier
seed showed the opposite wall ordering. Candidate geometry strongly changes
scores (3,783/8,022 meaningful mean signs flipped across two child seeds).
See the second-seed section in `FASTGS_EVENT_GATE_RESULTS.md`. Deprioritize
the one-shot veto as a strong candidate. The main goal still requires a
larger, reproducible, cost-accounted full-resolution mechanism; do not burn
30K runs on this weak policy without a changed opportunity case.

## Current (2026-09-27): full-resolution gate has real CUDA endpoint parity, not training benefit

Read `FASTGS_SHADOW_GATE_CUDA_RESULTS.md`. At room@4100, the 4099 checkpoint
and pre-event PLY match exactly across all six parameter tensors. A one-family
isolated CUDA fixed-support diagnostic on author projected/sorted buffers
reproduces native `g=0/1` 1600×1066 RGB within 1.79e-7, zero pixels above
2e-5, and three selected-pixel gate VJPs against FP64 within 6.09e-15. Its
separate full-image pass costs ~1.95 ms, excluding construction and training.
The complete 9,345-parent event's shadow union has 2.43% more Gaussian–tile
instances than native post-split, but this count is not a pixel-work or wall
speed measurement. No quality, complete training speed, or originality gain
has been shown. Next implement and cost all-family gradients/rollback; reject
the method if probation overhead is not repaid, then run matched-state
full-resolution 64/512-step trials before any 30K or SOTA claim.

The half-resolution main-training + occasional full-resolution topology scout
remains retired for this user's joint objective: sparse scouts do not restore
full-resolution photometric gradients to ordinary steps. The historical
12-pair arm is a speed–quality tradeoff and also changed optimizer, so it does
not isolate resolution's causal effect.

## Current (2026-09-27): topology homotopy is an unimplemented hypothesis

Native FastGS room@4100 split-mask evidence now confirms a positive
zero-start dead zone: among 9,345 selected split parents, center gate onset
has median .00257 and 860 remain below the alpha cutoff at g=.01. The native
tile list also uses gated opacity, so merely removing the forward pixel cutoff
would not recover the candidate. The scalar oracle gradient is for a smooth
fixed-support extension, **not** the derivative of the native hard-cut
renderer. See `FASTGS_GATE_DEADZONE_RESULTS.md` and the two probe scripts.
Next must establish an explicit shadow-support renderer/VJP and measure its
real pixel work before this can be a credible faster-and-better candidate.
An independent 2,500-case FP64 fixed-support compositor now checks the
event-only shadow VJP formula, including 271 exclusive early-stop cases and
1,682 arbitrary child-after-parent orders (max VJP error 3.55e-15). This is
not GPU parity; see `topology_shadow_vjp_oracle.json`. Implement the small
real-region CUDA reference and cost gate next, then stop if it fails.

Read `TOPOLOGY_HOMOTOPY_DESIGN.md` and `topology_homotopy_oracle.json`.
An optical-depth gate connects the native pre-split and post-split render
endpoints in a scalar alpha compositor even with unrelated splats interleaved
in depth. Ten thousand FP64 random checks passed at ~2.22e-16 maximum image
error. A hard-cutoff counterexample reverses the zero-gate loss-gradient sign
when transparent children are culled; a real renderer must retain their
candidate support for gate differentiation. A rerun retaining children by
their ungated alpha restored the scalar gate gradient to -0.04367347; this
does not establish correct FastGS candidate support or affordable GPU work.
No FastGS GPU parity, savings,
quality gain or originality is shown. An abstract/full-text screen also
rejected simple spatially varying Gaussian color as novel because Textured-GS,
SVGS, Textured Gaussians and LiteTex-GS already cover it. Next gate is a
small-region full-resolution FastGS renderer/VJP reference plus actual
temporary-pixel-work cost; stop this hypothesis if either fails.

## Current (2026-09-27): full-resolution FastGS input-pipeline trial

Read `PARALLEL_CAMERA_LOAD_RESULTS.md`. The isolated 16-thread CPU image
preparation changes only one camera-loading source file and passes bitwise
`torch.equal` for all 311 serial/parallel image tensors. The complete author
FastGS-Big room 30K wall fell from 287.27 s to 248.50 s in a separate run;
the trial scored 32.22626/.930709/.188195 on the same 39 GT images versus
the author run's 32.20546/.930243/.188115. The LPIPS mean is slightly worse,
and stochastic training trajectories are not bitwise identical. In this one
room run the trial also beat Faster-GS `main` by 2.81 s and all three image
metrics, but that small wall gap is not a robust cross-system result.

This is a useful exact-input systems component, not a novel 3DGS algorithm or
a validated reconstruction-quality mechanism. Keep the user goal open. Next,
replicate on untouched scenes/seeds and pin identical train-image resize for
close cross-system comparisons. In parallel, identify a distinct, affordable,
full-resolution quality-positive mechanism that survives current prior art;
do not relabel parallel data loading as the pioneering contribution.

## Current (2026-09-27): author FasterGSFused room baseline also fails the joint gate

The author's pinned `FasterGSFused` branch completed one full-resolution 30K
room run in 252.98 s complete process wall, with 1,179,916 Gaussians. On the
same 39 FastGS PNG targets and metric implementation, it scored
31.70539 PSNR / .925952 SSIM / .200744 LPIPS. The author `main` arm took
251.31 s and scored 31.72786 / .926164 / .200405; FastGS-Big took 287.27 s
and scored 32.20546 / .930243 / .188115. Thus fused was slightly slower and
slightly worse than its own `main` in this one run, while both Faster-GS arms
were faster but lower-quality than FastGS-Big. Differences between Faster-GS
branches are too small to treat as stable without repeats. Full protocol and
raw evidence: `FASTERGS_BASELINE_PROTOCOL.md` and `fastgs_evidence/`.

The rejected half-resolution scout changes the gradient in nearly all training
steps and has no demonstrated joint benefit. The current 12 paired historical
arms show a mean 1.205x speedup with lower mean PSNR/SSIM and worse LPIPS;
the arm also changed optimizer, so the loss cannot be uniquely attributed to
resolution. Retain it only as a speed-quality tradeoff study. Next research
must first show a measured full-resolution compute opportunity in a strong
baseline and a separate reproducible quality mechanism, then confirm complete
wall time and unseen-view quality on untouched scenes and multiple seeds.
No project candidate is yet validated across the required scope or supports a
SOTA claim.

## Current (2026-09-27): visibility-lazy Adam does not clear the joint-objective cost gate

The source-derived screen-envelope probe recovered zero positive-radius
misses across all 272 room training cameras at the real 4,099 checkpoint
after eight hypothetical zero-gradient Adam updates, but admitted 31.6598%
of Gaussians on average. This is one state/gap and not a correctness proof.
`fastgs_evidence/room_big_4099_screen_envelope_all_train.log` and
`fastgs_screen_envelope_probe.py` contain the executed evidence.

More decisively, the FastGS optimizer schedule has only 11,292 Adam calls in
the 4,099->35K continuation. Extrapolating the 44-step early profiler's
1.651 ms/call gives 18.64 s for **all** Adam work, or 12.74 s if the
68.34% envelope-negative share were erased at zero cost. Both are below the
observed 29.06 s complete-wall gap between 35K and 30K references; exact
debt catch-up, gate and flush would consume more. This is an opportunity
estimate, not a measured end-to-end bound, but it does not justify writing
GPU lazy Adam to buy extra 35K full-resolution optimization. It also offers
no quality mechanism by itself. Stop this implementation path for the
present joint objective; seek a full-resolution quality-positive mechanism
with a larger measured speed lever and compare it to current strong methods.

All sections below are prior stages retained for provenance.

## Current (2026-09-27): exact visibility-lazy optimizer opportunity gate

The first real-momentum geometry gate **failed**. At room@4,099 after eight
hypothetical zero-gradient Adam steps, 401 Gaussian-camera cases acquired
positive raster radius; the stale-center sphere with per-row drift and
2*3-sigma scale still missed three positive-radius cases in 12 cameras.
One of those three was positive even at the original state. The FastGS CUDA
predicate uses screen-space covariance/opacity tile assignment, not this
world-sphere approximation. A 4*3-sigma sphere missed zero in this sample
but admitted 39.206% of rows and is not a proof. The debug log is in
`fastgs_evidence/room_big_4099_momentum_visibility_debug.log`. Replace this
proxy with a conservative gate derived from the actual renderer semantics,
then remeasure its full cost; do not promote lazy GPU Adam meanwhile.

Read VISIBILITY_LAZY_ADAM_OPPORTUNITY.md. The previous cap32 rule is retired.
A 44-step FastGS room profiler assigns about 22.27% of ordinary early-step
self CUDA time to Adam; 40 training-camera renders project about 24.86%
of room@10K Gaussians per view. A 12-camera loose-sphere pilot found no
false negatives against positive renderer radius, but that is not a proof
under optimizer-driven motion. CPU FP64 zero-gradient Adam catch-up matches
PyTorch Adam to 4.44e-16 max parameter error, and its scalar displacement
bound passed 10,000 random cases. The author Faster-GS already fuses
backward/Adam; its reported skip-invisible variant saves substantial time
but loses 0.14 dB PSNR on five outdoor scenes. Generic fusion/sparse Adam is
prior art. Exact deferred settlement and conservative visibility remain
untested on GPU and their originality is unconfirmed.

An unmodified 4,099 -> 35K room extension raises held-out development-view
PSNR by 0.03616 dB, SSIM by .000217 and lowers LPIPS by .000412 versus its
own 30K snapshot, but costs an additional 29.06 s full wall versus a
separate 30K run. A lazy 35K system needs to save >9.97% of its own full
wall merely to be faster than that 30K reference, before attempting other
strong methods. One scene/trajectory cannot establish a reproducible
quality gain. Keep this as an opportunity test, not the new claimed method.

Next: validate conservative frustum expansion against real momentum-driven
state changes with zero false negatives; measure possible-visible fraction,
catch-up arithmetic and flush cost on a full early trajectory. If the
realistic end-to-end savings ceiling is insufficient, stop this implementation
path. If it passes, build an isolated GPU variant and test parity through
topology and full training. A separate reproducible quality improvement or
extra full-resolution steps within a shorter wall budget is required for the
joint objective. Confirm on untouched scenes/seeds against FastGS-Big,
Faster-GS and ACE-GS where official code/protocols permit; do not infer SOTA
from current room data.

All sections below are historical stage records retained for provenance.

## Earlier (2026-09-27): cap32 complete 30K trial failed the joint gate

Read FASTGS_BARRIER_AUDIT.md, especially its paired 4,099 -> 30K verdict.
The cap32 rule kept the author renderer/loss and fixed 1600x1066 effective
resolution. On the same room checkpoint, its built-in training time was
197.9282 s versus 192.0766 s for the author rule; full process wall was
267.66 s versus 262.40 s. At 30K its 39-view PSNR was 32.220028 versus
32.229950 dB, SSIM .9310054 versus .9302623, and LPIPS .1871286 versus
.1887598. The 10K and 20K snapshots also lost aggregate PSNR. The earlier
standalone 10K pilot's PSNR improvement did not replicate in this full
trajectory. Retire this fixed per-view cap as a strong joint candidate;
SSIM/LPIPS gains alone do not compensate for longer training or establish
overall reconstruction superiority. Room held-out views have been used for
development and cannot be an untouched confirmation set for this rule.

The half-resolution main-training plus occasional full-resolution scout idea
also remains outside the user's target: sparse scouts do not supply the
full-resolution loss and gradients at every optimization step. Its quality
recovery is unproven, and the earlier multi-scene half-resolution arm had a
material quality deficit. Keep future candidate training at a matched input
resolution and count complete wall time. First find a measured full-resolution
time saving against FastGS-Big or a quality-positive training mechanism,
then test their combination on untouched scenes, multiple seeds and an
up-to-date strong-method/prior-art comparison. No candidate meets the
faster-and-higher-quality objective yet.

All sections below are historical stage records retained for provenance.

## Earlier (2026-09-27): first matched FastGS-Big barrier established

Read FASTGS_BARRIER_AUDIT.md. Isolated author FastGS @44e02a5 builds with
PyTorch 2.9.1+cu128 on EPIC-05. One complete room-Big 30K run on A100-SXM4
reported 231.798 s built-in training time, 572,546 Gaussians, and 39 official
held-out images at 1600x1066 with PSNR 32.042473, SSIM .9298893,
LPIPS .1887548. The first LPIPS attempt lacked the calibration weight; the
second unchanged author metrics run used the installed lpips v0.1 VGG weight
and succeeded. Local raw metrics, config, logs, image hashes and source
references are in fastgs_evidence/. This is one scene/seed and different
software from the paper; it does not establish system comparison or novelty.

FastGS's ten-camera score uses two extra renders per camera and selects by
mean high-error *pixel-hit count*, not a distinct-view vote. Two early score
events were only ~1.88% of an instrumented 700-step training timer, so
amortizing those score renders alone is not a credible strong speed lever.
At a saved post-event room@600 state, 10,348 of 63,293 metric-gate passers
projected in >=3 sample views had effective error-view count <1.5, but this
does not include live gradient qualifiers or establish a quality defect.

The exact 45-event live diagnostic through 5K found 22.88% of eligible
selections projected into >=3 cameras had effective error-view count <1.5.
One fixed `sum_v min(c_v,32)` pilot from a common optimizer-complete 4099
checkpoint replaced 30.16% of parents while matching clone/split counts at
all 60 events through 10K. Room's 39 held-out development views at 10K moved
PSNR +.084695, SSIM +.000526 and LPIPS -.000859 with near-identical training
timer; one no-edit replay moved -.008667 dB/-.0000984/+.0004272 versus the
first control and had a nonidentical PLY. These are weak early signals, not
final quality or a speed advantage. Room test views are now development data.

Next: finish matched 4099->30K baseline and cap32 branches with 10K/20K/30K
model saves, compare final unseen-view PSNR/SSIM/LPIPS, per-scene full wall
including load/save, point counts and quality-time curves. If the final gain
survives, test other scenes and seeds with no tuning on their held-out views,
then perform full-text/source prior-art audit and compare other strong
methods. If final quality or time fails, close this simple scoring rule.
No current candidate has met the user's joint speed-and-quality objective.

All sections below are earlier stage records, retained for provenance.

## Current (2026-09-27): fixed-depth pair movement gate failed

Read SURVIVAL_MOTION_RESULTS.md and survival_motion_summary.json. Same 256
room@2000 parents, six full-resolution training cameras, symmetric 0/.1/.3/.45
long-axis sigma moves: union tile coverage is complete, but at .1 sigma
67.6%-94.5% of tiles containing both children have another Gaussian between
them in physical depth order. A grouped primitive is a different occlusion
model, not a faithful pair of independent sorted children. A-group mean loss
stays close to unsplit; B-group mean worsens for all positive displacements.
The independent physical B mean is better mainly because of one view. This
cannot establish final quality on official unseen views or a speed gain.

First run had an invisible-child opacity leak with huge RGB artifacts; its
failed output/source remain versioned. The corrected six-view outputs and
source hashes pass analyze_survival_motion.py. Do not launch a 64/512-step
campaign for the present fixed-parent-depth group configuration. A renewed
proposal needs coherent moved-child ordering/exit, a cross-view opportunity
above replay and a credible complete-cost path. Keep the original ambition:
same-protocol strong-method competition on time and official unseen-view
quality, with a source-level originality audit. No candidate has met it.

All sections below are older stage records, retained for provenance.

## Current (2026-09-27): CUDA birth parity passed; training lifecycle unverified

Read SURVIVAL_PAIR_RESULTS.md. An isolated CUDA group compositor now passes
independent-child VJP and three 1920x1080 room@2000 birth audits, including all
260,530 parents paired. Native independent children fail alpha cutoff,
clamping and mid-pair early-exit examples. The first grouped expression also
failed the full-image peak gate; its stable replacement passed the original
gate. Preserve both results. Recheck with analyze_survival_pair.py.

The current prototype retains parent projection coverage/depth order. Next
implement and audit union coverage, group ordering/occlusion semantics, and
transient exit before any local training continuation. Do not silently treat
the fixed-list birth prototype as supporting arbitrary child movement.
Then compare identical parents/child counts and preserved Adam history with
native and published conservation controls at 64/512 steps; count all costs.
Revising Densification (2404.06109) Sec.3.3 is direct prior art for the square-root
clone-opacity correction and its rationalization. Neither formula is ours.
No complete-training advantage or originality is established. All probe
processes have exited; the final GPU 6 check was empty.

The paragraphs below describe the earlier gate, now superseded by this update.

Read TRANSMITTANCE_SPLIT_DESIGN.md and PRUNING_DELETION_RESULTS.md. The
80-candidate real P11 room deletion score predicts its own A views but not the
three held-back B views (Spearman first-order/B=-0.054), so do not promote the
quality-aware pruning sketch or write its production CUDA. No-edit replay was
bitwise identical; the apparent B batch gain is dominated by one Gaussian.

The new candidate is a temporary non-Gaussian opacity kernel for two newborn
splats: h(a)=1-sqrt(1-a), which preserves parent alpha-composited image and the
shared opacity derivative at co-location when children are adjacent and share
color. A 10,000-trial FP64 oracle passed. A separate 10,000-pixel PyTorch
reference also passed finite FP32 forward/VJP checks through logit 40; naive
FP32 sqrt(1-sigmoid(20)) backpropagates NaN. This is not production GPU parity.
Real split candidates at room@2000
have median opacity 0.861, but this does not prove that actual native split
error comes from opacity or that zero-shock births improve convergence.

Next gate: implement only an isolated reference forward/VJP and measure fixed
event renderer parity, saturation behavior and full kernel cost. Compare
same-parent native and published conservation controls before any full run.
FastGS-Big is the relevant quality-and-speed comparator; our historical
1.164x gsplat arm is insufficient to establish competition with it. A full
source-level novelty audit is mandatory; do not call the algebra original.
No complete-training advantage or originality is established.

All older queues below are historical and superseded by this section.

## Current (2026-09-27): full-resolution speed and quality gate

Read FULL_RESOLUTION_PARETO_DIRECTION.md. The user rejected half-resolution
training plus sparse full-resolution scouts as a solution to simultaneous
training speed and reconstruction-quality improvement. Keep every optimization
step at full resolution and the original loss in the active direction.

The 132-job full-resolution fused/prune10/rclip05 arm has 1.164x mean speedup
versus gsplat and passes non-inferiority, but paired mean LPIPS is +0.001412
(worse). Visible-only and 25K controls fail their gates. Local per-scene raw
records and the dedicated frozen accel15 protocol were not found; do not infer
which component or scene causes the LPIPS loss. Paper source hashes can be
reconstructed with `core.abbrev=8`, but dataset/environment identity is not
matched and no full training was run in this reconstruction.

First recover matched protocol/results. Then isolate fused, prune and radius
clip in paired full-resolution training; only if a material quality loss source
is found, test a budgeted repair against equally strong controls. Count the
subsequent point/optimizer cost. Require official unseen-view, full 30K,
multi-scene three-seed quality-and-wall-time confirmation before any SOTA claim.
No current proposal has met the user's combined goal.

All older queues below are historical and superseded by this section.

## Current (2026-09-27): resolution/topology protocol correction

Read RESOLUTION_TOPOLOGY_DESIGN.md. The 4-scene 30K `higs_proposed` arm is
half-resolution until 24K with visible Adam, not tile sampling. Default
densification stops at 15K, so it never sees full-resolution training signal.
Do not use CAPACITY_COVERAGE_DESIGN.md's tile-miss mechanism to explain that arm.
P11 fixed-state room/garden/bicycle probes show half-resolution threshold
candidates are more numerous; same-budget overlap with full resolution is
81.41%/85.01%/78.26%.
This rejects simple candidate under-counting and leaves quality causality open.

Next: same-paper-trainer 12K switch control, then sparse full-resolution topology
scouts with a matched scout-but-original-topology arm. Count all extra compute,
point-budget and optimizer-state effects. Full 30K official unseen-view paired
quality/time is mandatory. No current SOTA, novelty or end-to-end win.

The stale transported-mean v3 diagnostic localizes latent partition differences
to ~1e-7 relative, but combined rotation VJP mean residual remains .00648;
old-cache geometry noise remains a separate failure. Do not revive this route.

All older queues below are historical and superseded by this section.

## Latest: shared-sort counterfactual closed; spatial cache still not usable

Read SPATIAL_PREDICTOR_RESULTS.md and SPATIAL_PREDICTOR_DESIGN.md. Six new GPU
diagnostics completed, plus two separate 128-step native dense trajectories used
to test cache aging (not cache-trained trajectories). All own handles terminal.

Fine16/coarse8 intersection lists matched exactly on three garden views. Making
coarse16 sorting free still lost to native in all three. Do not implement tile8
solely on the premise that eliminating sorting is enough.

Center-allocated Adam history failed on all garden groups. Fresh same-view spatial
moment quadrature improved noise with beta=.5 selected on garden and then tested
on room; all room groups passed. This uses unavailable fresh full-image moments.

Actual old latent gradients failed after 16/64/128 native updates. Reintegrating a
fixed screen-space polynomial adjoint helped but still increased geometry noise.
Worse, transported rotation complete-partition mean residual is 2%-3.2%, versus
ordinary sampling 0.016%-0.041%. Do not call the GPU transport numerically verified
unbiased; mean max-abs was not recorded, and the usual OR gate cannot be inferred.
Preserve both raw runs and the negative result. No complete-training benefit.

Next if this mechanism is pursued: first isolate the transported mean discrepancy
at one stale state (sum latent predictions vs full latent prediction, then apply
the same native projection Jacobian; include absolute errors and independent GPU
replays). This may distinguish predictor accumulation from conditioned quaternion
backprop, but neither explanation is established. Do not loosen gates. Even after
a numerical fix, long-lived geometry cache correlation is independently weak.
Any replacement algorithm must provide genuinely cheap current-view prediction,
count cache refresh/storage/maintenance, and prove quality-versus-time in training.

The user's broad goal remains active and unmet. No SOTA or novelty claim. All
older queues below are superseded.

## Latest: two-resolution gradient opportunity, cost gate still failed

Read MULTILEVEL_RESULTS.md and MULTILEVEL_DESIGN.md first. Native half-size proxy
reduced raw gradient and saved-history Adam-update error on all six fixed views.
Factor4 was unreliable. These are dense-render variance oracles, not training.

Room native/component SH discrepancy was caused by clamp_min versus native ReLU
at zero. The versioned v2 helper passes the unchanged gate. Garden shared fine
render/halo and selected-core SSIM pass the same gate. No frozen source changed.

Garden remains slower after sharing projection/SH, selected-core loss, and linear
conic scaling: final scaled+patch 25.314/21.282/19.943 ms versus same-run native
18.458/19.565/18.633 ms. Do not repeat these variants or launch full training to
rescue a headline. The new proxies have NOT had their gradient variance measured;
shared_gradient_probe.py and scaled_gradient_probe.py are unexecuted.

Next concrete opportunity, only if pursuing this mechanism: profile duplicated
intersection/sort cost and test list equality for scaled proxy fine tile16 versus
coarse tile8. Both cover the same continuous image region. Existing pinned 3DGS
raster supports only {4,16}, so tile8 requires an isolated implementation; do not
modify the frozen runtime. Sharing lists still leaves two alpha traversals.
If the removable cost cannot cover the deficit, close this implementation path.

No own GPU process remains. Recheck occupancy before launching anything. Full
training with nonlinear densification, official unseen-view metrics, matched
quality/time comparisons, modern strong baselines and novelty audit remain
required. The ambitious goal stays active and unmet.

All older queues below are historical and superseded by this section.

## Latest: sibling and transient-coordinate probes completed

Read TRANSIENT_RESULTS.md and SIBLING_RESULTS.md. This turn completed 52
permanent-coordinate arms and 40 transient-coordinate arms, plus an independent
garden numerical replay audit. All own process handles are terminal; GPU 6 was
empty at final check. Other GPUs have foreign jobs: recheck before any launch.

Identical initial physical parameters/images and chain rules were verified.
Garden initially failed the fixed rotation-gradient tolerance; the failure and
an independently measured native replay envelope are preserved, with an explicit
validation amendment. Do not call this an unchanged preregistered gate.

Permanent coordinates did not give a stable final win. A 32-step handoff selected
from room development curves preserved parameters/images and raw EMA histories,
but still did not give stable cross-scene final A/B gains. Its extra step cost was
about 2.06%/2.38%, not a native-training speedup. Do not promote this configuration,
write production CUDA, or continue sweeping lifetimes to rescue a headline.

Meaningful technical progress: under explicitly zero-extended relative prehistory
and common EMA betas/counts, raw moments can be transported exactly despite
different bias clocks used for parameter updates. Cross moments are necessary;
physical handoff uses the old clock and does not preserve effective step sizes.
sibling_transport.py passed independent PyTorch history checks. It is a reusable
research helper, not an achieved training optimization.

The ambitious user goal remains active and unmet. No next mechanism is selected
as a proven powerful alternative. Before committing to another small optimizer
variant, reassess opportunity against actual full densification workloads and
strong published implementations: these probes touch only 256 parents out of
260530/1841621 points. Any revival needs a material mechanism opportunity, a
fairly normalized/matched-step control, and original-loss/time accounting.
Full training, official unseen-view evaluation, competition with current strong
methods, and source-level novelty checks remain required for the goal's claims.

All older queues below are historical and superseded by this section.

## Latest: newborn state audit completed

Read NEWBORN_RESULTS.md and NEWBORN_PROTOCOL.md first. Two scenes x 17 arms
completed successfully; no GPU experiment remains running. Initial child hashes
match across policies; the dense fresh-age helper passed two independent Adam
tests. Current source and raw evidence are frozen by their recorded SHA256.

BOGausS (EUSIPCO 2025) directly covers per-Gaussian lifespan and parent-to-child
moment inheritance, including 0.2/0.04 scaling. Do not claim those mechanisms as
novel or call our conventional-Adam controls a full BOGausS reproduction.

Fresh-age made B worse in all six final matched split draws, and A also got
worse on average. Constant newborn LR scaling did not fix it. Inheritance had
short-term benefits and mixed final results. Stop promotion of these standalone
controls as the powerful optimization; do not run a full campaign just to try
to rescue a headline.

Next candidate hypothesis is specified in SIBLING_COORDINATES_DESIGN.md:
separate shared child motion from newly added relative freedom, with distinct
histories. It is NOT implemented, validated or established as novel. First verify
identical initial renderer output and gradient chain rules, then compare basis-only,
shared-history and differential-age effects against native and inherited-moment
controls. Do not proceed to repeated topology or production CUDA without a real
benefit; repeated split/prune state transport and unequal-age bias correction
remain mandatory unresolved requirements, not optional polish.

All older queues below are historical and superseded by this section.

## Current decision after real GPU probes

Read REAL_TOPOLOGY_RESULTS.md first. Room/garden@2000, 64-step branches, and
room's 512-step extension are complete (30 arms total). All process handles
returned exit_code=0. Source hashes, optimizer survivor moments and zero newborn
moments were verified. No experiment remains running.

The current conditional finite-edit scoring configurations did not establish
a robust useful selection advantage. Do not implement their production CUDA
or spend a large full-training campaign on the strength of the algebra checks.
Horizon-dependent ranking, A/B tradeoffs and replay variation are material.

Next concrete mechanism audit: native newborn rows have zero Adam moments but
inherit group step=1985 at room@2000, giving a first update 2.9375x fresh Adam
in the measured case. This is not yet an optimizer bug or a novel method.
Check prior art and compare native, age-adjusted, and justified transported
moment states under identical split candidates/initialization, with stronger
replay controls. Determine whether split initialization, transient optimizer
behavior or genuine capacity explains any effect before another design claim.

If pursuing longer-horizon topology value instead, account for baseline replay
variance and view coverage; immediate score cannot substitute for mature-child
benefit. A shared-context shadow comparator may reduce trajectory noise, but
is only an unimplemented research question and not a full counterfactual.

The earlier queues below are retained as history, superseded by this section.

The user asked for a powerful 3DGS design that trains faster and reconstructs
better. This goal is not achieved by the present candidate or its eight tests.
This turn made progress by producing executable oracles and falsifying several
optimistic assumptions. All launched local probes have completed; do not infer
that GPU training is running.

## Superseding direction: finite topology edits (2026-09-26)

Read TOPOLOGY_DESIGN.md first. The old SH solver positioning is closed after
direct prior-art collision (OGC / Instant Colorization) and the 0/5 result
against scaled full Adam. The numbered solver steps below are historical,
not the current execution queue.

SteepGS already separates mean-shift and split through a second-order matrix;
A New Split Algorithm already preserves moments. Neither is our novelty.
The new candidate uses real finite alpha edits conditional on old-parameter
repair, original-loss validation and remaining interaction cost. Its six groups
of CPU checks establish algebra/counterexamples, not training benefits.

Next: use faithful early/mid P11 stategen checkpoints with optimizer history
for paired finite-edit branches. Compare absgrad, FastGS-style scoring,
SteepGS, raw finite edits and conditional finite edits on identical candidates.
Check long-horizon useful-split rejection and omitted old repair directions.
Only if that opportunity exists, build a sparse estimator and measure all costs.
Remote EPIC-05 states exist; inspect occupancy before any GPU run. No new GPU
training is running. Full-training speed, quality and novelty remain unverified.

## Historical solver queue (superseded)

1. Revalidate current files and read the three subspace evidence JSONs. The
   bounded candidate did not beat scaled Adam in any of five development seeds.
   Do not promote it based on its advantage over a fixed small Adam step.
2. Before any CUDA rewrite, check real early/middle/late training states for
   coupled geometry/appearance opportunity, with Adam history preserved. The
   current PLY fixtures are late pretrained states only; locate genuine earlier
   states or create a separate faithful small training trace. Do not fabricate
   an early state by perturbing a converged model and call it training evidence.
3. Compare matched block coverage and equal trial budgets: base Adam, scaled
   Adam, independent per-Gaussian updates, coupled updates. Count model build,
   Jacobian work, trial rendering and rollback. Use original photometric loss
   and keep an internal training-view validation split.
4. Verify fixed-geometry production-renderer parity before interpreting real
   image quality. The installed gsplat backend may trigger a JIT build and its
   loader removes a cache lock; inspect current runtime/build state before
   importing it. Do not edit installed or frozen P12 code to make this probe run.
5. If the real-state coupling opportunity is insufficient, close this specific
   solver design and prioritize topology/capacity allocation against FastGS and
   Taming controls. Preserve the original ambitious goal; do not substitute
   late-stage color refinement as its completion condition.

Final acceptance still requires a concrete design supported at the scope of its
claims. SOTA/2x assertions require matched full-training results, not these
diagnostic scripts. Novelty needs further source-level prior-art comparison.
