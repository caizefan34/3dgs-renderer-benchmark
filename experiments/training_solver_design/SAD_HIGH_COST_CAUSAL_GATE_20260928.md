# SAD high-cost primitive causal opportunity gate (frozen before results)

Date: 2026-09-28. This is an **existing-checkpoint diagnostic**, not a training method, formal untouched-scene evaluation, or originality claim. Motivation: the SAD original-loss room 5K system is faster than FastGS-Big but loses .9865 dB mean PSNR; 79.45% of excess mean MSE is in `>64px` residual blocks. Eight training views show top 10% projected-footprint primitives contribute about half of forward accepted pairs, but spatial residual deciles do not concentrate accepted work. A direct *primitive-level* causal overlap remains unmeasured.

## Frozen state and selection

- Author SAD-GS commit `029814fb5ed773e4a794cec5115951a64f610ab8`, original-loss room 5K PLY SHA256 `3a088855de73bcd4e123d105812791ab75dd621bd19b0dc6cc89789b4c552894`, effective 1600×1066, `mult=.25` for training-view work/quality diagnostic.
- Use the previously registered eight sorted training cameras at indices `0,39,77,116,155,194,232,271`. Run the existing native `get_flag`/all-one `metric_map` counter. Rank Gaussian IDs by **sum of accepted Gaussian–pixel pairs across those eight views**, breaking ties by lower ID. This choice uses no held-out GT or held-out work.
- Select the top 32 IDs; record each and their aggregate share of the eight-view baseline accepted count. If the top 32 account for less than 5% of work, stop before deletion tests: individual edits cannot by themselves establish a material full-training cost route. The 5% is an opportunity filter, not a predicted wall-time saving.

## Conditional counterfactual

If the cost filter passes, in memory only, set each selected Gaussian's raw opacity logit to −30 one at a time, render the same eight selection cameras and eight **disjoint training** validation cameras at sorted indices `20,59,97,136,174,213,251,270`, then restore the exact original logit. Compare the original full-resolution L1+fusedSSIM loss and RGB PSNR on all 16 cameras. Do not inspect official test GT for candidate choice or this gate. On a candidate with a positive validation quality signal, re-run native accepted-pair counting before and after deletion on those cameras, because removing an occluder can increase deeper work. Do **not** add individual candidate savings; interactions overlap and early exit changes.

Stop if no selected ID reduces mean loss on the eight disjoint training validation cameras without worsening more than one of them, or if actual accepted count does not fall. A pass only identifies a candidate for a later training-only rule and multi-view group intervention; it does not prove held-out generalization, a viable replacement, original novelty, SSIM/LPIPS noninferiority, or complete training wall benefit. Keep all 39 views in any eventual official metrics. Do not modify the author checkout or launch a training arm from this gate alone.
