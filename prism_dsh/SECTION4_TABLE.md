# Section 4/7 交付表（自动生成）

单元数：12（其中完整 12，期望 12）

## 跨 seed 汇总（每场景/迭代）

### bicycle 14k seeds=['s20260927', 's20260928', 's20260929']

- direct_geometry_adam: train_delta_mean=-6.63805e-06 audit_delta_mean=3.99016e-05 per_seed_audit=['3.13409e-05', '5.58123e-05', '3.25516e-05'] with_eval_s_mean=0.386742 peak_MB_mean=6442.8
- screen_nonlinear: train_delta_mean=-8.32106e-06 audit_delta_mean=6.25973e-06 per_seed_audit=['1.22823e-05', '3.74019e-06', '2.75671e-06'] with_eval_s_mean=0.40161 peak_MB_mean=3701.8
- screen_one_step: train_delta_mean=-6.93566e-06 audit_delta_mean=6.90048e-06 per_seed_audit=['1.2219e-05', '5.26011e-06', '3.22238e-06'] with_eval_s_mean=1.06469 peak_MB_mean=3701.8
- audit 最优臂（逐 seed，越小越好）: ['screen_one_step', 'screen_nonlinear', 'screen_nonlinear']

### bicycle 5k seeds=['s20260927', 's20260928', 's20260929']

- direct_geometry_adam: train_delta_mean=-4.4196e-05 audit_delta_mean=3.48041e-05 per_seed_audit=['2.20127e-05', '5.82375e-05', '2.41622e-05'] with_eval_s_mean=0.287081 peak_MB_mean=3723.6
- screen_nonlinear: train_delta_mean=-3.17925e-05 audit_delta_mean=-3.93515e-06 per_seed_audit=['-1.04308e-07', '2.98023e-07', '-1.19992e-05'] with_eval_s_mean=0.344632 peak_MB_mean=2202.9
- screen_one_step: train_delta_mean=-2.58336e-05 audit_delta_mean=-2.13956e-06 per_seed_audit=['2.11224e-06', '1.81049e-06', '-1.03414e-05'] with_eval_s_mean=1.02852 peak_MB_mean=2202.9
- audit 最优臂（逐 seed，越小越好）: ['screen_nonlinear', 'screen_nonlinear', 'screen_nonlinear']

### garden 14k seeds=['s20260927', 's20260928', 's20260929']

- direct_geometry_adam: train_delta_mean=-5.46293e-06 audit_delta_mean=3.31911e-05 per_seed_audit=['2.91243e-05', '3.32221e-05', '3.72268e-05'] with_eval_s_mean=0.335512 peak_MB_mean=5400.6
- screen_nonlinear: train_delta_mean=-7.33551e-06 audit_delta_mean=7.07805e-08 per_seed_audit=['4.91738e-07', '-4.5076e-07', '1.71363e-07'] with_eval_s_mean=0.36046 peak_MB_mean=3113.9
- screen_one_step: train_delta_mean=-6.11113e-06 audit_delta_mean=3.71287e-07 per_seed_audit=['1.58697e-06', '8.41916e-07', '-1.31503e-06'] with_eval_s_mean=1.069 peak_MB_mean=3113.9
- audit 最优臂（逐 seed，越小越好）: ['screen_nonlinear', 'screen_nonlinear', 'screen_one_step']

### garden 5k seeds=['s20260927', 's20260928', 's20260929']

- direct_geometry_adam: train_delta_mean=-2.79579e-05 audit_delta_mean=2.72666e-05 per_seed_audit=['3.18661e-05', '2.9467e-05', '2.04667e-05'] with_eval_s_mean=0.275407 peak_MB_mean=3669.8
- screen_nonlinear: train_delta_mean=-1.76562e-05 audit_delta_mean=6.74029e-06 per_seed_audit=['1.18837e-05', '4.47035e-08', '8.2925e-06'] with_eval_s_mean=0.334119 peak_MB_mean=2213.7
- screen_one_step: train_delta_mean=-1.41677e-05 audit_delta_mean=7.16994e-06 per_seed_audit=['8.87364e-06', '6.85453e-07', '1.19507e-05'] with_eval_s_mean=1.0504 peak_MB_mean=2213.7
- audit 最优臂（逐 seed，越小越好）: ['screen_one_step', 'screen_nonlinear', 'screen_nonlinear']

## 逐单元明细

## bicycle 14k seed s20260927

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=3573165; 共同可见=645971; active=512
- cameras: [163, 28, 6, 192, 72]
- observability: min_rel_sing=0.230503, median=0.399123, frac<1e-6=0
- shared_setup_seconds: 0.135466
- before_per_view_loss: [0.06196104735136032, 0.08661653101444244, 0.07192636281251907, 0.08140524476766586, 0.0702553391456604]
- screen_one_step: train_delta=-6.4373e-06 audit_delta=1.2219e-05 opt_s=0.939097 with_eval_s=1.02445 bwd_views=3 peak_MB=3701.8
- screen_nonlinear: train_delta=-7.86657e-06 audit_delta=1.22823e-05 opt_s=0.338688 with_eval_s=0.38728 bwd_views=9 peak_MB=3701.8
- direct_geometry_adam: train_delta=-9.22879e-06 audit_delta=3.13409e-05 opt_s=0.336044 with_eval_s=0.38492 bwd_views=9 peak_MB=6442.8

## bicycle 14k seed s20260928

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=3573165; 共同可见=645971; active=512
- cameras: [163, 28, 6, 192, 72]
- observability: min_rel_sing=0.238485, median=0.414648, frac<1e-6=0
- shared_setup_seconds: 0.132614
- before_per_view_loss: [0.06196104735136032, 0.08661653101444244, 0.07192636281251907, 0.08140524476766586, 0.0702553391456604]
- screen_one_step: train_delta=-7.80821e-06 audit_delta=5.26011e-06 opt_s=1.07367 with_eval_s=1.17042 bwd_views=3 peak_MB=3701.8
- screen_nonlinear: train_delta=-8.5781e-06 audit_delta=3.74019e-06 opt_s=0.368689 with_eval_s=0.420119 bwd_views=9 peak_MB=3701.8
- direct_geometry_adam: train_delta=-1.16477e-06 audit_delta=5.58123e-05 opt_s=0.338169 with_eval_s=0.388879 bwd_views=9 peak_MB=6442.8

## bicycle 14k seed s20260929

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=3573165; 共同可见=645971; active=512
- cameras: [163, 28, 6, 192, 72]
- observability: min_rel_sing=0.227775, median=0.403664, frac<1e-6=0
- shared_setup_seconds: 0.134732
- before_per_view_loss: [0.06196104735136032, 0.08661653101444244, 0.07192636281251907, 0.08140524476766586, 0.0702553391456604]
- screen_one_step: train_delta=-6.56148e-06 audit_delta=3.22238e-06 opt_s=0.910015 with_eval_s=0.999207 bwd_views=3 peak_MB=3701.8
- screen_nonlinear: train_delta=-8.5185e-06 audit_delta=2.75671e-06 opt_s=0.347042 with_eval_s=0.39743 bwd_views=9 peak_MB=3701.8
- direct_geometry_adam: train_delta=-9.5206e-06 audit_delta=3.25516e-05 opt_s=0.337428 with_eval_s=0.386427 bwd_views=9 peak_MB=6442.8

## bicycle 5k seed s20260927

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=1879621; 共同可见=209975; active=512
- cameras: [163, 28, 6, 192, 72]
- observability: min_rel_sing=0.220622, median=0.399248, frac<1e-6=0
- shared_setup_seconds: 0.116322
- before_per_view_loss: [0.11102078855037689, 0.1385633796453476, 0.12809669971466064, 0.1568148136138916, 0.12022870779037476]
- screen_one_step: train_delta=-2.19891e-05 audit_delta=2.11224e-06 opt_s=0.953456 with_eval_s=1.0255 bwd_views=3 peak_MB=2202.9
- screen_nonlinear: train_delta=-3.24696e-05 audit_delta=-1.04308e-07 opt_s=0.309989 with_eval_s=0.344893 bwd_views=9 peak_MB=2202.9
- direct_geometry_adam: train_delta=-4.35064e-05 audit_delta=2.20127e-05 opt_s=0.251413 with_eval_s=0.287463 bwd_views=9 peak_MB=3723.6

## bicycle 5k seed s20260928

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=1879621; 共同可见=209975; active=512
- cameras: [163, 28, 6, 192, 72]
- observability: min_rel_sing=0.22477, median=0.395998, frac<1e-6=0
- shared_setup_seconds: 0.112531
- before_per_view_loss: [0.11102078855037689, 0.1385633796453476, 0.12809669971466064, 0.1568148136138916, 0.12022870779037476]
- screen_one_step: train_delta=-2.15371e-05 audit_delta=1.81049e-06 opt_s=0.937097 with_eval_s=1.00978 bwd_views=3 peak_MB=2202.9
- screen_nonlinear: train_delta=-2.23567e-05 audit_delta=2.98023e-07 opt_s=0.312578 with_eval_s=0.347355 bwd_views=9 peak_MB=2202.9
- direct_geometry_adam: train_delta=-2.92833e-05 audit_delta=5.82375e-05 opt_s=0.252831 with_eval_s=0.288536 bwd_views=9 peak_MB=3723.6

## bicycle 5k seed s20260929

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=1879621; 共同可见=209975; active=512
- cameras: [163, 28, 6, 192, 72]
- observability: min_rel_sing=0.227283, median=0.398916, frac<1e-6=0
- shared_setup_seconds: 0.108775
- before_per_view_loss: [0.11102078855037689, 0.1385633796453476, 0.12809669971466064, 0.1568148136138916, 0.12022870779037476]
- screen_one_step: train_delta=-3.39746e-05 audit_delta=-1.03414e-05 opt_s=0.978298 with_eval_s=1.05027 bwd_views=3 peak_MB=2202.9
- screen_nonlinear: train_delta=-4.0551e-05 audit_delta=-1.19992e-05 opt_s=0.307107 with_eval_s=0.341647 bwd_views=9 peak_MB=2202.9
- direct_geometry_adam: train_delta=-5.97984e-05 audit_delta=2.41622e-05 opt_s=0.250352 with_eval_s=0.285244 bwd_views=9 peak_MB=3723.6

## garden 14k seed s20260927

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=2922031; 共同可见=295319; active=512
- cameras: [163, 28, 6, 72, 64]
- observability: min_rel_sing=0.358905, median=0.61083, frac<1e-6=0
- shared_setup_seconds: 0.137233
- before_per_view_loss: [0.08413533121347427, 0.1053733080625534, 0.07719264179468155, 0.08813448250293732, 0.09149180352687836]
- screen_one_step: train_delta=-7.71135e-06 audit_delta=1.58697e-06 opt_s=0.966691 with_eval_s=1.04794 bwd_views=3 peak_MB=3113.9
- screen_nonlinear: train_delta=-9.26107e-06 audit_delta=4.91738e-07 opt_s=0.320642 with_eval_s=0.3634 bwd_views=9 peak_MB=3113.9
- direct_geometry_adam: train_delta=-5.17567e-06 audit_delta=2.91243e-05 opt_s=0.293812 with_eval_s=0.336891 bwd_views=9 peak_MB=5400.6

## garden 14k seed s20260928

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=2922031; 共同可见=295319; active=512
- cameras: [163, 28, 6, 72, 64]
- observability: min_rel_sing=0.373085, median=0.60948, frac<1e-6=0
- shared_setup_seconds: 0.129197
- before_per_view_loss: [0.08413533121347427, 0.1053733080625534, 0.07719264179468155, 0.08813448250293732, 0.09149180352687836]
- screen_one_step: train_delta=-4.03325e-06 audit_delta=8.41916e-07 opt_s=1.00454 with_eval_s=1.08383 bwd_views=3 peak_MB=3113.9
- screen_nonlinear: train_delta=-4.84784e-06 audit_delta=-4.5076e-07 opt_s=0.313584 with_eval_s=0.355684 bwd_views=9 peak_MB=3113.9
- direct_geometry_adam: train_delta=-5.64009e-06 audit_delta=3.32221e-05 opt_s=0.291607 with_eval_s=0.334065 bwd_views=9 peak_MB=5400.6

## garden 14k seed s20260929

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=2922031; 共同可见=295319; active=512
- cameras: [163, 28, 6, 72, 64]
- observability: min_rel_sing=0.209849, median=0.599694, frac<1e-6=0
- shared_setup_seconds: 0.118921
- before_per_view_loss: [0.08413533121347427, 0.1053733080625534, 0.07719264179468155, 0.08813448250293732, 0.09149180352687836]
- screen_one_step: train_delta=-6.5888e-06 audit_delta=-1.31503e-06 opt_s=0.993955 with_eval_s=1.07522 bwd_views=3 peak_MB=3113.9
- screen_nonlinear: train_delta=-7.89762e-06 audit_delta=1.71363e-07 opt_s=0.319538 with_eval_s=0.362297 bwd_views=9 peak_MB=3113.9
- direct_geometry_adam: train_delta=-5.57303e-06 audit_delta=3.72268e-05 opt_s=0.292498 with_eval_s=0.335579 bwd_views=9 peak_MB=5400.6

## garden 5k seed s20260927

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=1813591; 共同可见=239120; active=512
- cameras: [163, 28, 6, 72, 64]
- observability: min_rel_sing=0.226664, median=0.552604, frac<1e-6=0
- shared_setup_seconds: 0.106826
- before_per_view_loss: [0.15662480890750885, 0.17162451148033142, 0.13894285261631012, 0.14218562841415405, 0.14987517893314362]
- screen_one_step: train_delta=-1.5681e-05 audit_delta=8.87364e-06 opt_s=0.940074 with_eval_s=1.0132 bwd_views=3 peak_MB=2213.7
- screen_nonlinear: train_delta=-1.80552e-05 audit_delta=1.18837e-05 opt_s=0.298998 with_eval_s=0.334235 bwd_views=9 peak_MB=2213.7
- direct_geometry_adam: train_delta=-2.61366e-05 audit_delta=3.18661e-05 opt_s=0.238959 with_eval_s=0.274326 bwd_views=9 peak_MB=3669.8

## garden 5k seed s20260928

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=1813591; 共同可见=239120; active=512
- cameras: [163, 28, 6, 72, 64]
- observability: min_rel_sing=0.225712, median=0.555123, frac<1e-6=0
- shared_setup_seconds: 0.115447
- before_per_view_loss: [0.15662480890750885, 0.17162451148033142, 0.13894285261631012, 0.14218562841415405, 0.14987517893314362]
- screen_one_step: train_delta=-1.45088e-05 audit_delta=6.85453e-07 opt_s=1.0195 with_eval_s=1.09515 bwd_views=3 peak_MB=2213.7
- screen_nonlinear: train_delta=-1.78069e-05 audit_delta=4.47035e-08 opt_s=0.301615 with_eval_s=0.33766 bwd_views=9 peak_MB=2213.7
- direct_geometry_adam: train_delta=-2.84413e-05 audit_delta=2.9467e-05 opt_s=0.241909 with_eval_s=0.277699 bwd_views=9 peak_MB=3669.8

## garden 5k seed s20260929

- status: `PROBE_COMPLETE_NO_GO_VERDICT`; N=1813591; 共同可见=239120; active=512
- cameras: [163, 28, 6, 72, 64]
- observability: min_rel_sing=0.392548, median=0.54814, frac<1e-6=0
- shared_setup_seconds: 0.113359
- before_per_view_loss: [0.15662480890750885, 0.17162451148033142, 0.13894285261631012, 0.14218562841415405, 0.14987517893314362]
- screen_one_step: train_delta=-1.23133e-05 audit_delta=1.19507e-05 opt_s=0.970657 with_eval_s=1.04287 bwd_views=3 peak_MB=2213.7
- screen_nonlinear: train_delta=-1.71065e-05 audit_delta=8.2925e-06 opt_s=0.295588 with_eval_s=0.330464 bwd_views=9 peak_MB=2213.7
- direct_geometry_adam: train_delta=-2.92957e-05 audit_delta=2.04667e-05 opt_s=0.238997 with_eval_s=0.274197 bwd_views=9 peak_MB=3669.8
