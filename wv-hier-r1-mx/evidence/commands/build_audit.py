"""Recompute audit outputs from preserved WV0 summaries; never invent raw data."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'raw/evidence'
REMOTE='/mnt/storage_pool/liaoyuanjun/wv0_raw_audit_20261008_2148'
GROUPS=['means','quats','scales','opacities','colors']
GEOM=GROUPS[:3]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def textfile(name,txt):(ROOT/name).write_text(txt,encoding='utf-8')
def csvfile(name,rows):
 with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean_metric(c,m):return statistics.mean(c['groups'][g][m]['mean'] for g in GEOM)
states=load(E/'wv0_state_summaries_exact.json')
cameras=load(E/'camera_image_identity.json')
ckpts=load(E/'checkpoint_hash_verification.json')
source=load(E/'gsplat_source_manifest.json')
pre=load(E/'mx_preflight.json')
cost_ledger=load(E/'WV1_ORIGINAL_COST_LEDGER.json')
camera_map={x['scene']:x for x in cameras}
raw_json=load(E/'wv0_snapshot/WV_HIER_ORACLE0_MX/WV0_GRADIENT_RAW.json')
inventory=load(ROOT/'raw/recovery_file_inventory.json')
local_inventory=load(ROOT/'raw/local_file_inventory.json')
for rec in inventory+local_inventory:
 name=Path(rec['path']).name
 if name.endswith('.pt') and name[:-3] in states:
  s=states[name[:-3]]
  rec.update({'schema':'torch-saved compact state summary, inspected with weights_only=True',
   'state_id':s['state_id'],'scene':s['scene'],'checkpoint_sha256':s['ckpt_sha256'],
   'camera_count':s['n_cams'],'conditions':list(s['conditions']),
   'saved_condition_records':len(s['conditions']),'saved_individual_camera_seed_records':0})
 elif name=='WV0_GRADIENT_RAW.json':
  rec.update({'schema':'WV0_GRADIENT_RAW.v1; content is aggregate dictionaries',
   'state_ids':list(states),'G_domain':[16,8,4,1],'p_domain':[.25,.4],
   'seed_domain_claimed':'0..31','saved_individual_camera_seed_records':0,
   'saved_state_condition_records':72,'condition_index':'raw/CONDITION_INDEX.csv'})
condition_index=[]
absolute=[]
pareto=[]
equal=[]
group_audit=[]
for sid,s in states.items():
 h,w=s['cam_hw'];npx=h*w;scene=s['scene']
 selected=camera_map[scene]['selected_first_8']
 for ck,c in s['conditions'].items():
  G=int(ck.split('_')[0][1:]);p=float(ck.split('_p')[1]);iw=s['work']['per_G'][str(G)]['dense_isects'];i16=s['work']['per_G']['16']['dense_isects']
  condition_index.append({'state_id':sid,'scene':scene,'checkpoint_sha256':s['ckpt_sha256'],
   'camera_ids_reported_by_source':';'.join(str(a['id']) for a in selected),'camera_count':s['n_cams'],
   'G':G,'p':p,'seed_domain':'0..31','mask_replications_per_camera':32,'saved_camera_seed_rows':0,'summary_sample_unit':'camera-average per seed, then statistics over 32 seeds'})
  absolute.append({'state_id':sid,'scene':scene,'checkpoint_sha256':s['ckpt_sha256'],'N':s['N'],
   'cost_camera_index':0,'cost_camera_id':selected[0]['id'],'height':h,'width':w,'G':G,'p':p,
   'evidence_class':'DERIVED_STRUCTURAL_ORACLE_ONLY','dense_intersection_proxy':iw,
   'expected_selected_intersection_proxy':p*iw,'normalized_to_own_G_dense':p,
   'ratio_to_same_p_G16_proxy':iw/i16,'ratio_to_G16_dense_proxy':p*iw/i16,
   'expected_sort_entries_if_proxy_valid':p*iw,'key_value_bytes_per_entry':12,
   'key_value_payload_bytes_if_proxy_valid':12*p*iw,
   'expected_selected_pixels':p*npx,'observed_saved_mean_selected_pixels':c['sel_px_mean'],
   'observed_saved_active_pixel_fraction':c['sel_px_mean']/npx,
   'observed_saved_mean_selected_tiles':c['sel_tiles_mean'],
   'physical_cuda_intersections':'NOT_MEASURED','actual_sort_bytes':'NOT_MEASURED',
   'kernel_ms':'NOT_MEASURED','forward_backward_ms':'NOT_MEASURED'})
  pareto.append({'state_id':sid,'G':G,'p':p,'nmse_geometry_macro':mean_metric(c,'nmse'),
   'cosine_geometry_macro':mean_metric(c,'cos'),'expected_absolute_intersection_proxy':p*iw,
   'estimated_key_value_payload_bytes':12*p*iw,'frontier_same_p':None,'frontier_all_saved_p':None,
   'dominated_by_same_p':'','dominated_by_all_saved_p':'','matched_cost_camera_and_quality_camera':False,
   'quality_camera_scope':'mean of 8 cameras','cost_camera_scope':'camera 0 only',
   'evidence_class':'RECOMPUTED_SAVED_SUMMARIES_PLUS_DERIVED_PROXY','measured_FB_ms':'NOT_AVAILABLE'})
  base=s['conditions'][f'G16_p{p}'];n=mean_metric(c,'nmse');bn=mean_metric(base,'nmse')
  r=iw/i16;pi=p/r;K=n*p/(1-p);pred=K*(1-pi)/pi
  equal.append({'state_id':sid,'G':G,'source_p':p,'target_cost_reference':f'G16_p{p}',
   'target_absolute_proxy_budget':p*i16,'implied_candidate_p':pi,
   'estimated_K_from_source_p':K,'model_predicted_nmse':pred,'G16_saved_nmse':bn,
   'model_predicted_ratio_vs_G16':pred/bn,'target_p_was_measured':G==16,
   'evidence_class':'DERIVED_BERNOULLI_EXPECTATION_MODEL_NOT_NEW_MEASUREMENT',
   'assumptions':'fixed gradient; independent tile Bernoulli; saved finite-seed NMSE estimates expectation; camera0 cost represents camera-average metric'})
  for g in GROUPS:
   d=c['groups'][g]
   group_audit.append({'state_id':sid,'G':G,'p':p,'group':g,
    'saved_nmse_mean':d['nmse']['mean'],'saved_cosine_mean':d['cos']['mean'],
    'saved_single_mask_relL2_mean':d['rel_l2']['mean'],
    'saved_mean_gradient_relative_error':d['bias_rel_l2'],
    'saved_pooled_camera_mask_variance':d['variance'],
    'saved_stat_n':d['nmse']['n'],'stored_per_seed_array':isinstance(d['nmse'],list),
    'raw_export_equals_summary':raw_json['states'][sid]['conditions'][ck]['groups'][g]['nmse_per_mask']==d['nmse']})
for a in pareto:
 def dominates(b):return b['expected_absolute_intersection_proxy']<=a['expected_absolute_intersection_proxy'] and b['nmse_geometry_macro']<=a['nmse_geometry_macro'] and (b['expected_absolute_intersection_proxy']<a['expected_absolute_intersection_proxy'] or b['nmse_geometry_macro']<a['nmse_geometry_macro'])
 for same,flag,label in [(True,'frontier_same_p','dominated_by_same_p'),(False,'frontier_all_saved_p','dominated_by_all_saved_p')]:
  ds=[b for b in pareto if b['state_id']==a['state_id'] and (not same or b['p']==a['p']) and dominates(b)]
  a[flag]=not ds;a[label]=';'.join(f"G{b['G']}_p{b['p']}" for b in ds)
csvfile('WV0_ABSOLUTE_COST.csv',absolute)
csvfile('WV0_PARETO_RECHECK.csv',pareto)
csvfile('raw/EQUAL_COST_MODEL.csv',equal)
csvfile('raw/CONDITION_INDEX.csv',condition_index)
csvfile('raw/GROUP_STATISTICS_AUDIT.csv',group_audit)
dump('WV0_RAW_INVENTORY.json',{'schema':'WV0_RAW_INVENTORY.audit.v1','evidence_class':'OBSERVED',
 'search_roots':['/mnt/storage_pool/liaoyuanjun','/home/liaoyuanjun','/dev/shm','C:/Users/36570/3dgs-renderer-benchmark/WV_HIER_ORACLE0_MX'],
 'maxdepth_primary_search':9,'search_command':'commands/05_recovery_search.sh',
 'remote_files':inventory,'local_files':local_inventory,'archives':load(ROOT/'raw/archive_inventory.json'),
 'local_archive':load(ROOT/'raw/local_archive_inventory.json'),
 'condition_index':'raw/CONDITION_INDEX.csv','deduplication':'Files with equal SHA256 are copies, never additional samples.',
 'recovery_result':'ORIGINAL_AGGREGATES_RECOVERED_NO_CAMERA_SEED_RAW_METRICS_FOUND'})
dump('WV0_DATA_COMPLETENESS.json',{'schema':'WV0_DATA_COMPLETENESS.audit.v1','status':'BLOCKED_RAW_EVIDENCE',
 'checkpoints':ckpts,'checkpoint_sha_pass_count':sum(x['matches_wv0'] for x in ckpts),'original_manifest_pass_count':94,
 'claimed_estimate_count':9216,'cartesian_source_implied_count':18432,'camera_averaged_seed_units_if_retained':2304,
 'actual_saved_state_condition_count':len(condition_index),'actual_saved_group_stat_record_count':len(group_audit),
 'actual_saved_metric_stat_object_count':len(group_audit)*3,'actual_saved_camera_seed_records':0,
 'actual_saved_per_seed_metric_arrays':0,'actual_saved_raw_gradient_tensors':0,
 'missing_camera_seed_observation_keys':{'representation':'Cartesian domain, not synthesized observations','states':list(states),'camera_index':'0..7','G':[16,8,4,1],'p':[.25,.4],'seed':'0..31','count':18432},
 'execution_inference':'All nine logs report rc=0 and SAVED, cams=8; source loops every combination. 18432 is inferred execution count, not an independently countable raw record count.',
 'deduplication_effect':'No supported rule yields 9216. Deduplicating same-shape masks gives 768 unique spatial masks, not 9216 gradient observations.',
 'effective_sample_unit':'32 seed clusters per condition; same-size cameras and states reuse masks; five parameter groups are not independent experiments.',
 'camera_identity':cameras,'camera_split':'WV0 first 8 of 24 subsampled cameras; no train/eval filter in loader. Historical membership in train/test split remains unverified.',
 'missing':['per-camera/seed metrics and gradients','historical mask file hashes','dense gradient norms/tensors per camera','real per-camera intersection arrays','matched-source kernel and F+B timings','historical renderer source/binary SHA pair'],
 'statistics_recomputable':['means from saved group summary means','macro group means','ratios of means','Pareto of saved means against structural proxies','deterministic mask and active-pixel counts'],
 'statistics_not_recomputable':['raw gradients','seed-paired covariance/confidence intervals','per-camera heterogeneity','exact fixed-camera mask variance','historical per-seed completeness index']})
g4=[]
for sid,s in states.items():
 for p in [.25,.4]:
  a=s['conditions'][f'G4_p{p}'];b=s['conditions'][f'G16_p{p}']
  g4.append({'state':sid,'p':p,'ratio':mean_metric(a,'nmse')/mean_metric(b,'nmse'),'cos_delta':mean_metric(a,'cos')-mean_metric(b,'cos')})
g4cost=[r['ratio_to_same_p_G16_proxy'] for r in absolute if r['G']==4]
g4eq=[r['model_predicted_ratio_vs_G16'] for r in equal if r['G']==4]
same_count=sum(r['frontier_same_p'] for r in pareto)
joint_count=sum(r['frontier_all_saved_p'] for r in pareto)
g4quarter=sum(not r['frontier_all_saved_p'] for r in pareto if r['G']==4 and r['p']==.25)
verdict={'schema':'WV0_CODEX_READINESS.audit.v1','verdict':'BLOCKED_RAW_EVIDENCE',
 'prior_verdict':'GO_CODEX_HIER_EXECUTION','prior_GO_accepted_as_runtime_evidence':False,
 'evidence_class':'STRUCTURAL_ORACLE_ONLY','same_pixel_budget':{'saved_summary_G4_better':18,'total_state_budget_pairs':18,'at_least_30pct_lower':sum(x['ratio']<=.7 for x in g4),'G4_G16_NMSE_ratio_range':[min(x['ratio'] for x in g4),max(x['ratio'] for x in g4)],'paired_statistical_significance':'BLOCKED_NO_RAW'},
 'same_absolute_work':{'measured':False,'G4_G16_structural_proxy_ratio_range':[min(g4cost),max(g4cost)],'equal_proxy_cost_model_NMSE_ratio_range':[min(g4eq),max(g4eq)],'model_is_extrapolation':True,'model_invalid_as_runtime_result':True},
 'timing_evidence':'NO_MATCHED_G4_OR_HIERARCHICAL_KERNEL_OR_F_B_TIMINGS',
 'hierarchy':'PLAUSIBLE_MECHANISM_UNVALIDATED; nearly all parents activate under iid fine sampling',
 'allowed_next_action':'Provide raw evidence, or consider separately authorized minimal instrumented replay after reviewing this audit. No CUDA modification or GPU run performed by this audit.',
 'minimum_remeasurement_proposal':{'purpose':'small provenance and repeatability pilot, not replacement for the 9-state claim','one_high_N_existing_checkpoint':True,'cameras':2,'granularities':[16,4],'budgets':[.25,.4],'seeds':32,'sparse_gradients':256,'dense_reference_gradients':2,'training_steps':0,'requested_evidence':['per-camera/seed/group nmse, cosine, relative L2, dense/estimate norm and dot','pixel/tile mask bits and SHA','per-camera work counts for G16/G8/G4/G1 without claiming unimplemented kernel time','exact source/binary/input hashes','fixed-camera online mean/variance accumulators with stated retention'],'automatic_execution':False},
 'full_WV0_reconstruction_if_ever_needed':'18432 sparse estimates and 72 dense references, separately authorized; not automatically launched.',
 'GPU_compute_seconds_this_audit':0,'CUDA_source_edits':0,'new_training_steps':0}
dump('WV0_CODEX_READINESS.json',verdict)
camera_lines=[]
for c in cameras:camera_lines.append(f"| {c['scene']} | "+', '.join(str(v['id']) for v in c['selected_first_8'])+' | '+str(next(s['cam_hw'] for s in states.values() if s['scene']==c['scene']))+' |')
stat='''# WV0 统计与原始证据审计

**VERDICT：BLOCKED_RAW_EVIDENCE。** 本文区分直接读取的事实、从汇总复算的结果和数学推导。未复算不存在的原始梯度。

## OBSERVED — 数据恢复

读取 MX `/dev/shm/wv0_out/` 的完整输出，并校验原交付 manifest：94/94 匹配；9 个 checkpoint SHA 均匹配。指定 home/pool 深度 9 搜索只找到本轮已保存的副本，原始结果位于易失的 `/dev/shm`。归档 `wv0_deliverables.tgz` 含 95 个文件（含 manifest），其中 RAW 的 SHA 与解包版本相同；数据归档只含相机/图像。Windows 的 95 个交付文件及归档也编入清单。哈希相同的副本不重复计为样本。筛选 shell history 未找到 WV0 命令；保留的 launcher 和执行日志则可读。

关键证据链：`raw/evidence/wv0_live_scripts/_wv0_campaign_state.py:115` 将 8 个 camera 指标平均；`:142` 定义 `stats()`；`:166–173` 只保存统计字典；`:181` 保存 `.pt`。`_wv0_addraw.py` 将这些字典改名为 `*_per_mask`，没有还原逐 seed 数组。

## OBSERVED / DERIVED — 样本量与有效样本单位

9×8×4×2×32 = 18,432 是源码与日志维度所暗示的 camera-level 估计数。32 个 seed 的每个指标先平均 8 cameras，因此若保留，应有 2,304 个 camera-averaged seed 记录。实际仅保留 72 个 state-condition 汇总、360 个 group 汇总、1,080 个 metric-stat 字典；逐 camera/seed 行为 0，逐 seed 数组为 0，完整梯度张量为 0。

9,216 没有对应的去重或聚合规则。所有 9 个日志均有 cams=8、rc=0 和 SAVED，未发现减半相机的证据，但没有逐调用索引可独立确认缺失组合。`n=32` 是保存的汇总样本数，不能乘以 5 组或 8 cameras 冒充独立重复。

每个 camera 都重置同一 CPU seed；相同尺寸的 camera 使用同一空间 mask。跨 checkpoint 也复用种子。应以 seed 为配对簇，保留 camera 层级，不把同一 mask 的多视角响应当独立随机 mask。跨 G 或 p 的 RNG 序列也复用，配对协方差需要原始记录。`numpy.std` 默认 ddof=0。

## OBSERVED — 相机与参数语义

本机准备脚本从完整列表的 `floor(i*n/24)`（i=0..23）取 24 个视角；MX runner 使用其中前 8 个。相机 JSON 没有 train/test 字段，WV0 loader 不执行分割。当前通用 GTDataset 加载所有匹配视角，也不能据此证明历史 checkpoint 的独立 eval 约定。room 历史 provenance 的 trainer SHA 与当前 trainer 不同；garden/bicycle 对应目录没有 provenance.json。**这些视角不是已验证的 held-out 评估集**，本轮只解释冻结梯度 oracle。

| scene | WV0 camera IDs | [H,W] |
|---|---|---|
'''+ '\n'.join(camera_lines)+'''

保留完整相机 JSON、原图路径/SHA、checkpoint SHA。加载器使用 c2w 的逆矩阵作为 w2c，K 主点设为 W/2,H/2，双线性 resize，FP32 RGB/255；tile_size=16、packed=False、SH=3、eps2d=.3、radius_clip=0。梯度对 activated means、normalized quats、exp(scales)、sigmoid(opacity)、SH colors 求导，并非原训练器所有未激活参数的链式梯度；不能把这五组统计直接当作原训练优化器更新效果。

## OBSERVED / DERIVED — 损失与 HT

`_wv0_grad.py:45`：L = Σ_pixels,RGB (r−y)²；`:63`：Lhat = Σ M_tile/p · (r−y)²。是平方误差总和，不是已除以 3HW 的 MSE。tile 由 `torch.rand < p` 独立 Bernoulli 产生，没有固定个数抽样，没有按实际选中像素数再归一化；边缘 tile 中的每个有效像素仍以 p 入选。

对冻结 pixel-gradient a_x，E[M_tile/p]=1，因此 E[Σ M_tile/p a_x]=Σa_x。若目标定义为 per-channel MSE，应在 dense 与 sparse 同除以固定 3HW。共同标量不改变同 camera 的 NMSE/cosine/relL2，但绝对方差乘以其平方。该线性可加证明不覆盖 DSSIM 或自适应选择，也不替代 CUDA backward 正确性测试。

NMSE = ||ghat−g||²/(||g||²+1e−12)；relative L2 = ||ghat−g||/(||g||+1e−12)；cosine = dot(g,ghat)/(||g||||ghat||+1e−12)。各组先单独计算，再对 geometry 三组等权 macro，不能改写成拼接向量指标。

## RECOMPUTED — 汇总所支持的结论

重新计算保存的均值：G4 在 18/18 个 state-budget 条件的 geometry NMSE 低于 G16，16/18 个达到至少 30% 降低；G4/G16 范围 0.414947–0.707107。cosine 在所有条件改善；原 gate 使用 Δ(1−cos)<0，符号正确。该结果是**汇总层面的描述性复算**，不是独立 raw 复现或新置信区间。

报告的 Bias 列错误引用 `geom_rel_l2_mean`（单 mask 误差）。真正保存的 per-group `bias_rel_l2` 是有限 32-seed 的 mean-gradient 对 camera-mean dense gradient 的相对误差；geometry macro 范围 0.0383417–0.2372444。两者均是相对范数单位，不能把 1.3 写成 1.3%。零均值估计器在有限样本下也会有非零 mean-gradient 误差。

保存的 variance = E_camera,mask ||ghat||² − ||E_camera,mask ghat||²，混入 camera 间梯度差异，不等于固定 camera 的 mask variance；加之缺少向量及每 camera 累积量，无法从这些标量恢复条件方差或独立评估 bias 的置信区间。

## RECOMPUTED — Mask 的独立可重现部分

在 MX 上用相同 torch 2.4.1 CPU generator、原源码和 seeds 重建 768 个唯一空间 mask，展开为 2,304 个 state-condition-seed 项，所有 72 个保存的 selected_pixels 均值完全匹配。packed mask bits 和 SHA 已保存。它们明确标为现在重建的 mask，不是原始运行记录。像素边界为 540×810/813/833，已包含非 16 整除尺寸。没有运行 GPU gradient HT/NaN/forward/backward 正确性实验。

## BLOCKED — 不能从现存结果回答的统计问题

不能重算原始梯度、逐 camera/seed 指标、配对差异分布、可靠配对 CI、camera 异质性、固定 camera mask 方差或原始 mask 文件一致性。文件名 RAW、32 的 n 字段以及 SHA 校验只能证明保存文件的身份与一致性，不能证明不存在的记录。
'''
textfile('WV0_STATISTICS_AUDIT.md',stat)
cost='''# WV0 分层成本模型与绝对工作量审计

**VERDICT / evidence class：STRUCTURAL_ORACLE_ONLY；没有 G4/H16→4 kernel 或 F+B 实测。**

## OBSERVED — 成本计数器的范围与误差

原 1.4M / 4.0M / 13.2M / 180M 对应 room iter5000 的 camera 0，精确保存值为 1,424,660 / 4,021,594 / 13,162,206 / 180,296,285。它不是全部场景均值。每个 state 只保存 camera 0 的 bbox 交数，而 gradient 表是 8 cameras 平均，成本与误差轴不完全配对。

源码 `compute_work_counters` 用 floor 上界、闭区间、clip(tw−1)；真实 CUDA `IntersectTile.cu` 用 ceil 上界、半开区间、clip(tw)。在 64×64、G16、mean=(8,8)、radii=(8,8) 的公式例子，旧代理计 4，CUDA 区间规则计 1。这是边界公式反例，不是 synthetic 场景性能结果。原始 projected tensors / emitted IDs 没保留，不能量化真实场景偏差或称其为 physical isect_count。

## DERIVED — 绝对计数与 payload

对保存的 bbox 代理 I_G，均匀 tile Bernoulli 的预期选中代理为 p·I_G。每 G 自己归一化得到 p，并不代表 I_G 一样。G4/G16 的比值为 4.657154–9.238840；因此 p=.25 时 G4 相对 G16-dense 代理为 1.164289–2.309710，p=.4 时为 1.862862–3.695536。

CUDA 源码为每 entry 存 int64 isect_id 与 int32 flatten_id，即 12 bytes。`WV0_ABSOLUTE_COST.csv` 的 payload = 12pI_G 只是假定代理成立时的一套 key/value 大小，不含双缓冲、排序 workspace、扫描、读写流量或时间。实际 intersections、sort bytes、kernel_ms、F+B_ms 保持 NOT_MEASURED。

## RECOMPUTED — 两个 p 的 Pareto

`WV0_PARETO_RECHECK.csv` 给出每 state、每 p 四个候选的 NMSE–absolute-proxy 与 NMSE–payload frontier，另列允许两种 p 相互比较时的 frontier。12-byte payload 是同一绝对 proxy 的线性换算，所以 frontier 相同。

固定 p 下，降低 G 同时降低误差、提高代理成本，72/72 个点都是非支配的权衡点；“处于 frontier”不表示 G4 最优或免费改进。跨两种预算一起比较时，一部分点会被另一粒度/预算支配；详见 CSV 的 dominated_by 列。由于 cost=cam0、quality=8-camera mean，这仍只是代理 Pareto，不能包装成严格配对的真实 GPU 工作 frontier。NMSE–measured F+B frontier 不存在。

## DERIVED — 相同绝对成本的有限模型

令 a_t 为一个 tile 的全部像素梯度贡献，冻结 state/camera 时，独立 Bernoulli HT 的理论 NMSE 为 `(1−p)/p · K_G`，其中 `K_G=Σ_t ||a_t||²/||g||²`（每参数组分别定义，再做 macro）。从保存的有限样本均值估计 `K_G≈NMSE_G(p)·p/(1−p)`。

在同一代理成本 C 下，`p_G=C/I_G`，故模型 `NMSE_G(C)=K_G·(I_G/C−1)`。以 G16 的 .25 或 .4 成本为基准，推得 G4 NMSE/G16 为 **3.8391–6.3482**，G8 为 **1.8181–2.7605**，G1 为 **16.5749–31.3270**。这些外推点通常远低于已有采样 p，**不是观测结果**，假设还把 camera0 成本用于 camera-mean quality，且 I_G 本身是有边界缺陷的代理。

因此，现有数据不支持“同绝对交数成本下 G4 更好”；这个有限模型反而提示成本惩罚可能超过梯度收益。但其假设和证据缺口不允许据此直接 DROP 分层 B2，更不能换算为 CUDA 时间。

## DERIVED / HYPOTHESIS — 16→4 分层账本

| 成本项 | WV0 实际代码 | 原生 G4 的待测项 | 共享父列表 H16→4 的待测项 |
|---|---|---|---|
| Projection | 每次对全部输入 Gaussian 投影 | 通常仍需全部 N | 通常仍需全部 N；细 mask 不自动跳过 |
| Intersection/binning | 始终 G16 dense；G4 只有离线代理 | G4 发射/压缩真实 pair | 仅一套 G16 parent ranges；若父节点 active 才发射，仍几乎全 active |
| Sorting | dense G16 排序 | G4 key/value 排序 | 父列表排序一次；不得复制 16 份再排 |
| Fine mask / scheduling | CPU rand、像素 mask、CPU→GPU | tile list/packing/launch 成本 | fine child compact/index 与 parent-range 复用成本 |
| Rasterization | 全图渲染后 loss mask | 只有选中 G4 像素/tiles compositing | 只选中 child 的像素 compositing；遍历父候选可能多于原生 G4 |
| Backward VJP | dense forward 图上的 masked loss | 真实跳过未选中 child；精确有符号参数累积 | 同上；保持早停、depth-order、alpha/transmittance、SH/camera |
| Mask / IPW loss | 1/p 加权的 RGB SSE | 采样与固定分母 loss 必须计时 | 同上，不排除 host sync、分配等开销 |

对完整父块，P(active parent)=1−(1−p)^16：.25→0.989977404，.40→0.999717889。边界父块用实际有效 child 数 k：1−(1−p)^k。`raw/evidence/WV1_RECONSTRUCTED_MASKS.csv` 的 observed-now mask counts 包含边界。均匀 child 抽样不能声称父 bin/sort 节约 75%。按粗略结构模型，B2 相对同 p 的 G16 sparse 父 intersection 比约 q/p：3.9599 或 2.4993；它仍可能比 native G4 更少重复 intersections，但不保证总时间更快。

B2 是否有利取决于 blend/backward 节省能否覆盖父遍历、列表管理和调度开销。4×4=16 pixels，小于 warp32；packing、shared range reuse、早停、寄存器、occupancy 和 lane idle 均待测。本轮无 CUDA 修改，也无 profiler 数据。

## BLOCKED / HYPOTHESIS — 时间与训练

没有可用的 G16/G4/H16→4 同源码、同 state、同 camera、同 mask objective 的 kernel/F+B 配对计时。旧训练 phase_timing.json 属于其他方法、轨迹与 runtime，不纳入本比较。

假如将来 B2 每步更慢，time-to-quality 改善至少要求 `steps_B2/steps_B0 < time_B0/time_B2`。当前没有时间比，不能给数值 break-even，更不能把 NMSE 比代入迭代数比。冻结梯度误差不是收敛速度。
'''
textfile('WV0_HIERARCHICAL_COST_MODEL.md',cost)
summary_rows=[]
for sid in states:
 for p in [.25,.4]:
  rows=[r for r in pareto if r['state_id']==sid and r['p']==p]
  rr={r['G']:r for r in rows}
  summary_rows.append(f"| {sid} | {p:.2f} | {rr[16]['nmse_geometry_macro']:.4f} | {rr[4]['nmse_geometry_macro']:.4f} | {rr[4]['nmse_geometry_macro']/rr[16]['nmse_geometry_macro']:.4f} | {rr[16]['expected_absolute_intersection_proxy']/1e6:.4f} | {rr[4]['expected_absolute_intersection_proxy']/1e6:.4f} |")
so=pre['/dev/shm/txext/gsplat_cuda/gsplat_cuda.so']['sha256']
report=f'''# WV-HIER-ORACLE0-MX 原始证据恢复与科学审计

审计日期：2026-10-08（Asia/Shanghai）。**VERDICT：BLOCKED_RAW_EVIDENCE。**

**回答：现存汇总支持 G4 在相同像素采样预算下有更好的冻结梯度估计质量；还没有足够的可独立复核证据，将历史 GO 升级为真实分层 CUDA 执行的性能依据。** 分层共享排序是合理的待验证机制，但本轮不晋级 READY_FOR_CODEX_FEASIBILITY 或 CONDITIONAL_CODEX_GO。缺失逐样本记录、绝对成本并未实测，无法给统计显著性或真实时间优势。也没有依据把证据缺口直接判成算法 DROP。

## OBSERVED — 本轮范围与完整性

- 在 MX `bms-39468022-001` 执行只读搜索、哈希检查、CPU mask 重建及统计/成本审计。
- 新目录：`{REMOTE}`。原始 `/dev/shm/wv0_out/` 和既有源码/环境/checkpoint 均保留。
- 指定 home/pool 深度 9 搜索、原始 `/dev/shm`、两个 MX 归档、本机交付文件及归档均检查；manifest 和源码/日志给出一致的“仅保存汇总”证据链。
- 原 manifest **94/94 匹配**，checkpoint **9/9 SHA 匹配**。这验证文件身份，不补足不存在的 raw arrays。
- 本轮 GPU compute **0 秒**，CUDA 修改 **0**，编译 **0**，训练 **0 步**；没有 3K/30K 或小规模 replay。原 WV0 日志累计 {cost_ledger['sum_process_wall_seconds']:,.1f} 单 GPU 进程墙钟秒（{cost_ledger['sum_process_wall_hours']:.4f} 小时），包含 CPU/I/O，不是实测 GPU 活跃秒或账单。GPU 金额未知。

## OBSERVED / RECOMPUTED — 关键裁定

| 项目 | 裁定 | 证据和边界 |
|---|---|---|
| 9,216 样本 | CORRECTION / BLOCKED | 源码循环暗示 18,432 camera-level 估计；缺原始索引，不能实数调用 |
| RAW 文件 | BLOCKED | `*_per_mask` 是统计字典；9 个原 `.pt` 同样没有逐 seed 数组 |
| 保存的数据量 | OBSERVED | 72 条 condition、360 条 group、1,080 个 metric-stat 对象；逐 camera/seed 原始行=0 |
| 损失与 HT | 源码数学审计通过，标签需纠正 | RGB SSE 非 MSE；固定 p 的 tile Bernoulli 1/p 权重；不等于 CUDA backward 验证 |
| Bias | CORRECTION | 报告误用单 mask relative error；真正 mean-gradient error 的 geometry macro 为 .03834–.23724 |
| Cosine | PASS_SOURCE_AUDIT | 相似度增加，1−cos 减少；符号一致 |
| Variance / ESS | CORRECTION | 混合 camera/mask 方差；同 seed 跨 camera/state 复用；32 seed clusters |
| G4 同像素预算 | RECOMPUTED | 汇总 18/18 更低 NMSE，16/18 至少降低 30%；配对显著性 BLOCKED |
| G4 同绝对结构预算 | BLOCKED_MEASUREMENT | 没有匹配预算观察；有限模型预测更差，但依赖代理/外推，不能当实测 |
| G4 runtime / F+B | STRUCTURAL_ORACLE_ONLY | 没有对应实现与有效实测；所有时间字段 NOT_MEASURED |
| Hierarchical 16→4 | HYPOTHESIS | 可共享父排序，但 .25 抽样约 99% 完整父块仍激活 |

## RECOMPUTED / DERIVED — 18 个状态预算比较

下面质量是保存的 8-camera geometry macro 均值；成本是 camera0 的 bbox 代理乘 p，**不是实测 CUDA intersections**。

| state | p | NMSE G16 | NMSE G4 | G4/G16 NMSE | E[G16 proxy] M | E[G4 proxy] M |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(summary_rows)+f'''

原生 G4 在相同 p 下交数代理是 G16 的 **{min(g4cost):.4f}–{max(g4cost):.4f} 倍**。固定 p 的四个粒度呈误差/成本权衡，{same_count}/72 点都在代理非支配集；放开两个已有 p 一起比较时仅 {joint_count}/72 非支配，其中 G4 p=.25 在 {g4quarter}/9 个状态被较便宜候选支配。完整各粒度数据及支配者见 `WV0_PARETO_RECHECK.csv`。

相同绝对代理成本下，Bernoulli 理论配合有限样本 K_G 估计得到 G4/G16 NMSE **{min(g4eq):.4f}–{max(g4eq):.4f}**。该值是 **DERIVED 外推**，不能冒充实验；更不能换成训练时间。主要限制：成本 camera0 与质量 camera-mean 不完全匹配、bbox 边界公式不同于 CUDA、外推到未测 p。详见成本模型。

## OBSERVED — 精确源码、二进制与相机身份

- 观察到项目 `master`，HEAD `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`；已有三个 tracked 修改未触碰。
- 实际 gsplat 来源是 user-site 1.5.3，不是该 repo 的 Git tree；保存 1,729 个源码/头文件等文件的清单。当前树 SHA256：`{source['tree_sha256']}`。
- 现有 `gsplat_cuda.so`：`{so}`，23,069,312 bytes，原路径 `/dev/shm/txext/gsplat_cuda/gsplat_cuda.so`。本轮只复制/哈希，没有加载执行或重编译。
- Python 3.10.19 / torch 2.4.1+cu124 / CUDA compiler 12.4.131 / C++11 ABI=0 / driver 595.71.05。未混用其他 HiGS/cu128 环境。
- 旧 WV0 没保存历史 source/binary SHA 对；当前哈希不是对历史运行的追溯认证。
- camera IDs、原图 SHA、w2c/K、边界尺寸及 train/eval 限制见 `WV0_STATISTICS_AUDIT.md` 和 `raw/evidence/camera_image_identity.json`。
- 768 个唯一 mask 是本次 CPU 重建，72/72 selected-pixel 均值匹配；不是历史 mask 文件。完整 bits/hash 在证据目录中。

## BLOCKED — 最小补证/补测清单（仅建议，未执行）

首选提供旧逐 camera/seed 指标、原始 projected/isect 数组和历史源码/二进制身份。当前所有已定位副本及归档都没有这些记录，不能从分位数恢复数据。

若另行批准新的取证 replay，先限制为 **1 个现有高 N checkpoint × 2 cameras × G16/G4 × 两个 p × 32 seeds = 256 次 sparse gradient**，加 2 个 dense reference；无 optimizer、无训练。保存逐 camera/seed/group 指标、dense/estimate norm 和 dot、mask bits/hash、全部输入 SHA、源码/.so SHA，以及固定 camera 的在线梯度均值/方差累积量。需要独立验证 bias 时应明确保留足够的向量累积量或可重复输入，不能仅靠标量 summary。

结构成本最低补证是在相同投影上取得 G16/G8/G4/G1 的真实 per-tile counts（可分块计数，避免盲目物化 G1 的大排序数组），并把每 camera 的成本与它自己的误差配对。此项不自动假设存在相应 sparse CUDA renderer，也不产生 G4 F+B 速度结论。

这个 256 次 pilot 只能检查取证链与初步重复性，不能替代原 9-state 结论。若要完整重建原统计，需另行批准 18,432 次 sparse 与 72 次 dense，并先评估成本；本轮不启动。真实 native G4 / hierarchical CUDA 比较仍要在以后单独通过代码正确性、跳过工作证据和 paired timing 门槛。

## VERDICT — 为什么是 BLOCKED，为什么没有 DROP

保留下来的描述性信号一致，HT 公式也成立；但关键 raw 数据从保存阶段就被丢弃，跨粒度绝对成本的旧 GO 理由无效，时间证据为空。故当前实现决策为 **BLOCKED_RAW_EVIDENCE**。共享父列表有减少 native G4 重复 bin/sort 的机制依据，尚无证据证明总收益；不能据代理外推判 DROP，也不能据历史 GO 开始实现或训练。

## 交付与复现

八项指定文件全部生成：`WV0_RAW_INVENTORY.json`、`WV0_DATA_COMPLETENESS.json`、`WV0_STATISTICS_AUDIT.md`、`WV0_ABSOLUTE_COST.csv`、`WV0_PARETO_RECHECK.csv`、`WV0_HIERARCHICAL_COST_MODEL.md`、`WV0_CODEX_READINESS.json`、`WV0_AUDIT_REPORT.md`，另有 manifest、raw/、commands/、logs/、gpu_snapshots/。

`commands/05_recovery_search.sh` 是保存的只读搜索及证据复制命令；`commands/03_cpu_oracle_audit.py` 是此前 mask 审计命令（内部输出路径保留原样）；`commands/build_audit.py` 从本审计目录的证据复算八项结果。`SHA256_MANIFEST.json` 校验全部交付文件，不包含自身和外部 ZIP，外部 ZIP 单列 SHA。历史证据文件一律不覆盖。
'''
textfile('WV0_AUDIT_REPORT.md',report)
dump('RUN_STATUS.json',{'status':'AUDIT_COMPLETE','readiness':'BLOCKED_RAW_EVIDENCE','scope':'LATEST_USER_RAW_RECOVERY_AND_SCIENTIFIC_AUDIT_ONLY','remote_directory':REMOTE,'new_cuda_edits':0,'new_GPU_compute_seconds':0,'new_training_steps':0,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps({'same_p_frontier':same_count,'combined_frontier':joint_count,'G4_quarter_dominated_states':g4quarter,'G4_improved':len(g4),'G4_30percent_improved':sum(x['ratio']<=.7 for x in g4),'verdict':verdict['verdict']},indent=2))
