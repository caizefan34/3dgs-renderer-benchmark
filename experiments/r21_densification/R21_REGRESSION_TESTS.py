#!/usr/bin/env python3
"""R21 CPU diagnostic regressions, NOT renderer/MX parity tests.

The actual GaussianModel is loaded from a temporary copy with ONLY literal
device="cuda" changed to device="cpu". Equations, topology and Adam code are
unchanged. This makes provenance and the validation boundary explicit.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from R21_STATE import (RAW, align_ids, candidates, child_ids, cpu, digest, load_state,
                       model_state, remap, restore_model, restore_rng, rng_state,
                       row_digest, save_state, tensor_metrics, write_json, topology_digest)
from R21_INSTRUMENT import Session, install_topology_observers, instrument_trainer
from R21_COMPARE_GAUSSIAN_STATE import compare_journals, compare_states
from R21_ANALYZE_THRESHOLD_CROSSING import analyze

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        path=Path(cls.temp.name)/"model_cpu.py"
        source=(REPO/"baseline/reference_v1/gaussian_model.py").read_text(encoding="utf-8")
        path.write_text(source.replace('device="cuda"','device="cpu"'),encoding="utf-8")
        spec=importlib.util.spec_from_file_location("r21_model_cpu",path)
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)
        cls.Model=cls.module.GaussianModel

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def model(self):
        m=self.Model()
        m._xyz=torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(4,3)/10)
        m._shs=torch.nn.Parameter(torch.ones(4,16,3)*.1)
        m._opacity=torch.nn.Parameter(torch.logit(torch.tensor([.2,.3,.001,.4])))
        m._scaling=torch.nn.Parameter(torch.tensor([[.001]*3,[.02]*3,[.003]*3,[.03]*3]).log())
        m._rotation=torch.nn.Parameter(torch.tensor([[1.,0,0,0]]).repeat(4,1))
        m._gaussian_ids=torch.arange(4);m._next_gaussian_id=4
        m.max_radii2D=torch.arange(4,dtype=torch.float32)+20
        config={"position_lr_init":.00016,"position_lr_final":.0000016,"position_lr_delay_mult":.01,
                "position_lr_max_steps":30000,"feature_lr":.0025,"opacity_lr":.025,
                "scaling_lr":.005,"rotation_lr":.001,"percent_dense":.01}
        m.training_setup(config)
        sum(getattr(m,"_"+k).square().sum() for k in RAW).backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        # Fixed scores: one clone, one split, one opacity prune, one below grow.
        m.xyz_gradient_accum=torch.tensor([[.002],[.002],[0.],[.0001]])
        m.denom=torch.ones(4,1)
        return m

    def observer(self,m,arm="B",donor=None):
        s=Session.__new__(Session)
        s.model=m;s.stable_ids=m._gaussian_ids.clone();s.iteration=600;s.events=[];s.decisions={}
        s.arm=arm;s.donor=donor;s.options={};s.row={}
        install_topology_observers(m,s)
        return s

    def test_01_trainer_anchors_and_no_formula_edits(self):
        source=(REPO/"experiments/r20_true_absgrad/R20_TRAINER.py").read_text(encoding="utf-8")
        modified=instrument_trainer(source,REPO)
        ast.parse(modified)
        for line in ("        loss.backward()","        model.optimizer.step()",
                     "        visibility_filter = (radii > 0).any(dim=-1)",
                     "                max_grad=config.densify_grad_threshold, min_opacity=config.min_opacity,"):
            self.assertIn(line,modified)
        with self.assertRaises(ValueError):
            instrument_trainer(source.replace("        radii = radii_full\n",""),REPO)

    def test_02_lineage_independent_of_other_births(self):
        a=child_ids(torch.tensor([1,9]),600,"split",2)
        b=child_ids(torch.tensor([9]),600,"split",2)
        self.assertTrue(torch.equal(a[torch.tensor([1,3])],b))
        self.assertFalse(torch.equal(b,child_ids(torch.tensor([9]),600,"clone",2)))
        rows1,rows2=align_ids(torch.tensor([3,1,7]),torch.tensor([7,3,1]))
        self.assertTrue(torch.equal(torch.tensor([3,1,7])[rows1],torch.tensor([7,3,1])[rows2]))
        self.assertEqual(row_digest(torch.tensor([2,1]),torch.tensor([[20.],[10.]])),
                         row_digest(torch.tensor([1,2]),torch.tensor([[10.],[20.]])))
        with self.assertRaises(ValueError):remap(torch.ones(2),torch.tensor([1,2]),torch.tensor([1,3]))

    def test_03_visibility_changes_denom_without_absgrad_difference(self):
        a,b=self.model(),self.model()
        for m in (a,b):m.xyz_gradient_accum.zero_();m.denom.zero_()
        proxy=torch.zeros(1,4,2);proxy.absgrad=torch.zeros_like(proxy)
        proxy.absgrad[0,1,0]=.000004
        a.add_densification_stats(proxy,torch.tensor([False,True,False,False]),width=1000,height=500)
        b.add_densification_stats(proxy,torch.tensor([False,True,False,False]),width=1000,height=500)
        proxy.absgrad.zero_()
        a.add_densification_stats(proxy,torch.zeros(4,dtype=torch.bool),width=1000,height=500)
        b.add_densification_stats(proxy,torch.tensor([False,True,False,False]),width=1000,height=500)
        self.assertEqual(float(a.denom[1]),1);self.assertEqual(float(b.denom[1]),2)
        self.assertEqual(float(a.xyz_gradient_accum[1]),float(b.xyz_gradient_accum[1]))
        self.assertAlmostEqual(float(a.xyz_gradient_accum[1]/a.denom[1]),2*float(b.xyz_gradient_accum[1]/b.denom[1]))

    def test_04_radii_magnitudes_do_not_select_clone_split_at_600(self):
        m=self.model();before=candidates(m,.0008,1.,.005,None)
        m.max_radii2D.fill_(100000)
        after=candidates(m,.0008,1.,.005,None)
        for key in ("clone","split","prune_parent_preview"):
            self.assertTrue(torch.equal(before[key],after[key]))

    def test_05_instrumented_actual_topology_preserves_values_rng_and_adam(self):
        a,b=self.model(),self.model();state=rng_state()
        with torch.no_grad():
            restore_rng(state);a.densify_and_prune(.0008,.005,1.,None,torch.ones(4))
            arng=rng_state();s=self.observer(b)
            restore_rng(state);b.densify_and_prune(.0008,.005,1.,None,torch.ones(4))
        self.assertEqual(digest(arng),digest(rng_state()))
        for name in RAW:self.assertTrue(torch.equal(getattr(a,"_"+name),getattr(b,"_"+name)))
        aa,bb=model_state(a,a._gaussian_ids),model_state(b,b._gaussian_ids)
        self.assertEqual(digest(aa["optimizer"]),digest(bb["optimizer"]))
        self.assertEqual(len(s.stable_ids),5)
        self.assertEqual(int(s.decisions["clone"]["mask"].sum()),1)
        self.assertEqual(int(s.decisions["split"]["mask"].sum()),1)
        self.assertEqual(int(s.decisions["prune"]["mask"].sum()),1)

    def test_06_checkpoint_restores_stats_ids_moments_and_next_step(self):
        a=self.model();a._gaussian_ids=torch.tensor([8,9,10,11]);a._next_gaussian_id=12
        state=model_state(a,a._gaussian_ids)
        b=self.model();ids=restore_model(b,state)
        self.assertEqual(digest(state),digest(model_state(b,ids)))
        for m in (a,b):
            sum(getattr(m,"_"+k).square().sum() for k in RAW).backward();m.optimizer.step()
        for key in RAW:self.assertTrue(torch.equal(getattr(a,"_"+key),getattr(b,"_"+key)))

    def test_07_checkpoint_hash_guard_and_safe_load(self):
        m=self.model();s=model_state(m,m._gaussian_ids)
        s.update(schema="r21.gaussian-state.v1",iteration=599,phase="post_step")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.pt";save_state(path,s)
            self.assertEqual(digest(s),digest(load_state(path)))
            with self.assertRaises(FileExistsError):save_state(path,s)
            with path.open("ab") as f:f.write(b"corrupt")
            with self.assertRaises(ValueError):load_state(path)

    def test_08_forced_decisions_only_downstream_control(self):
        a,b=self.model(),self.model();sa=self.observer(a);seed=rng_state()
        with torch.no_grad():
            a.densify_and_prune(.0008,.005,1.,None,torch.ones(4))
            b.xyz_gradient_accum.zero_()
            sb=self.observer(b,"F",{"decisions":sa.decisions})
            restore_rng(seed);b.densify_and_prune(.0008,.005,1.,None,torch.ones(4))
        self.assertTrue(torch.equal(sa.stable_ids,sb.stable_ids))
        for key in RAW:self.assertTrue(torch.equal(getattr(a,"_"+key),getattr(b,"_"+key)))

    def test_09_empty_clone_still_resets_reference_max_radii(self):
        m=self.model();m.xyz_gradient_accum.zero_();m.max_radii2D.fill_(50)
        with torch.no_grad():ev=m.densify_and_prune(.0008,.005,1.,20,torch.ones(4))
        self.assertEqual(ev["pruned_screen"],0)
        self.assertTrue(bool((m.max_radii2D==0).all()))

    def test_10_statistics_interventions_one_factor(self):
        for arm in "BCDE":
            m=self.model();s=Session.__new__(Session)
            s.model=m;s.stable_ids=m._gaussian_ids;s.arm=arm;s.options={};s.row={}
            s.donor={"stable_ids":m._gaussian_ids,"statistics_radii":torch.ones(4,2)*99,
                     "visibility":torch.tensor([False,False,True,True]),
                     "stats_before":{"xyz_gradient_accum":torch.ones(4,1)*7,"denom":torch.ones(4,1)*8}}
            s.iteration=600
            radii=torch.ones(4,2);vis=torch.tensor([True,True,False,False])
            rr,vv=s.statistics_inputs(radii,vis)
            self.assertTrue(torch.equal(rr,s.donor["statistics_radii"] if arm=="C" else radii))
            self.assertTrue(torch.equal(vv,s.donor["visibility"] if arm=="D" else vis))
            if arm=="E":self.assertTrue(bool((m.denom==8).all()))
            else:self.assertTrue(bool((m.denom==1).all()))

    def test_11_metrics_handle_nonfinite_empty_dtype(self):
        self.assertTrue(tensor_metrics(torch.empty(0),torch.empty(0))["equal"])
        self.assertFalse(tensor_metrics(torch.tensor([float("nan")]),torch.ones(1))["close"])
        self.assertFalse(tensor_metrics(torch.ones(1),torch.ones(1,dtype=torch.float64))["compatible"])

    def snapshot(self,m):
        state=model_state(m,m._gaussian_ids)
        state.update(iteration=600,phase="pre_topology",camera={"id":7},
                     candidates=cpu(candidates(m,.0008,1.,.005,None)),visibility=torch.ones(4,dtype=torch.bool))
        return state

    def test_12_threshold_crossing_report(self):
        a,b=self.model(),self.model();b.xyz_gradient_accum[1]=.0007
        report=analyze(self.snapshot(a),self.snapshot(b))
        self.assertEqual(report["mismatch_count"],1)
        self.assertTrue(report["rows"][0]["grow_threshold_crossed"])
        self.assertEqual(report["root_cause"],"NOT_ESTABLISHED")

    def test_13_state_comparison_matches_reordered_ids(self):
        a=self.snapshot(self.model());b=cpu(a);perm=torch.tensor([2,0,3,1])
        b["stable_ids"]=b["stable_ids"][perm]
        for cat in ("raw","raw_grad","stats"):
            b[cat]={k:(v[perm] if v is not None else None) for k,v in b[cat].items()}
        for group in b["optimizer"]["state"].values():
            for key in ("exp_avg","exp_avg_sq"):group[key]=group[key][perm]
        b["visibility"]=b["visibility"][perm]
        for key,v in b["candidates"].items():
            if isinstance(v,torch.Tensor):b["candidates"][key]=v[perm]
        report=compare_states(a,b)
        self.assertTrue(all(v["equal"] for v in report["fields"].values()))

    def test_14_first_numeric_vs_topology_step(self):
        keys=("raw_before","raw_after","radii","projection","visibility","absgrad","raw_grad",
              "stats_before","stats_after","candidates","actual_decisions","stable_ids_after","topology_events","camera")
        a=[{"schema":"r21.step.v2","iteration":i,**{k:"same" for k in keys}} for i in range(1,601)];b=cpu(a)
        b[0]["raw_after"]="different";b[599]["stable_ids_after"]="different"
        with tempfile.TemporaryDirectory() as tmp:
            p,q=Path(tmp)/"a.jsonl",Path(tmp)/"b.jsonl"
            for path,rows in ((p,a),(q,b)):path.write_text("\n".join(json.dumps(r) for r in rows))
            report=compare_journals(p,q)
        self.assertEqual(report["first_parameter_divergence"],1)
        self.assertEqual(report["first_topology_divergence"],600)
        self.assertTrue(report["proves_first_since_initialization"])

    def test_15_capture_resume_lifecycle_with_actual_model(self):
        config=types.SimpleNamespace(densify_grad_threshold=.0008,percent_dense=.01,min_opacity=.005,
            max_screen_size=20,opacity_reset_interval=3000,scene="bicycle",seed=42)
        config.to_dict=lambda:{k:v for k,v in vars(config).items() if k!="to_dict"}
        args=types.SimpleNamespace(backend="std_ll",scene="bicycle",seed=42)
        cam=types.SimpleNamespace(K=torch.eye(3),viewmatrix=torch.eye(4),camera_center=torch.zeros(3),
                                  image_width=4,image_height=4,image_name="fake")
        import numpy as np
        sequence=np.zeros(600,dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            options={"out":str(Path(tmp)/"capture"),"stop_after":599,"capture_at":[599],
                     "resume_every":100,"identity":{"model_sha256":"fixture"}}
            with patch.dict("os.environ",{"R21_OPTIONS":json.dumps(options)}),patch.object(Session,"install_projection_observer"):
                m=self.model();s=Session(m,config,args,None,sequence,1.)
            s.begin(599,0,cam,torch.zeros(4,4,3))
            proxy=torch.zeros(1,4,2,requires_grad=True);proxy.grad=torch.ones_like(proxy);proxy.absgrad=torch.ones_like(proxy)
            rad=torch.ones(1,4,2,dtype=torch.int32)
            s.observe_std({"means2d":proxy,"radii":rad,"depths":torch.ones(1,4),"conics":torch.ones(1,4,3)})
            loss=m._xyz.square().sum();loss.backward();s.after_backward(rad,proxy,loss)
            rr,vv=s.statistics_inputs(rad[0],torch.ones(4,dtype=torch.bool))
            m.add_densification_stats(proxy,vv,width=4,height=4);s.after_statistics();s.before_optimizer()
            m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);s.end();s.finish()
            checkpoint=Path(options["out"])/"iter_000599_post_step.pt"
            before=load_state(Path(options["out"])/"iter_000599_pre_topology.pt")
            self.assertEqual(before["projection_canonical"]["means2d"].shape,(4,2))
            self.assertEqual(before["raw_grad"]["xyz"].shape,(4,3))
            options.update(out=str(Path(tmp)/"resumed"),resume=str(checkpoint),stop_after=600)
            with patch.dict("os.environ",{"R21_OPTIONS":json.dumps(options)}),patch.object(Session,"install_projection_observer"):
                restored=self.model();r=Session(restored,config,args,None,sequence,1.)
            self.assertEqual(r.start_iteration,600)
            self.assertEqual(digest(model_state(m,s.stable_ids)),digest(model_state(restored,r.stable_ids)))

    def test_16_projection_canonicalization_ignores_invalid_slots(self):
        m=self.model();s=Session.__new__(Session);s.model=m;s.stable_ids=m._gaussian_ids
        s.radii=torch.tensor([[[1,1],[0,0],[2,3],[0,0]]],dtype=torch.int32)
        s.projection={"master_rows":torch.arange(4),"means2d":torch.full((1,4,2),float("nan")),
                      "depths":torch.ones(1,4),"conics":torch.ones(1,4,3)}
        s.projection["means2d"][0,0]=1;s.projection["means2d"][0,2]=2
        result=s.canonical_projection()
        self.assertTrue(bool(torch.isfinite(result["means2d"]).all()))
        self.assertTrue(bool((result["means2d"][[1,3]]==0).all()))

    def test_17_causal_cli_builds_six_isolated_same_state_commands(self):
        state=self.snapshot(self.model())
        state.update(schema="r21.gaussian-state.v1",iteration=599,phase="post_step",scene="bicycle",seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/"state.pt";save_state(checkpoint,state)
            out=Path(tmp)/"plan"
            result=subprocess.run([sys.executable,str(HERE/"R21_CAUSAL_REPLAY.py"),"--state",str(checkpoint),
                "--reference-root",str(Path(tmp)/"std"),"--candidate-root",str(Path(tmp)/"higs"),
                "--out",str(out),"--plan-only"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads((out/"CAUSAL_PLAN.json").read_text())
            self.assertEqual(plan["status"],"NOT_EXECUTED")
            self.assertEqual(set(plan["commands"]),set("ABCDEF"))
            for arm,command in plan["commands"].items():
                self.assertEqual(command[command.index("--resume")+1],str(checkpoint))
                self.assertEqual(command[command.index("--stop-after")+1],"600")
                self.assertEqual("--donor" in command,arm in "CDEF")
            self.assertEqual(plan["E_history_mode"],"same_state_noop_control")

    def test_18_causal_replay_rejects_camera_history_and_row_order_confounds(self):
        import R21_CAUSAL_REPLAY as replay
        state=self.snapshot(self.model())
        state.update(schema="r21.gaussian-state.v1",iteration=600,phase="pre_topology",scene="bicycle",seed=42,
            recipe={"threshold":.0008},scene_extent=1.,native_stats_before=cpu(state["stats"]),
            input_raw_digest="same_raw",input_optimizer_digest="same_adam",input_rng_digest="same_rng")
        changes={"camera":lambda x:x["camera"].update(id=999),
                 "native_stats_before":lambda x:x["native_stats_before"]["denom"].add_(1),
                 "stable_ids":lambda x:x.update(stable_ids=x["stable_ids"].flip(0))}
        for name,mutate in changes.items():
            with self.subTest(confound=name),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"input.pt";path.write_bytes(b"fixture")
                altered=cpu(state);mutate(altered)
                def load(filename):
                    filename=Path(filename)
                    if filename==path:return {**state,"iteration":599,"phase":"post_step"}
                    return altered if filename.parent.name=="B" else state
                argv=["replay","--state",str(path),"--reference-root",tmp,"--candidate-root",tmp,
                      "--out",str(Path(tmp)/"out")]
                with patch.object(sys,"argv",argv),patch.object(replay,"load_state",side_effect=load), \
                     patch.object(replay.subprocess,"run",return_value=types.SimpleNamespace(returncode=0)), \
                     patch.object(replay,"compare_states",return_value={}):
                    with self.assertRaisesRegex(ValueError,name):replay.main()

    def test_19_transient_topology_difference_is_not_hidden_by_survivor_set(self):
        keys=("raw_before","raw_after","radii","projection","visibility","absgrad","raw_grad",
              "stats_before","stats_after","candidates","actual_decisions","stable_ids_after","topology_events","camera")
        a={"schema":"r21.step.v2","iteration":600,**{key:"same" for key in keys}};b=cpu(a)
        # A clone born and pruned in one transaction can leave the same survivors.
        b["topology_events"]="birth_then_prune"
        with tempfile.TemporaryDirectory() as tmp:
            p,q=Path(tmp)/"a.jsonl",Path(tmp)/"b.jsonl"
            p.write_text(json.dumps(a));q.write_text(json.dumps(b))
            report=compare_journals(p,q)
        self.assertEqual(report["first_topology_divergence"],600)
        self.assertIsNone(report["first_survivor_set_divergence"])

    def test_20_topology_events_align_lineage_and_detect_birth_then_prune(self):
        parents=torch.tensor([1,5]);children=child_ids(parents,600,"split",2)
        a=[{"kind":"birth","operation":"split","parents":parents,"children":children,"copies":2},
           {"kind":"remove","stable_ids":parents}]
        b=[{"kind":"birth","operation":"split","parents":parents.flip(0),
            "children":children[torch.tensor([1,0,3,2])],"copies":2},
           {"kind":"remove","stable_ids":parents.flip(0)}]
        self.assertEqual(topology_digest(a),topology_digest(b))
        clone=child_ids(torch.tensor([1]),600,"clone")
        transient=[{"kind":"birth","operation":"clone","parents":torch.tensor([1]),"children":clone,"copies":1},
                   {"kind":"remove","stable_ids":clone}]
        self.assertNotEqual(topology_digest([]),topology_digest(transient))


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(HERE/"R21_LOCAL_TEST_RESULTS.json",{"status":"PASS" if result.wasSuccessful() else "FAIL",
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "scope":"CPU diagnostics; actual GaussianModel math with device allocation substitution only",
        "MX_capture":"NOT_EXECUTED","MX_causal_replay":"NOT_EXECUTED","production_fix":"NOT_IMPLEMENTED_PENDING_CAUSAL_EVIDENCE"})
    sys.exit(0 if result.wasSuccessful() else 1)
