#!/usr/bin/env python3
"""A-F causal replay from one exact post-599 state in isolated processes.
Status: IMPLEMENTED; MX execution NOT_EXECUTED locally. No production fix.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from R21_STATE import load_state, write_json, sha256, digest
from R21_COMPARE_GAUSSIAN_STATE import compare_states

HERE=Path(__file__).resolve().parent


def replay_inputs(state):
    """All common inputs BEFORE the named consumer intervention.

    Preserve row order here: RNG samples are assigned in selected-parent order.
    State comparison can align reordered IDs, but a same-RNG causal replay must
    not silently relabel the random samples by accepting a permutation.
    """
    scalar_keys=("iteration","phase","scene","seed","recipe","scene_extent",
                 "active_sh_degree","percent_dense","next_native_id")
    tensor_keys=("camera","stable_ids","native_ids","native_stats_before")
    result={key:digest(state[key]) for key in scalar_keys+tensor_keys}
    for key in ("input_raw_digest","input_optimizer_digest","input_rng_digest"):
        result[key]=state[key]
    identity=state.get("source_identity",{})
    context_keys=("model_sha256","torch","cuda","device","deterministic_algorithms",
                  "cudnn_benchmark","cudnn_deterministic","cudnn_allow_tf32","matmul_allow_tf32")
    result["numeric_context"]=digest({key:identity.get(key) for key in context_keys})
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",required=True,type=Path,help="One common post_step checkpoint, normally iteration 599")
    ap.add_argument("--reference-root",required=True,type=Path)
    ap.add_argument("--candidate-root",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--repo",type=Path,default=HERE.parents[1])
    ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--history-reference",type=Path,help="Optional std trajectory post-599 history for E; otherwise E is a deliberate no-op control")
    ap.add_argument("--plan-only",action="store_true")
    args=ap.parse_args()
    state=load_state(args.state)
    if state["phase"]!="post_step":raise ValueError("Replay requires post_step state")
    step=state["iteration"]+1
    if args.out.exists():raise FileExistsError("Use a new causal output directory")
    args.out.mkdir(parents=True)
    commands={}
    for arm in "ABCDEF":
        backend="std_ll" if arm=="A" else "higs"
        root=args.reference_root if arm=="A" else args.candidate_root
        cmd=[sys.executable,str(HERE/"R21_CAPTURE_FIRST_DIVERGENCE.py"),"--repo",str(args.repo),
             "--project",str(root),"--backend",backend,"--scene",state["scene"],"--seed",str(state["seed"]),
             "--out",str(args.out/arm),"--resume",str(args.state),"--stop-after",str(step),
             "--capture-at",str(step),"--gpu",str(args.gpu),"--arm",arm,"--causal"]
        if arm not in "AB":cmd += ["--donor",str(args.out/"A"/f"iter_{step:06d}_pre_topology.pt")]
        if arm=="E" and args.history_reference:cmd += ["--history-reference",str(args.history_reference)]
        commands[arm]=cmd
    plan={"schema":"r21.causal.v1","status":"NOT_EXECUTED","common_state_sha256":sha256(args.state),
          "iteration":step,"commands":commands,
          "E_history_mode":"native_trajectory_history" if args.history_reference else "same_state_noop_control",
          "production_fix":False}
    write_json(args.out/"CAUSAL_PLAN.json",plan)
    if args.plan_only:return
    for arm,cmd in commands.items():
        with (args.out/f"{arm}.log").open("w",encoding="utf-8") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(args.out/"CAUSAL_STATUS.json",{"status":"FAIL","failed_arm":arm,"exit_code":result.returncode})
            raise SystemExit(result.returncode)
    donor=load_state(args.out/"A"/f"iter_{step:06d}_pre_topology.pt")
    for arm in "BCDEF":
        other=load_state(args.out/arm/f"iter_{step:06d}_pre_topology.pt")
        reference_inputs,arm_inputs=replay_inputs(donor),replay_inputs(other)
        for key in reference_inputs:
            if reference_inputs[key] != arm_inputs[key]:
                write_json(args.out/"CAUSAL_STATUS.json",{"status":"INVALID_CONFOUNDED_INPUT","arm":arm,"field":key})
                raise ValueError(f"Causal arm {arm} differs in {key}")
        write_json(args.out/f"A_vs_{arm}_pre.json",compare_states(donor,other))
        write_json(args.out/f"A_vs_{arm}_post.json",compare_states(
            load_state(args.out/"A"/f"iter_{step:06d}_post_step.pt"),
            load_state(args.out/arm/f"iter_{step:06d}_post_step.pt")))
    write_json(args.out/"CAUSAL_STATUS.json",{"status":"REPLAY_COMPLETE","root_cause":"NOT_ESTABLISHED",
        "interpretation":"Review one-factor contrasts and repeated A/A, B/B controls before assigning cause."})


if __name__=="__main__":main()
