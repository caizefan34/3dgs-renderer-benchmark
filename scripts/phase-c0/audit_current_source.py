#!/usr/bin/env python3
"""Trace the actual GaussianModel and optimizer path used by C42--C53."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OFFICIAL_SHA, git, sha256, write_json

MODEL = ROOT / "scripts" / "epic05" / "phase7" / "gaussian_model.py"
PHASES = {
    "C42": "scripts/phase-c42/c42_p2_training_validation_30k.py",
    "C44": "scripts/phase-c44/c44_unified_experiment.py",
    "C45": "scripts/phase-c45/c45_unified_experiment.py",
    "C49": "scripts/phase-c49/grad_filter_experiment.py",
    "C50": "scripts/phase-c50/gradient_predictability.py",
    "C51_early": "scripts/phase-c51/simulated_sparse_backward.py",
    "C51_stage4b": "scripts/phase-c51-stage4b/canonical_training.py",
    "C51_stage5": "scripts/phase-c51-stage5/benchmark.py",
    "C52": "scripts/phase-c52-stage0/benchmark.py",
    "C53_discovery": "scripts/phase-c53-discovery/collect_signals.py",
    "C53_validation": "scripts/phase-c53-validation/collect_actual_work.py",
    "C53_validation2": "scripts/phase-c53-validation2/collect_workload_pairs.py",
}
FUNCTIONS = ("set_sh_degree", "accumulate_positional_gradient", "densification", "prune", "prune_and_reset")


def excerpt(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))


def model_functions() -> dict:
    tree = ast.parse(MODEL.read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            result[node.name] = {
                "file": MODEL.relative_to(ROOT).as_posix(),
                "lines": [node.lineno, node.end_lineno],
                "excerpt": excerpt(MODEL, node.lineno, node.end_lineno),
            }
    return result


def evidence_lines(path: Path) -> list[dict]:
    needles = (
        "from gaussian_model import GaussianModel", "from gsplat import rasterization",
        "optimizer =", "model.densification", "model.prune(",
        "model.prune_and_reset", "model.set_sh_degree",
    )
    return [
        {"line": i, "text": line.strip()}
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if any(needle in line for needle in needles)
    ]


def optimizer_classification(phase: str) -> str:
    if phase == "C42":
        return "global_reset_after_SH_and_topology"
    if phase in {"C44", "C45", "C49", "C50", "C51_early"}:
        return "orphaned_optimizer_after_parameter_replacement"
    return "global_reset_after_topology; SH_parameter_orphaned_after_degree_change"


def main() -> None:
    phase_data = {}
    for phase, relative in PHASES.items():
        path = ROOT / relative
        phase_data[phase] = {
            "training_script": relative,
            "gaussian_model": "scripts/epic05/phase7/gaussian_model.py",
            "optimizer": "torch.optim.Adam",
            "renderer": "gsplat.rasterization",
            "optimizer_topology_behavior": optimizer_classification(phase),
            "source_sha256": sha256(path),
            "evidence": evidence_lines(path),
        }
    payload = {
        "status": "PASS",
        "project_commit": git("rev-parse", "HEAD"),
        "freeze_branch": "audit/pre-c0-current-semantics",
        "model_sha256": sha256(MODEL),
        "model_functions": model_functions(),
        "call_graph": phase_data,
        "important_finding": (
            "C44/C45/C49/C50/early-C51 replace model Parameters without rebinding Adam; "
            "later C51/C52/C53 recreate Adam after topology and therefore reset all state."
        ),
    }
    print(write_json("current_source_audit.json", payload))
    current_sha = payload["project_commit"]
    current_model = "scripts/epic05/phase7/gaussian_model.py"
    official_model = "scene/gaussian_model.py"
    def ev(current_lines: str, official_lines: str, current_excerpt: str, official_excerpt: str) -> dict:
        return {
            "current": {"commit": current_sha, "file": current_model, "function": "GaussianModel", "lines": current_lines, "excerpt": current_excerpt},
            "reference": {"commit": OFFICIAL_SHA, "file": official_model, "function": "GaussianModel", "lines": official_lines, "excerpt": official_excerpt},
        }
    rows = [
        {"operation": "split selection", "current": "avg ||dL/dxyz|| >= threshold; large iff not(all scale axes <= per-event median)", "reference": "padded avg view-space ||dL/dmean2D_xy|| >= threshold AND max(scale) > percent_dense*scene_extent", "match": False, "severity": "HIGH", "potential_effect": "different masks and population allocation", "evidence": ev("152-188", "409-416,471-473", "xyz.grad norm; median-scale partition", "viewspace xy norm; percent_dense*extent")},
        {"operation": "split child count", "current": "2", "reference": "N=2 default", "match": True, "severity": "NONE", "potential_effect": "none", "evidence": ev("226-231", "409,418-428", "two concatenated copies", "repeat(N), default N=2")},
        {"operation": "split parent", "current": "retained; removed=0", "reference": "removed after children are appended", "match": False, "severity": "CRITICAL", "potential_effect": "one extra resident Gaussian per split selection", "evidence": ev("245-266", "430-433", "append and return removed: 0", "densification_postfix then prune selected parents")},
        {"operation": "split child position", "current": "parent + iid N(0,0.0025^2) in world axes", "reference": "parent + R(quaternion)*N(0,diag(activated_scale^2))", "match": False, "severity": "CRITICAL", "potential_effect": "different immediate rendered function and spatial support", "evidence": ev("220-243", "418-423", "torch.randn_like * 0.0025", "normal std=scale, rotated into parent frame")},
        {"operation": "split child scale", "current": "log_scale_child = log_scale_parent - log(2), i.e. scale/2", "reference": "log_scale_child = log(scale_parent/(0.8*N)); N=2 gives scale/1.6", "match": False, "severity": "HIGH", "potential_effect": "current children are 20% smaller per axis than reference", "evidence": ev("222-230", "423-424", "subtract log(2)", "inverse_activation(scale/(0.8*N))")},
        {"operation": "split rotation", "current": "raw quaternion inherited", "reference": "raw quaternion inherited", "match": True, "severity": "NONE", "potential_effect": "none", "evidence": ev("221,228", "421,424", "concatenated parent rotation", "parent rotation repeated N")},
        {"operation": "split opacity", "current": "logit inherited", "reference": "logit inherited", "match": True, "severity": "NONE", "potential_effect": "none", "evidence": ev("223,230", "427", "parent opacity concatenated", "parent opacity repeated N")},
        {"operation": "split SH/features", "current": "all stored SH coefficients inherited", "reference": "DC and rest coefficients inherited", "match": True, "severity": "NONE", "potential_effect": "layout differs but semantic inheritance matches", "evidence": ev("224,231", "425-426", "parent shs concatenated", "parent DC/rest repeated N")},
        {"operation": "clone", "current": "parent retained; child position receives N(0,(0.01*scale)^2) noise; other fields copied", "reference": "parent retained; exact parameter copy including position", "match": False, "severity": "HIGH", "potential_effect": "different immediate function and density placement", "evidence": ev("208-216", "435-450", "cloned_xyz + scale-relative noise", "new_xyz = selected parent xyz")},
        {"operation": "prune", "current": "opacity only in prune(); repeated at densification and reset; no screen/world masks", "reference": "opacity always; after opacity_reset_interval also max_radii2D>20 OR max(scale)>0.1*extent", "match": False, "severity": "CRITICAL", "potential_effect": "different size control and population trajectory", "evidence": ev("269-315", "452-465", "sigmoid(opacity)<threshold", "opacity plus conditional screen/world size")},
        {"operation": "opacity reset", "current": "only survivors in [threshold,10*threshold) set to 2*threshold, and prune runs first", "reference": "all activated opacities clamped to at most 0.01; opacity moments zeroed", "match": False, "severity": "CRITICAL", "potential_effect": "different post-reset quality and pruning trajectory", "evidence": ev("297-315", "258-261,316-329", "near-threshold selective assignment", "min(opacity,0.01) through optimizer replacement")},
        {"operation": "gradient accumulation", "current": "world xyz gradient norm, all rows, normally sampled only on event iteration; one global denominator", "reference": "view-space xy gradient norm for visible rows each iteration; per-row visibility denominator", "match": False, "severity": "CRITICAL", "potential_effect": "densification statistic and thresholds are not comparable", "evidence": ev("152-180", "452-454,471-473", "xyz.grad.norm and _denf_steps", "viewspace grad[:2] and denom[visible]")},
        {"operation": "optimizer topology handling", "current": "phase-dependent: older paths orphan Adam; later C51/C52/C53 reconstruct Adam and globally reset state", "reference": "replace Parameter but slice survivor moments, zero-pad appended moments, preserve scalar step", "match": False, "severity": "CRITICAL", "potential_effect": "frozen/orphaned parameters or loss of Adam trajectory", "evidence": {"current": {"commit": current_sha, "file": "scripts/phase-c51-stage4b/canonical_training.py", "function": "run_experiment", "lines": "334-373", "excerpt": "topology then optimizer = get_optimizer(model)"}, "reference": {"commit": OFFICIAL_SHA, "file": official_model, "function": "_prune_optimizer/cat_tensors_to_optimizer", "lines": "331-386", "excerpt": "slice survivor exp_avg/exp_avg_sq; append zero rows; retain stored_state including step"}}},
        {"operation": "SH progression/storage", "current": "resizes and replaces shs Parameter; many paths do not rebind optimizer", "reference": "allocates max SH once and increments active_sh_degree only", "match": False, "severity": "CRITICAL", "potential_effect": "historical SH parameter can become detached from Adam", "evidence": ev("121-147", "145-147,153-171", "self.shs = nn.Parameter(new_shs)", "active_sh_degree += 1")},
    ]
    print(write_json("semantic_matrix.json", {
        "status": "PASS_STATIC_AUDIT",
        "current_project_commit": current_sha,
        "official_reference_commit": OFFICIAL_SHA,
        "rows": rows,
    }))


if __name__ == "__main__":
    main()
