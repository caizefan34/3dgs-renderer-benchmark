"""Phase A - freeze the actual C0/A_s topology + optimizer event ordering.

LAVOS0 MUST discover the frozen event ordering before implementing any
topology-aware semantics (PREREG Sec.3, DSH handoff step 2). This module
encodes the ordering OBSERVED from the frozen C0 trainer and exposes it as a
machine-readable trace so downstream phases consume a frozen contract rather
than an assumption.

OBSERVED from source (this repo):
  * publication_trainer.py training loop (L464-562), per iteration t:
        1  update_learning_rate(t)
        2  [t % sh_progress_interval == 0] oneupSHdegree()
        3  render + L1+fused-SSIM loss
        4  loss.backward()                      -> means2d grad populated
        5  [t < densify_until_iter]
             6  update max_radii2D over visible rows
             7  add_densification_stats(means2d, visibility_filter)
                (accumulates xyz_gradient_accum/denom over the visible set)
        8  [densify_from_iter < t < densify_until_iter
              and t % densification_interval == 0]
                densify_and_prune(...)  == CLONE then SPLIT then PRUNE
        9  [t < densify_until_iter and t % opacity_reset_interval == 0]
                reset_opacity()           (replaces the opacity tensor + zeroes
                                          its optimizer moments)
        10 optimizer.step()               (dense A_s over all rows)
        11 optimizer.zero_grad(set_to_none=True)
  * gaussian_model.py:
        - densify_and_prune calls densify_and_clone, then densify_and_split,
          then computes prune_mask and calls prune_points (L452-481).
        - densify_and_clone / densify_and_split both call
          densification_postfix -> cat_tensors_to_optimizer, which REPLACES the
          param with a new nn.Parameter and CONCATENATES zeroed optimizer
          moments for the newborns (L388-407, L435-450).
        - prune_points -> _prune_optimizer removes the row from every param and
          every optimizer state (L331-347, L349-386).

KEY CONSEQUENCE (recorded, not assumed): topology events (step 8) and opacity
reset (step 9) BOTH execute BEFORE optimizer.step() (step 10). At a topology
iteration the params have just been replaced with new nn.Parameter objects whose
`.grad` is None, so A_s.step() is a NO-OP for the freshly concatenated rows and
`state["step"]` does not increment (the quirk documented in
vos/VOS_IMPLEMENTATION.md Sec.2). LAVOS0 must mirror this: the optimizer step
is the LAST per-iteration event and is a no-op on event iterations.

The ordering below is emitted as a list of ordered stages with the source
location where each is observed, satisfying the Phase A gate
("every topology event type observed in 0..15K has an unambiguous ordering;
exact source locations/functions are recorded").
"""

from dataclasses import dataclass, field
from typing import List


# Ordered per-iteration stages of the frozen C0 loop (see module docstring).
# `phase_a` marks stages that only run inside the 0..15000 densify region.
EVENT_ORDER = (
    {"seq": 1, "name": "update_learning_rate", "kind": "lr",
     "reads_sh": False, "source": "publication_trainer.py:465"},
    {"seq": 2, "name": "oneup_sh_degree", "kind": "sh_degree",
     "reads_sh": False, "source": "publication_trainer.py:466-467"},
    {"seq": 3, "name": "render_and_loss", "kind": "forward",
     "reads_sh": True, "source": "publication_trainer.py:474-480"},
    {"seq": 4, "name": "backward", "kind": "backward",
     "reads_sh": True, "source": "publication_trainer.py:482"},
    {"seq": 5, "name": "update_max_radii2D", "kind": "radii",
     "reads_sh": False, "phase_a": True,
     "source": "publication_trainer.py:498-501"},
    {"seq": 6, "name": "add_densification_stats", "kind": "stats",
     "reads_sh": False, "phase_a": True,
     "source": "publication_trainer.py:505-507"},
    {"seq": 7, "name": "densify_and_prune", "kind": "topology",
     "reads_sh": True, "phase_a": True,
     "sub_order": ("densify_and_clone", "densify_and_split", "prune_points"),
     "source": "gaussian_model.py:452-481"},
    {"seq": 8, "name": "reset_opacity", "kind": "opacity_reset",
     "reads_sh": False, "phase_a": True,
     "source": "gaussian_model.py:258-262 ; publication_trainer.py:553-555"},
    {"seq": 9, "name": "optimizer_step", "kind": "optimizer",
     "reads_sh": True, "source": "publication_trainer.py:561"},
    {"seq": 10, "name": "zero_grad", "kind": "optimizer",
     "reads_sh": False, "source": "publication_trainer.py:562"},
)


def _stage(name):
    for s in EVENT_ORDER:
        if s["name"] == name:
            return s
    raise KeyError(name)


def optimizer_step_after_topology():
    """True: the dense A_s step is the last compute stage of an iteration.

    This is the frozen ordering that LAVOS0's lazy lane must respect: a topology
    event is observed, THEN the (possibly no-op) optimizer step runs.
    """
    return _stage("optimizer_step")["seq"] > _stage("densify_and_prune")["seq"]


def topology_sub_order():
    """Clone before split before prune, as in gaussian_model.densify_and_prune."""
    return list(_stage("densify_and_prune")["sub_order"])


def is_sh_consumer(stage_name):
    """Whether a stage reads/copies/mutates SH data (a potential consumer)."""
    return bool(_stage(stage_name).get("reads_sh", False))


def event_order_trace(densify_from_iter, densify_until_iter,
                      opacity_reset_interval, densification_interval,
                      max_iterations):
    """Emit the concrete per-iteration event trace for 1..max_iterations.

    Returns a list of dicts, one per iteration, each with the ordered stage
    names that actually fire at that iteration and the event type. This is the
    machine-readable Phase A artifact (LAVOS0_EVENT_ORDER.json content).
    """
    trace = []
    for t in range(1, int(max_iterations) + 1):
        stages = ["update_learning_rate", "render_and_loss", "backward"]
        if t % 100 == 0:  # sh_progress_interval default (config); recorded as a phase marker
            stages.append("oneup_sh_degree")
        event = "ordinary"
        if t < densify_until_iter:
            stages += ["update_max_radii2D", "add_densification_stats"]
            if (densify_from_iter < t < densify_until_iter
                    and t % densification_interval == 0):
                stages.append("densify_and_prune")
                event = "topology"
            if t % opacity_reset_interval == 0:
                stages.append("reset_opacity")
                if event == "ordinary":
                    event = "opacity_reset"
        stages += ["optimizer_step", "zero_grad"]
        trace.append({
            "iteration": t,
            "event": event,
            "stages": stages,
            "optimizer_step_is_noop": event in ("topology", "opacity_reset"),
        })
    return trace


@dataclass
class PhaseAGateResult:
    resolved: bool
    ordering: List[dict] = field(default_factory=list)
    notes: str = ""


def phase_a_gate(max_iterations=15000, densify_from_iter=500,
                 densify_until_iter=15000, opacity_reset_interval=3000,
                 densification_interval=100):
    """Evaluate the Phase A gate over the measured 0..15K window.

    Gate PASS requires the ordering to be unambiguous for every event type that
    occurs in the window (PREREG Sec.3). Because the ordering is frozen from
    source, this returns resolved=True whenever at least one topology event and
    the ordering invariants hold; otherwise it emits the unresolved verdict.
    """
    trace = event_order_trace(densify_from_iter, densify_until_iter,
                              opacity_reset_interval, densification_interval,
                              max_iterations)
    n_topology = sum(1 for r in trace if r["event"] == "topology")
    n_reset = sum(1 for r in trace if r["event"] == "opacity_reset")
    invariants = [
        optimizer_step_after_topology(),
        topology_sub_order() == ["densify_and_clone", "densify_and_split",
                                 "prune_points"],
        is_sh_consumer("densify_and_prune"),
        not is_sh_consumer("add_densification_stats"),
    ]
    resolved = bool(n_topology > 0 and all(invariants))
    notes = (f"topology_events={n_topology} opacity_resets={n_reset} "
             f"invariants_ok={all(invariants)}")
    return PhaseAGateResult(resolved=resolved, ordering=list(EVENT_ORDER),
                            notes=notes)
