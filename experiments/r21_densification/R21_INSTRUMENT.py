"""Read-only diagnostics injected into the exact R20 trainer in memory.

Status: IMPLEMENTED; production source files are never edited. CPU tests cover
source anchors, statistics, lineage and checkpoint restoration; MX pending.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
import time
import types
from pathlib import Path

import torch

from R21_STATE import (RAW, SCHEMA, STATS, candidates, child_ids, cpu, digest,
                       model_state, remap, restore_model, restore_rng, save_state,
                       sha256, validate_ids, write_json, load_state, row_digest, rng_state, optimizer_named,
                       topology_digest)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"R20 source anchor must occur exactly once: {old[:95]!r}")
    return source.replace(old, new, 1)


def instrument_trainer(source, repo):
    """Insert detached observation callbacks at actual trainer phase boundaries."""
    source = replace_once(source, 'REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")',
                          f"REPO = Path({str(repo)!r})")
    source = replace_once(source, "    for iteration in range(1, config.iterations + 1):",
        "    r21 = Session(model, config, args, dataset, camera_sequence, scene_extent)\n"
        "    for iteration in range(r21.start_iteration, min(config.iterations, r21.stop_after) + 1):")
    source = replace_once(source, "        cam, gt_image = dataset.get_item(cam_idx)\n",
        "        cam, gt_image = dataset.get_item(cam_idx)\n"
        "        r21.begin(iteration, cam_idx, cam, gt_image)\n")
    source = replace_once(source, '            means2d = meta["means2d"]',
        '            r21.observe_std(meta)\n            means2d = meta["means2d"]')
    source = replace_once(source, '            info = out["densification_info"]',
        '            r21.observe_higs(out)\n            info = out["densification_info"]')
    source = replace_once(source, "        radii = radii_full\n",
        "        r21.after_backward(radii_full, means2d, loss)\n        radii = radii_full\n")
    source = replace_once(source, "        visibility_filter = (radii > 0).any(dim=-1)\n",
        "        visibility_filter = (radii > 0).any(dim=-1)\n"
        "        radii, visibility_filter = r21.statistics_inputs(radii, visibility_filter)\n")
    source = replace_once(source, "        if iteration > config.densify_from_iter and iteration < config.densify_until_iter \\\n",
        "        r21.after_statistics()\n"
        "        if iteration > config.densify_from_iter and iteration < config.densify_until_iter \\\n")
    source = replace_once(source, "        model.optimizer.step()\n",
        "        r21.before_optimizer()\n        model.optimizer.step()\n")
    source = replace_once(source, "        pev[5].record()\n",
        "        pev[5].record()\n        r21.end()\n")
    source = replace_once(source, "    total_time = time.perf_counter() - t_start\n",
        "    r21.finish()\n    return\n    total_time = time.perf_counter() - t_start\n")
    ast.parse(source)
    return source


def install_topology_observers(model, session):
    """Execute actual model methods; intercept final masks, never duplicate math.

    Arm F alone can substitute masks at these exact diagnostic boundaries.
    Production/native arms return the original mask tensor verbatim.
    """
    for method_name, kind in (("densify_and_clone", "clone"), ("densify_and_split", "split")):
        original = getattr(type(model), method_name)
        source = textwrap.dedent(inspect.getsource(original))
        tree = ast.parse(source)
        assigns = [n for n in tree.body[0].body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "selected_pts_mask" for t in n.targets)]
        if len(assigns) != 2:
            raise ValueError(f"Unexpected selection contract: {method_name}")
        lines = source.splitlines(True)
        line = assigns[-1].end_lineno
        lines.insert(line, f'    selected_pts_mask = _r21_select(self, "{kind}", selected_pts_mask)\n')
        namespace = dict(original.__globals__)
        namespace["_r21_select"] = session.select
        exec(compile("".join(lines), inspect.getsourcefile(original), "exec"), namespace)
        setattr(model, method_name, types.MethodType(namespace[method_name], model))
    original = type(model).densify_and_prune
    source = replace_once(textwrap.dedent(inspect.getsource(original)),
        "    total_pruned = prune_mask.sum().item()",
        '    prune_mask = _r21_select(self, "prune", prune_mask)\n'
        "    total_pruned = prune_mask.sum().item()")
    namespace = dict(original.__globals__); namespace["_r21_select"] = session.select
    exec(compile(source, inspect.getsourcefile(original), "exec"), namespace)
    model.densify_and_prune = types.MethodType(namespace["densify_and_prune"], model)
    postfix, prune = model.densification_postfix, model.prune_points
    def observed_postfix(*args, **kwargs):
        parents = session.pending_parents
        count = args[0].shape[0]
        copies = count // len(parents) if len(parents) else 1
        new_ids = child_ids(parents, session.iteration, session.pending_kind, copies)
        if len(new_ids) != count:
            raise ValueError("Child ordering/count contract mismatch")
        result = postfix(*args, **kwargs)
        session.stable_ids = torch.cat([session.stable_ids, new_ids])
        validate_ids(session.stable_ids)
        session.events.append({"kind": "birth", "operation": session.pending_kind,
            "parents": cpu(parents), "children": cpu(new_ids), "copies": copies})
        return result
    def observed_prune(mask):
        session.events.append({"kind": "remove", "stable_ids": cpu(session.stable_ids[mask])})
        result = prune(mask)
        session.stable_ids = session.stable_ids[~mask]
        return result
    model.densification_postfix, model.prune_points = observed_postfix, observed_prune


class Session:
    def __init__(self, model, config, args, dataset, camera_sequence, extent):
        self.options = json.loads(os.environ["R21_OPTIONS"])
        self.out = Path(self.options["out"])
        self.model, self.config, self.args = model, config, args
        self.extent, self.dataset = extent, dataset
        self.sequence = torch.from_numpy(camera_sequence.copy())
        self.stable_ids = model._gaussian_ids.clone()
        self.start_iteration = 1
        self.stop_after = self.options["stop_after"]
        self.capture_at = set(self.options["capture_at"])
        self.resume_every = self.options["resume_every"]
        self.arm = self.options.get("arm", "A" if args.backend == "std_ll" else "B")
        self.donor = None
        self.projection_calls = []
        self.projection = {}
        self.events = []
        self.iteration = 0
        self.in_step = False
        self.overhead_s = 0.0
        self.identity = self.options["identity"]
        if self.options.get("resume"):
            state = load_state(self.options["resume"])
            if state["phase"] != "post_step":
                raise ValueError("Resume/replay requires post_step state with cleared gradients")
            # Cross-backend resume is intentional only for same-state causal replay.
            if not self.options.get("causal") and state["backend"] != args.backend:
                raise ValueError("Cross-backend resume requires causal mode")
            if not self.options.get("causal") and state["source_identity"].get("renderer_hashes") != self.identity.get("renderer_hashes"):
                raise ValueError("Renderer identity changed across trajectory resume")
            for key in ("scene", "seed", "recipe", "scene_extent"):
                current = {"scene": args.scene, "seed": args.seed,
                    "recipe": self.recipe(), "scene_extent": extent}[key]
                if state[key] != current:
                    raise ValueError(f"Resume contract mismatch: {key}")
            if not torch.equal(self.sequence[:len(state["camera_sequence"])], state["camera_sequence"]):
                raise ValueError("Camera sequence changed")
            if state["source_identity"]["model_sha256"] != self.identity["model_sha256"]:
                raise ValueError("GaussianModel source changed")
            self.stable_ids = restore_model(model, state)
            self.start_iteration = state["iteration"] + 1
            restore_rng(state["rng"])
        if self.start_iteration > self.stop_after:
            raise ValueError("No requested steps remain after resume checkpoint")
        if self.options.get("causal") and (not self.options.get("resume") or self.stop_after != self.start_iteration):
            raise ValueError("Causal replay must restore one checkpoint and execute exactly one step")
        if self.options.get("donor"):
            self.donor = load_state(self.options["donor"])
            if self.donor["iteration"] != self.start_iteration or self.donor["phase"] != "pre_topology":
                raise ValueError("Donor must be pre_topology at replay iteration")
            if self.donor["backend"] != "std_ll" or self.donor["arm"] != "A":
                raise ValueError("Intervention donor must be native std_ll arm A")
            if (self.donor["scene"] != args.scene or self.donor["seed"] != args.seed
                    or self.donor["recipe"] != self.recipe() or self.donor["scene_extent"] != extent):
                raise ValueError("Intervention donor scene/seed/recipe/extent mismatch")
        install_topology_observers(model, self)
        self.install_projection_observer()
        write_json(self.out / "RUN_IDENTITY.json", {"schema": "r21.run.v1", **self.identity,
            "options": self.options, "backend": args.backend, "scene": args.scene,
            "seed": args.seed, "recipe": self.recipe(), "scene_extent": extent,
            "instrumentation_overhead": True, "timing_valid_for_renderer_benchmark": False})

    def recipe(self):
        result = self.config.to_dict()
        for key in ("repo_root", "iterations", "eval_iterations", "instrument_iterations", "checkpoint_iterations"):
            result.pop(key, None)
        return result

    def install_projection_observer(self):
        import gsplat.cuda._wrapper as wrapper
        import gsplat.rendering as rendering
        original = wrapper.fully_fused_projection
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if not self.in_step:
                return result
            # Called after begin(), also during evaluation. begin clears those calls.
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            op = bound.arguments.get("opacities")
            self.projection_calls.append({"input_count": bound.arguments["means"].shape[-2],
                "opacities_supplied": op is not None,
                "radii": result[0].detach(), "means2d": result[1].detach(),
                "depths": result[2].detach(), "conics": result[3].detach(),
                "eps2d": bound.arguments.get("eps2d"),
                "near_plane": bound.arguments.get("near_plane"),
                "far_plane": bound.arguments.get("far_plane"),
                "radius_clip": bound.arguments.get("radius_clip")})
            return result
        for module in (wrapper, rendering):
            for name, value in list(vars(module).items()):
                if value is original:
                    setattr(module, name, observed)

    @torch.no_grad()
    def begin(self, iteration, cam_idx, cam, gt):
        started = time.perf_counter()
        self.in_step = True
        self.iteration = iteration; self.events = []; self.decisions = {}
        self.projection_calls = []; self.projection = {}
        self.camera = {"id": cam_idx, "image_name": str(getattr(cam, "image_name", "")),
                       "K": cpu(cam.K), "viewmatrix": cpu(cam.viewmatrix),
                       "camera_center": cpu(cam.camera_center),
                       "width": cam.image_width, "height": cam.image_height,
                       "gt_sha256": digest(gt)}
        self.row = {"schema": "r21.step.v2", "iteration": iteration, "backend": self.args.backend,
                    "camera": digest(self.camera), "count_before": len(self.stable_ids),
                    "stable_ids_before": digest(self.stable_ids.sort().values),
                    "rng_before": digest(rng_state()),
                    "optimizer_before": digest(optimizer_named(self.model)),
                    "raw_before": row_digest(self.stable_ids, {k: getattr(self.model, "_" + k) for k in RAW})}
        self.before_stats = {k: cpu(getattr(self.model, k)) for k in STATS}
        self.row["stats_before"] = row_digest(self.stable_ids, self.before_stats)
        if self.donor is not None:
            if digest(self.donor["camera"]) != digest(self.camera):
                raise ValueError("Donor camera/GT differs")
            if self.donor["input_raw_digest"] != self.row["raw_before"]:
                raise ValueError("Donor did not use identical replay input parameters")
            if self.donor["input_optimizer_digest"] != self.row["optimizer_before"] or self.donor["input_rng_digest"] != self.row["rng_before"]:
                raise ValueError("Donor optimizer/RNG input differs")
        self.overhead_s += time.perf_counter() - started

    def observe_std(self, meta):
        if not self.in_step:
            return
        self.projection = {k: meta[k].detach() for k in ("means2d", "radii", "depths", "conics") if k in meta}
        self.projection["master_rows"] = torch.arange(len(self.stable_ids), device=self.stable_ids.device)

    def observe_higs(self, out):
        if not self.in_step:
            return
        info = out["densification_info"]
        self.projection = {"radii": info["radii"].detach()}
        # Actual master rows are exposed by the real custom Function metadata.
        from gsplat.experimental.render.functional.gaussian_inference import _HigsAutogradFunction
        meta = _HigsAutogradFunction.last_forward_metadata
        rows = meta["visible_gaussian_ids"]
        self.projection["master_rows"] = rows.detach()
        if self.projection_calls and rows.numel():
            last = self.projection_calls[-1]
            if last["input_count"] != rows.numel():
                raise ValueError("Captured projection/subset count mismatch")
            for key in ("means2d", "depths", "conics"):
                self.projection[key] = last[key]
        self.projection["backward_backend"] = str(meta["backward_backend"])

    @torch.no_grad()
    def after_backward(self, radii, means2d, loss):
        started = time.perf_counter()
        if getattr(means2d, "absgrad", None) is None:
            raise ValueError("True AbsGrad missing")
        if self.args.backend == "higs" and self.projection.get("backward_backend") != "higs_native":
            raise ValueError("HiGS native backward required")
        self.absgrad = cpu(means2d.absgrad)
        self.signed_means2d = cpu(means2d.grad)
        self.radii = cpu(radii)
        self.raw_grads = {k: cpu(getattr(self.model, "_" + k).grad) for k in RAW}
        n = len(self.stable_ids)
        self.row.update({"raw_grad": row_digest(self.stable_ids, self.raw_grads),
                         "absgrad": row_digest(self.stable_ids, self.absgrad.reshape(n,2)),
                         "signed_means2d": row_digest(self.stable_ids, None if self.signed_means2d is None else self.signed_means2d.reshape(n,2)),
                         "radii": row_digest(self.stable_ids, self.radii.reshape(n,2)),
                         "projection": row_digest(self.stable_ids, self.canonical_projection()),
                         "loss": float(loss.detach())})
        self.overhead_s += time.perf_counter() - started

    @torch.no_grad()
    def statistics_inputs(self, radii, visibility):
        self.native_visibility = cpu(visibility)
        if self.arm == "C":
            # Fixed HiGS visibility isolates radius magnitudes at consumption.
            radii = remap(self.donor["statistics_radii"], self.donor["stable_ids"], cpu(self.stable_ids)).to(radii.device)
        elif self.arm == "D":
            visibility = remap(self.donor["visibility"], self.donor["stable_ids"], cpu(self.stable_ids)).to(visibility.device)
        elif self.arm == "E":
            history = load_state(self.options["history_reference"]) if self.options.get("history_reference") else None
            if history is not None and (history["iteration"] != self.iteration - 1 or history["phase"] != "post_step"
                    or history["recipe"] != self.recipe() or history["scene"] != self.args.scene
                    or history["seed"] != self.args.seed or history["backend"] != "std_ll"):
                raise ValueError("E historical donor must match recipe/scene and previous iteration")
            for key in ("xyz_gradient_accum", "denom"):
                source = history["stats"] if history is not None else self.donor["stats_before"]
                ids = history["stable_ids"] if history is not None else self.donor["stable_ids"]
                value = remap(source[key], ids, cpu(self.stable_ids))
                setattr(self.model, key, value.to(self.model._xyz.device))
        self.statistics_radii, self.visibility = cpu(radii), cpu(visibility)
        self.consumed_stats_before = {k:cpu(getattr(self.model,k)) for k in STATS}
        self.row["visibility"] = row_digest(self.stable_ids, self.visibility)
        return radii, visibility

    @torch.no_grad()
    def after_statistics(self):
        started = time.perf_counter()
        size = self.config.max_screen_size if self.iteration > self.config.opacity_reset_interval else None
        self.preview = cpu(candidates(self.model, self.config.densify_grad_threshold,
                        self.extent, self.config.min_opacity, size))
        self.row["stats_after"] = row_digest(self.stable_ids, {k: getattr(self.model, k) for k in STATS})
        self.row["candidates"] = row_digest(self.stable_ids, {k:self.preview[k] for k in ("clone", "split", "prune_parent_preview")})
        self.pre_topology = None
        if self.iteration in self.capture_at or self.options.get("capture_every_step"):
            self.pre_topology = self.snapshot("pre_topology")
        self.overhead_s += time.perf_counter() - started

    @torch.no_grad()
    def select(self, model, kind, mask):
        if self.arm == "F":
            donor = self.donor["decisions"][kind]
            mask = remap(donor["mask"], donor["stable_ids"], cpu(self.stable_ids)).to(mask.device)
        self.decisions[kind] = {"stable_ids": cpu(self.stable_ids), "mask": cpu(mask)}
        if kind in ("clone", "split"):
            self.pending_kind = kind
            self.pending_parents = self.stable_ids[mask].clone()
        return mask

    @torch.no_grad()
    def before_optimizer(self):
        # Explicit boundary: topology may replace parameters and clear their
        # .grad. Record this real behavior instead of reattaching stale grads.
        self.row["optimizer_input_grad"] = row_digest(self.stable_ids, {k:getattr(self.model, "_"+k).grad for k in RAW})

    def canonical_projection(self):
        n = len(self.stable_ids)
        radii = self.radii.reshape(n, -1)
        valid = (radii > 0).any(-1)
        rows = cpu(self.projection["master_rows"])
        result = {"radii": radii, "valid": valid}
        for key, width in (("means2d", 2), ("depths", 1), ("conics", 3)):
            value = self.projection.get(key)
            if value is None:
                if len(rows):
                    raise ValueError(f"Actual projection telemetry missing: {key}")
                full = torch.zeros(n, width, dtype=self.model._xyz.dtype)
            else:
                part = cpu(value).reshape(len(rows), width)
                full = torch.zeros(n, width, dtype=part.dtype)
                full[rows] = part
                full[~valid] = 0  # invalid native projection slots may be unwritten
            result[key] = full
        return result

    def snapshot(self, phase):
        state = model_state(self.model, self.stable_ids)
        state.update({"schema": SCHEMA, "iteration": self.iteration, "phase": phase,
            "backend": self.args.backend, "arm": self.arm, "scene": self.args.scene,
            "seed": self.args.seed, "recipe": self.recipe(), "scene_extent": self.extent,
            "camera_sequence": cpu(self.sequence), "camera": self.camera,
            "source_identity": self.identity, "input_raw_digest": self.row["raw_before"],
            "input_optimizer_digest": self.row["optimizer_before"], "input_rng_digest":self.row["rng_before"]})
        if phase == "pre_topology":
            state.update({"projection": cpu(self.projection), "projection_canonical": self.canonical_projection(),
                "projection_calls": cpu(self.projection_calls),
                "radii": self.radii, "visibility": self.visibility,
                "native_visibility": self.native_visibility, "statistics_radii": self.statistics_radii,
                "absgrad": self.absgrad, "signed_means2d": self.signed_means2d,
                "stats_before": self.consumed_stats_before, "native_stats_before": self.before_stats,
                "candidates": self.preview})
        return state

    @torch.no_grad()
    def end(self):
        started = time.perf_counter()
        if self.pre_topology is not None:
            self.pre_topology["decisions"] = cpu(self.decisions)
            self.pre_topology["topology_events"] = cpu(self.events)
            save_state(self.out / f"iter_{self.iteration:06d}_pre_topology.pt", self.pre_topology)
        if (self.iteration in self.capture_at or self.iteration % self.resume_every == 0
                or self.iteration == self.stop_after or self.options.get("capture_every_step")):
            state = self.snapshot("post_step")
            state["topology_events"] = cpu(self.events)
            save_state(self.out / f"iter_{self.iteration:06d}_post_step.pt", state)
        self.row.update({"raw_after": row_digest(self.stable_ids, {k:getattr(self.model, "_"+k) for k in RAW}),
            "stable_ids_after": digest(self.stable_ids.sort().values), "count_after": len(self.stable_ids),
            "actual_decisions": digest({k:row_digest(v["stable_ids"],v["mask"]) for k,v in self.decisions.items()}),
            "topology_events": topology_digest(self.events)})
        self.overhead_s += time.perf_counter() - started
        self.row["instrumentation_cumulative_host_s"] = self.overhead_s
        with (self.out / "steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.row, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
        self.pre_topology = None
        self.in_step = False

    def finish(self):
        write_json(self.out / "RUN_STATUS.json", {"status": "CAPTURE_COMPLETE",
            "last_iteration": self.iteration, "first_iteration": self.start_iteration,
            "instrumentation_host_s": self.overhead_s, "MX_quality_gate": "NOT_EXECUTED",
            "scope": "diagnostic capture only; no quality/performance PASS inferred"})
