#!/usr/bin/env python
"""LAVOS0 Phase B+D REAL A100 capture driver (zero source diff on the frozen C0).

Wraps publication_trainer_p13x.py exactly like vos_trace_driver.py: import the
frozen trainer as a module, wrap bootstrap_c0() so that immediately after
bootstrap we patch the frozen GaussianModel class methods (the trainer imports
gaussian_model.GaussianModel AFTER bootstrap_c0 adds REFERENCE_V1_DIR to
sys.path, so the same class object is patched), then call the trainer main()
verbatim.

Captured per training iteration (metadata-only; no RNG, no input mutation, no
tensor returned to the trainer is altered -- the D0 trace-neutrality rules):
  * SH consumer set: rows with nonzero _shs.grad, mapped to the model's stable
    logical lineage ids (_gaussian_ids, R0.2 Part H). Fed to a numpy-vectorized
    streaming debt replay (lavos0.streaming_debt, proven == frozen
    debt_replay.replay_debt).
  * topology events: clone/split parent logical ids (selected masks recomputed
    verbatim from the frozen model state, asserted against the returned counts)
    and prune logical ids.
  * per-iter counts: n_total_rows, n_consumers.
  * a bounded Phase-D window: the top-K rows by densification grad norm at
    w_start (these are the real clone/split parents at the next event), tracked
    through the window with their real SH grads + real topology, to drive the
    bitwise DENSE-vs-LAYOUT topology oracle offline.
  * SH Adam microbench (CUDA-event timed) at MICROBENCH_ITERS.

Walls (iter/optimizer/densify) are NOT re-measured here; they are read from the
trainer\'s own persisted results.json/timing.json (the trainer already times them
with CUDA events), so the capture hooks do not contaminate the reported timing.

Usage:
  python lavos0_capture.py --scene room --outdir /root/lavos0_headroom/room \
      --iterations 15000 --seed 42 --gpu 0 \
      --trainer /root/epic05_p13x/publication_trainer_p13x.py
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

MICROBENCH_ITERS = (1000, 5000, 10000, 14500)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _logical_to_real(model):
    """Map logical id -> real row index. Sized by _next_gaussian_id because
    logical ids are monotone and never reused (pruned ids leave holes)."""
    gid = model._gaussian_ids
    n = int(model._next_gaussian_id)
    inv = torch.full((n,), -1, dtype=torch.long, device=gid.device)
    inv[gid] = torch.arange(gid.shape[0], device=gid.device)
    return inv


class Capture:
    def __init__(self, scene, outdir, seed, gpu,
                 pd_w_start=14850, pd_w_iters=100, pd_k=48):
        self.scene = scene
        self.outdir = outdir
        self.seed = seed
        self.gpu = gpu
        self.pd_w_start = pd_w_start
        self.pd_w_end = pd_w_start + pd_w_iters
        self.pd_k = pd_k
        self.iter = 0
        self.n_initial = None
        self.replayer = None
        self.records = []
        self.topology_log = []
        self.pending_topo = []
        self.model = None
        self.microbench = {}
        self.densify_counts = {"clone": 0, "split": 0, "prune": 0}
        self.pd = None
        self.pd_tracked = {}          # logical_id -> {"birth":iter}
        self._in_split = False
        self.pd_debug = {}
        self.errors = []

    def ensure_replayer(self, n_rows):
        if self.replayer is None:
            self.n_initial = int(n_rows)
            from lavos0.streaming_debt import StreamingDebtReplayer
            self.replayer = StreamingDebtReplayer(self.n_initial)

    def flush_topology(self):
        for ev in self.pending_topo:
            self.replayer.replay_topology(ev["clone"], ev["split"], ev["prune"])
            self.topology_log.append({
                "iter": ev["iter"], "n_clone": len(ev["clone"]),
                "n_split": len(ev["split"]), "n_prune": len(ev["prune"]),
                "clone_parents": ev["clone"], "split_parents": ev["split"],
                "prune_ids": ev["prune"]})
            self.densify_counts["clone"] += len(ev["clone"])
            self.densify_counts["split"] += len(ev["split"])
            self.densify_counts["prune"] += len(ev["prune"])
        self.pending_topo = []

    def on_step(self, model):
        self.model = model
        self.iter += 1
        if self.iter == 1:
            self.ensure_replayer(int(model._xyz.shape[0]))
        self.flush_topology()
        shs = model._shs
        gid = model._gaussian_ids
        n_total = int(model._xyz.shape[0])
        consumers = []
        if shs.grad is not None:
            nz = (shs.grad != 0).any(dim=(1, 2))
            if bool(nz.any()):
                consumers = gid[nz].detach().cpu().numpy().astype(np.int64).tolist()
        self.replayer.replay_step(consumers)
        self.records.append({"iter": self.iter, "n_total": n_total,
                             "n_consumers": len(consumers)})
        if self.iter in MICROBENCH_ITERS:
            self.microbench[self.iter] = self._microbench_sh_adam(model)
        if self.pd_w_start <= self.iter < self.pd_w_end:
            self._pd_on_iter(model, consumers)

    def _microbench_sh_adam(self, model):
        try:
            shs = model._shs
            lr = 0.0025
            for grp in model.optimizer.param_groups:
                if grp.get("name") == "shs":
                    lr = grp["lr"]
            tmp = shs.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([tmp], lr=lr, betas=(0.9, 0.999), eps=1e-15,
                                   weight_decay=0.0, amsgrad=False, foreach=False)
            for _ in range(3):
                opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            reps = 20
            e0.record()
            for _ in range(reps):
                tmp.grad = torch.zeros_like(tmp)
                opt.step(); opt.zero_grad(set_to_none=True)
            e1.record()
            torch.cuda.synchronize()
            return {"ms": e0.elapsed_time(e1) / reps, "n_rows": int(shs.shape[0]),
                    "feat": int(shs.shape[1]), "lr": lr}
        except Exception as e:
            self.errors.append("microbench: " + repr(e))
            return None

    # ---- Phase-D window ----------------------------------------------------
    def _shs_state(self, model, logical_ids):
        """Return (theta, m, v) lists for the given logical ids at this iter."""
        if not logical_ids:
            return [], [], []
        inv = _logical_to_real(model)
        lg = torch.tensor(logical_ids, dtype=torch.long, device=model._shs.device)
        real = inv[lg]
        with torch.no_grad():
            theta = model._shs[real].detach().cpu().numpy().astype(np.float32).tolist()
            m = v = None
            st = None
            for grp in model.optimizer.param_groups:
                if grp.get("name") == "shs":
                    st = model.optimizer.state.get(grp["params"][0], None)
                    break
            if st is not None:
                m = st["exp_avg"][real].detach().cpu().numpy().astype(np.float32).tolist()
                v = st["exp_avg_sq"][real].detach().cpu().numpy().astype(np.float32).tolist()
        return theta, m, v

    def _pd_select(self, model):
        with torch.no_grad():
            # MUST match densify_and_prune grad computation exactly (accum/denom,
            # nan->0, NO clamp) so the top-K are the real clone/split parents.
            grads = model.xyz_gradient_accum / model.denom
            grads[grads.isnan()] = 0.0
            gnorm = grads.squeeze(-1)
            k = min(self.pd_k, int(gnorm.shape[0]))
            top = torch.topk(gnorm, k).indices.detach().cpu().numpy().astype(np.int64)
            logical = model._gaussian_ids[top].detach().cpu().numpy().astype(np.int64)
        self.pd_tracked = {}
        for r, lg in zip(top, logical):
            self.pd_tracked[int(lg)] = {"birth": self.pd_w_start}
        self.pd_debug["selected_logical"] = [int(x) for x in logical.tolist()]
        self.pd_debug["gnorm_top"] = [float(x) for x in gnorm[top].tolist()]
        self.pd_debug["gnorm_thresh"] = float(gnorm.min().item()) if gnorm.numel() else None
        theta, m, v = self._shs_state(model, list(self.pd_tracked.keys()))
        self.pd = {
            "w_start": self.pd_w_start, "w_end": self.pd_w_end, "k": len(self.pd_tracked),
            "init_state": {"logical": list(self.pd_tracked.keys()),
                           "theta": theta, "m": m, "v": v},
            "births": {}, "iters": {},
        }

    def _pd_on_iter(self, model, consumers):
        if self.iter == self.pd_w_start:
            self._pd_select(model)
        if self.pd is None:
            return
        consumer_set = set(consumers)
        inv = _logical_to_real(model)
        # MERGE (do not overwrite): _pd_on_topology may have already written an
        # "events" entry for this iter during the topology phase (before step).
        slot = self.pd["iters"].setdefault(
            str(self.iter), {"consumers": {}, "events": []})
        slot.setdefault("events", [])
        for lg in list(self.pd_tracked.keys()):
            real = int(inv[torch.tensor(lg, device=model._shs.device)].item())
            if real < 0:
                continue
            is_cons = lg in consumer_set
            g = None
            if is_cons and model._shs.grad is not None:
                g = model._shs.grad[real].detach().cpu().numpy().astype(np.float32).tolist()
            slot["consumers"][str(lg)] = {"c": int(is_cons), "g": g}

    def _pd_on_topology(self, clone_parents, clone_theta, split_parents,
                        split_theta, prune_ids):
        """Record Phase-D topology events for TRACKED rows. Thetas are captured
        by the wrapper BEFORE the frozen op runs (split parents are gone after)."""
        self.pd_debug["on_topo_calls"] = self.pd_debug.get("on_topo_calls", 0) + 1
        if self.pd is None:
            self.pd_debug["on_topo_pd_none"] = self.pd_debug.get("on_topo_pd_none", 0) + 1
            return
        ev = {"clone": [], "split": [], "prune": [], "child_theta": None}
        if clone_parents:
            ct = []
            for lg, th in zip(clone_parents, clone_theta):
                if lg in self.pd_tracked:
                    ev["clone"].append(int(lg)); ct.append(th)
            if ct:
                ev["child_theta"] = ct
        if split_parents:
            ct = []
            for lg, th in zip(split_parents, split_theta):
                if lg in self.pd_tracked:
                    ev["split"].append(int(lg))
                    ct.append(th); ct.append(th)   # N=2 children per parent
            if ct:
                ev["child_theta"] = ct
            self.pd_debug["last_split_tracked"] = len(ev["split"])
        for lg in prune_ids:
            if lg in self.pd_tracked:
                ev["prune"].append(int(lg))
                self.pd_tracked.pop(int(lg), None)
        if ev["clone"] or ev["split"] or ev["prune"]:
            it = str(self.iter + 1)
            self.pd["iters"].setdefault(it, {"consumers": {}})
            self.pd["iters"][it]["events"] = (
                self.pd["iters"][it].get("events", []) + [ev])


def _selected_mask_clone(model, grads, grad_threshold, extent):
    mask = torch.norm(grads, dim=-1) >= grad_threshold
    return torch.logical_and(
        mask, torch.max(model.get_scaling, dim=1).values <=
        model.percent_dense * extent)


def _selected_mask_split(model, grads, grad_threshold, extent):
    n_init = model._xyz.shape[0]
    padded = torch.zeros((n_init,), device="cuda")
    padded[:grads.shape[0]] = grads.squeeze()
    mask = padded >= grad_threshold
    return torch.logical_and(
        mask, torch.max(model.get_scaling, dim=1).values >
        model.percent_dense * extent)


def install_patches(T, cap):
    # The frozen trainer imports gaussian_model INSIDE main() (after bootstrap),
    # so it is not yet an attribute of T here. bootstrap_c0 added REFERENCE_V1_DIR
    # to sys.path; import it now and patch the class object. The trainer's later
    # `from gaussian_model import GaussianModel` reuses this same cached module.
    import gaussian_model
    GM = gaussian_model.GaussianModel
    orig_ts = GM.training_setup

    def training_setup(self, config):
        orig_ts(self, config)
        cap.model = self
        orig_step = self.optimizer.step

        def step_wrapped():
            cap.on_step(self)
            return orig_step()

        self.optimizer.step = step_wrapped
    GM.training_setup = training_setup

    orig_clone = GM.densify_and_clone

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        mask = _selected_mask_clone(self, grads, grad_threshold, scene_extent)
        parents = self._gaussian_ids[mask].detach().cpu().numpy().astype(np.int64).tolist()
        theta = self._shs[mask].detach().cpu().numpy().astype(np.float32).tolist() if parents else []
        n = orig_clone(self, grads, grad_threshold, scene_extent)
        if len(parents) != n:
            cap.errors.append(f"clone mismatch {len(parents)}!={n} @iter {cap.iter+1}")
        cap.pending_topo.append({"iter": cap.iter + 1, "clone": parents,
                                 "split": [], "prune": []})
        cap._pd_on_topology(parents, theta, [], [], [])
        return n
    GM.densify_and_clone = densify_and_clone

    orig_split = GM.densify_and_split

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        mask = _selected_mask_split(self, grads, grad_threshold, scene_extent)
        parents = self._gaussian_ids[mask].detach().cpu().numpy().astype(np.int64).tolist()
        theta = self._shs[mask].detach().cpu().numpy().astype(np.float32).tolist() if parents else []
        cap._in_split = True
        try:
            n = orig_split(self, grads, grad_threshold, scene_extent, N)
        finally:
            cap._in_split = False
        if len(parents) != n:
            cap.errors.append(f"split mismatch {len(parents)}!={n} @iter {cap.iter+1}")
        cap.pd_debug["split_parents"] = [int(x) for x in parents]
        cap.pd_debug["split_count"] = len(parents)
        cap.pending_topo.append({"iter": cap.iter + 1, "clone": [],
                                 "split": parents, "prune": []})
        cap._pd_on_topology([], [], parents, theta, [])
        return n
    GM.densify_and_split = densify_and_split

    orig_prune = GM.prune_points

    def prune_points(self, mask):
        parents = self._gaussian_ids[mask].detach().cpu().numpy().astype(np.int64).tolist()
        if cap._in_split:
            return orig_prune(self, mask)   # split-internal parent removal
        orig_prune(self, mask)
        cap.pending_topo.append({"iter": cap.iter + 1, "clone": [],
                                 "split": [], "prune": parents})
        cap._pd_on_topology([], [], [], [], parents)
    GM.prune_points = prune_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iterations", type=int, default=15000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trainer", default="/root/epic05_p13x/publication_trainer_p13x.py")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final-eval", default="subset")
    ap.add_argument("--pd-w-start", type=int, default=14850)
    ap.add_argument("--pd-w-iters", type=int, default=100)
    ap.add_argument("--pd-k", type=int, default=48)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cap = Capture(args.scene, args.outdir, args.seed, args.gpu,
                  pd_w_start=args.pd_w_start, pd_w_iters=args.pd_w_iters,
                  pd_k=args.pd_k)
    spec = importlib.util.spec_from_file_location("lavos0_trainer", args.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    orig_bootstrap = T.bootstrap_c0

    def bootstrap_wrapped():
        ident = orig_bootstrap()
        install_patches(T, cap)
        return ident

    T.bootstrap_c0 = bootstrap_wrapped
    targv = [args.trainer, "--arm", "c0", "--scene", args.scene,
             "--loss", "fused", "--gpu", str(args.gpu), "--eps2d", "0.3",
             "--iterations", str(args.iterations), "--outdir", args.outdir,
             "--checkpoints", "--final-eval", args.final_eval,
             "--timing-grade", "FUNCTIONAL_ONLY"]
    targv += ["--seed", str(args.seed)]
    targv += list(args.extra)
    sys.argv = targv
    t0 = time.time()
    exit_ok = True
    try:
        T.main()
    except SystemExit as e:
        exit_ok = (e.code in (0, None))
    except BaseException:
        exit_ok = False
        raise
    finally:
        _write_capture(cap, args, t0, exit_ok)
    sys.exit(0 if exit_ok else 1)


def _write_capture(cap, args, t0, exit_ok):
    stats = cap.replayer.finalize() if cap.replayer is not None else None
    walls = {}
    rp = os.path.join(args.outdir, "results.json")
    if os.path.isfile(rp):
        try:
            d = json.load(open(rp))
            t = d.get("timing", {})
            pm = t.get("phase_ms", {})
            walls = {"mean_iter_ms": t.get("mean_iter_ms"),
                     "median_iter_ms": t.get("median_iter_ms"),
                     "total_wall_s": t.get("total_wall_s"),
                     "phase_ms_mean": {k: v.get("mean_ms") for k, v in pm.items()},
                     "densification_events": d.get("densification_events", [])}
        except Exception as e:
            cap.errors.append("read results.json: " + repr(e))
    mb = [v["ms"] for v in cap.microbench.values() if v]
    out = {"scene": cap.scene, "seed": cap.seed,
           "iterations": args.iterations, "gpu": args.gpu,
           "trainer_path": args.trainer,
           "trainer_sha256": sha256_file(args.trainer),
           "exit_ok": exit_ok, "capture_wall_s": time.time() - t0,
           "n_initial_rows": cap.n_initial,
           "debt_stats": stats.as_dict() if stats else None,
           "n_iterations_captured": len(cap.records),
           "records_summary": _summarize_records(cap.records),
           "densify_counts_captured": cap.densify_counts,
           "topology_log": cap.topology_log,
           "microbench_sh_adam_ms": cap.microbench,
           "sh_adam_ms_per_iter_mean": (sum(mb) / len(mb)) if mb else None,
           "walls": walls, "phase_d": cap.pd, "pd_debug": cap.pd_debug, "errors": cap.errors}
    with open(os.path.join(args.outdir, "lavos0_capture.json"), "w") as f:
        json.dump(out, f, indent=2)


def _summarize_records(records):
    if not records:
        return {"n_iterations": 0}
    nt = [r["n_total"] for r in records]
    nz = [r["n_consumers"] for r in records]
    snt, snz = sum(nt), sum(nz)
    return {"n_iterations": len(records),
            "n_total_rows_mean": sum(nt) / len(nt),
            "n_consumers_mean": sum(nz) / len(nz),
            "nonzero_grad_fraction_mean": (snz / snt) if snt else None,
            "n_total_rows_first": nt[0], "n_total_rows_last": nt[-1]}


if __name__ == "__main__":
    main()
