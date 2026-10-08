"""Run gradient-energy diagnostics. Full-frame fine rendering; not a speedup benchmark."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from .core import Anchor, blocks, block_losses, upsample
from .renderers import synthetic, render_reference, render_gsplat


def load_bundle(path, device):
    # Canonical tensor-only export; intentionally no guessed legacy checkpoint schema.
    data = torch.load(path, map_location=device, weights_only=True)
    keys = {"means", "quats", "log_scales", "opacity_logits", "sh0", "shN"}
    for name in ("params", "anchor_params"):
        if set(data[name]) != keys:
            raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    p = {k: v.detach().float().requires_grad_() for k, v in data["params"].items()}
    old = {k: v.detach().float() for k, v in data["anchor_params"].items()}
    camera = data["camera"]
    camera = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in camera.items()}
    target = data["target"].detach().float()
    if target.shape != (camera["height"], camera["width"], 3):
        raise ValueError("Target dimensions must match original full-resolution camera")
    if not torch.isfinite(target).all() or target.min() < 0 or target.max() > 1:
        raise ValueError("Target must be finite RGB in [0, 1]")
    return p, old, camera, target, int(data["sh_degree"]), data.get("metadata", {})


def gradient(loss, p):
    grads = torch.autograd.grad(loss, tuple(p.values()), allow_unused=True)
    return {k: torch.zeros_like(v) if g is None else g for (k, v), g in zip(p.items(), grads)}


def timed_backward(make_loss, p, repeats):
    def sync():
        if next(iter(p.values())).is_cuda:
            torch.cuda.synchronize()
    samples = []
    for i in range(repeats+2):
        sync(); t = time.perf_counter()
        g = gradient(make_loss(), p)
        sync(); elapsed = 1000*(time.perf_counter()-t)
        del g
        if i >= 2:
            samples.append(elapsed)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def run(p, old, camera, target, degree, backend, coarse_div, block_size, max_blocks, seed, repeats):
    h, w = target.shape[:2]; ch, cw = max(1, h//coarse_div), max(1, w//coarse_div)
    renderer = render_reference if backend == "reference" else render_gsplat
    render = lambda state, H, W: renderer(state, camera, H, W, degree)
    with torch.no_grad():
        a = Anchor.capture(render(old, h, w), render(old, ch, cw))
    partition = blocks(h, w, block_size)
    indices = list(range(len(partition)))
    if max_blocks and max_blocks < len(indices):
        indices = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))[:max_blocks].tolist()
    rows = []
    # Rebuild graphs per block/arm. No B*N Jacobian allocation for multi-million-Gaussian scenes.
    for idx in indices:
        b = [partition[idx]]
        gf = gradient(block_losses(render(p, h, w), target, b).sum(), p)
        gp = gradient(block_losses(a.proxy(render(p, ch, cw)), target, b).sum(), p)
        row = {"index": idx, "center": vars(partition[idx]), "groups": {}}
        for k in p:
            row["groups"][k] = {"fine_energy": gf[k].double().square().sum().item(),
                                 "difference_energy": (gf[k]-gp[k]).double().square().sum().item()}
        del gp
        gr = gradient(block_losses(upsample(render(p, ch, cw), h, w).clamp(0, 1), target, b).sum(), p)
        for k in p:
            row["groups"][k]["raw_difference_energy"] = (gf[k]-gr[k]).double().square().sum().item()
        del gf, gr
        rows.append(row)
    totals = {}
    for k in p:
        totals[k] = {m: sum(row["groups"][k][m] for row in rows)
                     for m in ("fine_energy", "difference_energy", "raw_difference_energy")}
    totals["all_raw_units"] = {m: sum(v[m] for v in totals.values())
                               for m in ("fine_energy", "difference_energy", "raw_difference_energy")}
    for v in totals.values():
        den = v["fine_energy"]
        v["kappa"] = v["difference_energy"]/den if den > 1e-30 else None
        v["raw_kappa"] = v["raw_difference_energy"]/den if den > 1e-30 else None
        v["variance_ratio_p0125_vs_plain_p025"] = 7*v["kappa"]/3 if v["kappa"] is not None else None
    timing = {}
    if repeats:
        timing["dense_reference_loss_forward_backward"] = timed_backward(
            lambda: block_losses(render(p, h, w), target, partition).sum(), p, repeats)
        timing["proxy_only_forward_backward"] = timed_backward(
            lambda: block_losses(a.proxy(render(p, ch, cw)), target, partition).sum(), p, repeats)
    return {"backend": backend, "scope": "fixed-state gradient diagnostic, no optimizer or ADC",
            "resolution_hw": [h, w], "coarse_hw": [ch, cw], "num_gaussians": len(p["means"]),
            "sh_degree": degree, "blocks_total": len(partition), "blocks_probed": len(rows),
            "energy_scope": "exact all-block sum" if len(rows) == len(partition) else "uniform subset ratio estimate; not exact global kappa",
            "seed": seed, "anchor_bytes": sum(t.numel()*t.element_size() for t in (a.fine, a.coarse)),
            "groups": totals, "per_block": rows, "timing": timing,
            "timing_warning": "Diagnostic full-frame renderer and unfused loss. No sparse/shared-projection implementation; no speedup claim.",
            "decision": "GPU_REAL_SCENE_GATE_REQUIRED" if backend == "reference" else "DIAGNOSTIC_ONLY_REQUIRES_PROTOCOL_AGGREGATION"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["reference", "gsplat"], default="reference")
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=.18)
    ap.add_argument("--drift", type=float, default=.003)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--coarse-div", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--timing-repeats", type=int, default=0)
    args = ap.parse_args()
    if args.coarse_div < 1 or args.max_blocks < 0 or args.timing_repeats < 0:
        ap.error("coarse-div >= 1; max-blocks and timing-repeats >= 0")
    torch.set_num_threads(1)
    if args.backend == "gsplat":
        if args.bundle is None or not torch.cuda.is_available():
            ap.error("gsplat requires CUDA and an explicit canonical --bundle")
        p, old, c, target, degree, meta = load_bundle(args.bundle, "cuda")
    else:
        if args.bundle:
            ap.error("Real scene bundles must use --backend gsplat")
        p, old, c, target = synthetic(args.scale, args.drift, args.seed)
        degree, meta = 3, {"synthetic": True, "scale": args.scale, "drift": args.drift}
    result = run(p, old, c, target, degree, args.backend, args.coarse_div, args.block_size,
                 args.max_blocks, args.seed, args.timing_repeats)
    result["input_metadata"] = meta
    result["environment"] = {"torch": torch.__version__, "python": platform.python_version(),
                             "cuda": torch.version.cuda, "device": str(next(iter(p.values())).device)}
    if args.bundle:
        with args.bundle.open("rb") as f:
            result["bundle_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "groups": result["groups"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
