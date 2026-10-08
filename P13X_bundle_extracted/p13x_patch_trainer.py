#!/usr/bin/env python3
"""Patch publication_trainer_p13x.py (pristine P12 copy) with the P13-X
side-channel diagnostic hooks. Every insertion is an exact unique match;
the script fails loudly otherwise. Emits a unified diff for review."""
import difflib
import json
import sys

P = "/root/epic05_p13x/publication_trainer_p13x.py"
src = open(P, encoding="utf-8").read()
orig = src

EDITS = []

def edit(name, old, new):
    global src
    n = src.count(old)
    assert n == 1, f"[{name}] anchor match count = {n} (need exactly 1)"
    src = src.replace(old, new)
    EDITS.append(name)

# ---- 1. P13X CLI args (all default OFF) --------------------------------
edit("cli_args",
     "    args = ap.parse_args()\n",
     """    # ---- P13X: side-channel diagnostic args (all default OFF) ----
    ap.add_argument("--p13x-diag", type=int, default=0, choices=[0, 1],
                    help="P13-X diagnostics master switch (0 = P12-identical)")
    ap.add_argument("--p13x-diag-start", type=int, default=15000)
    ap.add_argument("--p13x-diag-end", type=int, default=30000)
    ap.add_argument("--p13x-diag-stride", type=int, default=250)
    ap.add_argument("--p13x-anchors", type=int, default=8)
    ap.add_argument("--p13x-lags", type=str, default="250,500,1000,2500")
    ap.add_argument("--p13x-x4-samples", type=int, default=256)
    ap.add_argument("--p13x-x5", type=int, default=1)
    ap.add_argument("--p13x-x5-anchors", type=int, default=8)
    ap.add_argument("--p13x-outdir", type=str, default=None)
    ap.add_argument("--p13x-rng-footprint", type=int, default=0,
                    help="1 = record RNG state hashes at milestones "
                         "(read-only; for the contamination preflight)")
    args = ap.parse_args()
""")

# ---- 2. construct diagnostics ------------------------------------------
edit("construct",
     """    eval_indices = list(range(0, n_cameras, max(1, n_cameras // 10)))
    eval_iterations = set(i for i in config.eval_iterations if i <= args.iterations)
""",
     """    eval_indices = list(range(0, n_cameras, max(1, n_cameras // 10)))
    eval_iterations = set(i for i in config.eval_iterations if i <= args.iterations)

    # ---- P13X: construct side-channel diagnostics (inert when disabled) ----
    p13x = None
    p13x_x4_set = set()
    if args.p13x_diag == 1:
        import p13x_diag
        _p13x_out = args.p13x_outdir or os.path.join(args.outdir, "p13x")
        p13x = p13x_diag.P13XDiag(
            outdir=_p13x_out, scene=args.scene, n_cameras=n_cameras,
            iterations=config.iterations, seed=config.seed, arm=args.arm,
            diag_start=args.p13x_diag_start, diag_end=args.p13x_diag_end,
            diag_stride=args.p13x_diag_stride, n_anchors=args.p13x_anchors,
            lags=[int(x) for x in args.p13x_lags.split(",")],
            x4_samples=args.p13x_x4_samples,
            x5_enabled=bool(args.p13x_x5) and args.loss == "fused",
            x5_anchors=args.p13x_x5_anchors)
        p13x.bind(model=model, dataset=dataset,
                  loss_fn=(train_ssim_fn if args.loss == "fused" else None),
                  eps2d=eps2d_eff)
        p13x_x4_set = p13x.x4_set
        p13x.write_protocol()
        print(f"  [P13X] diagnostics ARMED window="
              f"[{p13x.diag_start},{p13x.diag_end}] stride={p13x.diag_stride} "
              f"anchors={p13x.anchors} x4_iters={len(p13x.x4_iters)}", flush=True)
""")

# ---- 3. X4 retain_grad hook (value-transparent, no CUDA work) ----------
edit("retain_grad",
     """        pev[1].record()

        if args.loss == "fused":
""",
     """        pev[1].record()

        # ---- P13X X4 hook: retain image grad (flag set only; no CUDA work)
        if p13x is not None and iteration in p13x_x4_set:
            image.retain_grad()

        if args.loss == "fused":
""")

# ---- 4. diagnostics entry (strictly outside all timed regions) ---------
edit("diag_entry",
     """        vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_vram = max(peak_vram, vram)

        if iteration % 100 == 0 or iteration == 1:
""",
     """        vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_vram = max(peak_vram, vram)

        # ---- P13X diagnostics: after phase events + sync, never timed ----
        if p13x is not None:
            p13x.on_iteration(iteration, image)
        if args.p13x_rng_footprint == 1 and (
                iteration % 100 == 0 or iteration == 1):
            import hashlib as _hl
            _fp = {
                "iter": iteration,
                "torch_cpu_rng_sha256": _hl.sha256(
                    torch.get_rng_state().numpy().tobytes()).hexdigest(),
                "torch_cuda_rng_sha256": _hl.sha256(
                    torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
            }
            with open(os.path.join(args.outdir, "P13X_RNG_FOOTPRINT.jsonl"),
                      "a") as _f:
                _f.write(json.dumps(_fp) + "\\n")

        if iteration % 100 == 0 or iteration == 1:
""")

# ---- 5. finalize after the training loop -------------------------------
edit("finalize",
     """    total_time = time.perf_counter() - t_start

    # ---- phase timing aggregation (means over all iters) ----
""",
     """    total_time = time.perf_counter() - t_start

    # ---- P13X: write diagnostic artifacts (untimed) ----
    if p13x is not None:
        p13x.finalize()

    # ---- phase timing aggregation (means over all iters) ----
""")

with open(P, "w", encoding="utf-8") as f:
    f.write(src)

diff = "\n".join(difflib.unified_diff(
    orig.splitlines(), src.splitlines(),
    fromfile="publication_trainer_p12.py (pristine copy)",
    tofile="publication_trainer_p13x.py", lineterm="", n=3))
with open("/root/epic05_p13x/P13X_TRAINER_DIFF.patch", "w") as f:
    f.write(diff + "\n")

compile_ok = True
try:
    compile(src, P, "exec")
except SyntaxError as e:
    compile_ok = False
    print("SYNTAX ERROR:", e)
print(json.dumps({"edits_applied": EDITS, "syntax_ok": compile_ok,
                  "p12_lines": len(orig.splitlines()),
                  "p13x_lines": len(src.splitlines())}, indent=2)
      if compile_ok else "FAILED")
sys.exit(0 if compile_ok else 1)
