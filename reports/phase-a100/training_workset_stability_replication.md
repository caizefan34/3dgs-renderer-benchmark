# Training Workset Stability Replication — A100 Canonical

- Host: `bms-39468022-001`; GPU: `NVIDIA A100-PCIE-40GB`; gsplat: `1.5.3`
- Scene `room`, 500 steps, tile 16, 1080p
- Training: PSNR 19.74 → 24.52; N 1,593,376 → 1,584,097; NaN/Inf=False

## Exact online overlap (training view; camera changes confound this metric)

- A, lag-1: {"n": 499, "mean": 0.7024353821605961, "std": 0.25672335740040086, "p10": 0.28453893148178516, "p50": 0.7991693112974018, "p90": 0.9395639053049942, "min": 3.4800402292650504e-06, "max": 0.981432733710134}
- B, lag-1: {"n": 499, "mean": 1.0, "std": 0.0, "p10": 1.0, "p50": 1.0, "p90": 1.0, "min": 1.0, "max": 1.0}
- C, lag-1: {"n": 499, "mean": 0.08004057293350358, "std": 0.0813616629764593, "p10": 0.002299145473598027, "p50": 0.0553239317034357, "p90": 0.19256964346904953, "min": 0.0, "max": 0.5030273305835484}

## Same-viewpoint stability (five fixed cameras; consecutive evaluation checkpoints)

- A: {"n": 50, "mean": 0.7445001018933474, "std": 0.35202049970884997, "p10": 0.18803145523675835, "p50": 0.970089243622863, "p90": 0.9817296648466335, "min": 0.14833318869865986, "max": 0.986475474720383}
- B: {"n": 50, "mean": 1.0, "std": 0.0, "p10": 1.0, "p50": 1.0, "p90": 1.0, "min": 1.0, "max": 1.0}
- C: {"n": 50, "mean": 0.3461543132335271, "std": 0.2306049276227494, "p10": 0.0014629273591320465, "p50": 0.4253683912748981, "p90": 0.5786601727487265, "min": 0.0004525904389557207, "max": 0.7126139846420693}

## Evidence classification

- Gaussian-level reuse (A): `SUPPORTED`
- Tile-level reuse (B): `SUPPORTED`
- Gaussian-level cache + event-aware tile rebuild: `HYPOTHESIS` (overlap does not establish speedup).
