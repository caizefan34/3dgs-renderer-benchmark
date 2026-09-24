# Training Workset Stability Audit

Scene: `room`  
Image: 779x519, Steps: 500, Tile: 16

---

## Gaussian Count Evolution

- Min: 112627, P50: 112992, P90: 113333, Max: 113333, Final: 113333

## Training-View (Confounded) Stability

Measured across random camera views at each step. Low overlap expected due to viewpoint changes.

### Visible Gaussian (A)

- **Lag-1**: P50=0.1366, Mean=0.2409, P90=0.6524

- **Lag-2**: P50=0.1112, Mean=0.2165, P90=0.6141

- **Lag-4**: P50=0.1216, Mean=0.2474, P90=0.7373

- **Lag-8**: P50=0.1208, Mean=0.2364, P90=0.6294

### Active Tile (B)

- **Lag-1**: P50=0.4096, Mean=0.4282, P90=0.6536

- **Lag-2**: P50=0.3945, Mean=0.4097, P90=0.5923

- **Lag-4**: P50=0.4096, Mean=0.4347, P90=0.6663

- **Lag-8**: P50=0.4080, Mean=0.4254, P90=0.6416

## Same-Viewpoint Stability (Decisive)

Measured from 5 fixed cameras every 50 steps. This measures **pure parameter-evolution stability**.

- **Visible Gaussian** median P50: **0.921** (classification: **HIGH**)

- **Active Tile** median P50: **0.647** (classification: **LOW**)

Per-camera VG overlap at lag-1:

  - Camera 1: **0.923**

  - Camera 2: **0.919**

  - Camera 3: **0.921**

  - Camera 4: **0.923**

  - Camera 5: **0.913**

Per-camera AT overlap at lag-1:

  - Camera 1: **0.690**

  - Camera 2: **0.681**

  - Camera 3: **0.647**

  - Camera 4: **0.587**

  - Camera 5: **0.556**

## Classification

- visible_gaussian: **LOW**

- active_tile: **LOW**

- visible_gaussian_same_view: **HIGH**

- active_tile_same_view: **LOW**

## Overall

- Temporal persistence: YES for VG (0.921), NO for AT (0.647)
- Research opportunity: PARTIAL RESEARCH OPPORTUNITY
- Next: Pursue Gaussian-level reuse; investigate tile instability root cause
