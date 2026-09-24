#!/bin/bash
set -euo pipefail

# =====================================================================
# Candidate C R3 — Independent Audit Package Builder
# =====================================================================
REPO="/home/liaoyuanjun/3dgs-renderer-benchmark"
SRC_R3="$REPO/results/reference_v1/r3"
AUDIT="$REPO/audit_packages/candidate_c_r3_final"

mkdir -p "$AUDIT/raw/5000" "$AUDIT/raw/15000" "$AUDIT/raw/29970"
mkdir -p "$AUDIT/source"
mkdir -p "$AUDIT/docs"
mkdir -p "$AUDIT/analysis"
mkdir -p "$AUDIT/checkpoints"

# ----------------------------------------------------------
# 1. Raw window data (small JSON/log files; NPK entries recorded)
# ----------------------------------------------------------
echo "Copying raw window files..."
for d in 5000 15000 29970; do
  cp -v "$SRC_R3/$d"/*.json "$AUDIT/raw/$d/" 2>/dev/null || echo "Warning: no JSON in $d"
  cp -v "$SRC_R3/$d"/runner.log "$AUDIT/raw/$d/" 2>/dev/null || echo "Warning: no runner.log in $d"
done

# Top-level aggregate files
cp -v "$SRC_R3"/aggregated_*.json "$AUDIT/analysis/" 2>/dev/null || echo "Warning: no aggregate JSON"
cp -v "$SRC_R3"/final_decider.json "$AUDIT/analysis/" 2>/dev/null || echo "Warning: not updating top-level final_decider.json yet"
cp -v "$SRC_R3"/COMPLETE.marker "$AUDIT/" 2>/dev/null || echo "Warning: no COMPLETE.marker"

# ----------------------------------------------------------
# 2. Source code
# ----------------------------------------------------------
echo "Copying source code..."
cp -v "$REPO/experiments/r3/r3_certificate_certificates.py" "$AUDIT/source/" 2>/dev/null || echo "Warning: r3_certificate_certificates.py not found"
cp -v "$REPO/experiments/r3/r3_certificate_tightness.py" "$AUDIT/source/" 2>/dev/null || cp -v "$REPO/experiments/r3/r3_certificate_tightness.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/r3_certificate_correctness.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/r3_analyze.py" "$AUDIT/source/" 2>/dev/null || echo "Warning: r3_analyze.py not found"
cp -v "$REPO/experiments/r3/r3_decider.py" "$AUDIT/source/" 2>/dev/null || echo "Warning: r3_decider.py not found"
cp -v "$REPO/experiments/r3/r3_check.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/r3_validator.py" "$AUDIT/source/" 2>/dev/null || cp -v "$REPO/experiments/r3/r3_validate_work_certificate.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/r3_provenance.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/r3_sigma_min.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/r3_joint_skip.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/r3_cost_model.py" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/run_r3_remote.sh" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/run_r3_noop.sh" "$AUDIT/source/" 2>/dev/null || true
cp -v "$REPO/experiments/r3/work_certificate_validation_certificate.json" "$AUDIT/source/" 2>/dev/null || echo "Warning: validation certificate not found"

# ----------------------------------------------------------
# 3. Checkpoint metadata (symlinks documented)
# ----------------------------------------------------------
CKPT_DIR="$REPO/results/reference_v1/room_30k/checkpoints"
echo "Recording checkpoint metadata..."
{
  echo "# Checkpoint metadata for R3 windows"
  echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "## Files present:"
  ls -la "$CKPT_DIR" 2>/dev/null | head -30 || echo "(checkpoints dir not found)"
  echo ""
  echo "## Symlink details:"
  ls -la "$REPO/results/reference_v1/room_30k/" 2>/dev/null | grep -i iter || true
  echo ""
  echo "## 29970 window:"
  ls -la "$REPO/results/reference_v1/room_30k/checkpoints/iter_29970.pt" 2>/dev/null || echo "no iter_29970.pt"
  ls -la "$REPO/results/reference_v1/room_30k/checkpoints/iter_30000.pt" 2>/dev/null || echo "no iter_30000.pt"
  echo ""
  readlink -f "$REPO/results/reference_v1/room_30k/checkpoints/iter_29970.pt" 2>/dev/null || true
} > "$AUDIT/checkpoint_metadata.txt"
cat "$AUDIT/checkpoint_metadata.txt"

# ----------------------------------------------------------
# 3b. Compute SHA256 of all copied small files (for integrity)
# ----------------------------------------------------------
echo "Computing integrity hashes..."
find "$AUDIT" -type f ! -name "manifest.csv" ! -name ".git" -exec sha256sum {} \; > "$AUDIT/sha256sums.txt" 2>/dev/null || true
wc -l "$AUDIT/sha256sums.txt"

# ----------------------------------------------------------
# 4. Environment snapshot (as requested by spec)
# ----------------------------------------------------------
{
  echo "# Environment used for R3 measurements (captured from remote host)"
  echo ""
  echo "## Python"
  which python
  python --version 2>&1
  echo ""
  echo "## Key packages"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1
  python -c "import numpy; print('numpy', numpy.__version__)" 2>&1
  echo ""
  echo "## GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -5
  echo ""
  echo "## OS"
  uname -a
  cat /etc/os-release 2>/dev/null | head -3 || true
} > "$AUDIT/environment.txt"
cat "$AUDIT/environment.txt"

echo ""
echo "Audit package structure:"
find "$AUDIT" -type f | sort | head -50
echo ""
echo "Package size (excluding NPK):"
du -sh "$AUDIT"
echo ""
echo "BUILD_DONE"
