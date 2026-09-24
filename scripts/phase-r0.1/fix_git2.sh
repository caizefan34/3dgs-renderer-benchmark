#!/bin/bash
cd /home/liaoyuanjun/3dgs-renderer-benchmark

# Update .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
results/
logs/
data/
patches/
_check_*.py
*.log
EOF

git add .gitignore

# Also add the report and r0.1 code
mkdir -p reports
cp /dev/null reports/.gitkeep 2>/dev/null || true
git add reports/ 2>/dev/null || true

git commit -m "fix: expand .gitignore for clean git status"

echo "=== STATUS ==="
git status --porcelain | head -5
echo "=== DIRTY ==="
git diff --quiet && echo "diff_clean" || echo "diff_dirty"
if [ -z "$(git status --porcelain)" ]; then
    echo "git_dirty=false"
else
    echo "git_dirty=true"
fi
