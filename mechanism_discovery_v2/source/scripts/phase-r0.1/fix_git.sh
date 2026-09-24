#!/bin/bash
cd /home/liaoyuanjun/3dgs-renderer-benchmark

# Remove __pycache__ from tracking
git rm -r --cached baseline/reference_v1/__pycache__/ 2>/dev/null || true

# Add .gitignore
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo "results/" >> .gitignore
echo "logs/" >> .gitignore

git add .gitignore
git commit -m "fix: remove __pycache__ from tracking, add .gitignore"

echo "=== DIRTY ==="
git diff --quiet && echo false || echo true
git status --short | head -5
