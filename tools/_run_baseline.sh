export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.0"

# Fix for nvcc 11.5 + GCC 11.4: create g++ wrapper that strips -std=c++17
cat > ~/gxx_wrap.sh << 'GEOF'
#!/bin/bash
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "-std=c++17" ]]; then
    ARGS+=("-std=c++14")
  else
    ARGS+=("$arg")
  fi
done
exec /usr/bin/g++-11 "${ARGS[@]}"
GEOF
chmod +x ~/gxx_wrap.sh
export CUDAHOSTCXX=$HOME/gxx_wrap.sh

echo "=== Clean JIT cache ==="
rm -rf $HOME/.cache/torch_extensions

echo "=== Test gsplat import ==="
cd
python3 -c "import gsplat; print('OK:', gsplat.__version__)" 2>&1 | tail -5

echo "=== Run baseline timing ==="
python3 -u p5_baseline_runner.py
