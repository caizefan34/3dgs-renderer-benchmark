import subprocess
# search for prebuilt gsplat .so that matches torch 2.4.1 (anysplat)
cmd = r'''
echo "=== gsplat_cuda*.so on box ==="
find / -name "gsplat_cuda*.so" 2>/dev/null | head
echo "=== csrc.so variants ==="
find / -path "*gsplat*csrc*.so" 2>/dev/null | head
echo "=== experimental render ext ==="
find / -name "experimental_gaussian_render_inference_scene_cuda*.so" 2>/dev/null | head
echo "=== any .so with gsplat_render ==="
find /home/liaoyuanjun -name "*.so" 2>/dev/null | grep -i "gsplat\|cuda" | head -20
'''
r=subprocess.run(["bash","-lc",cmd],capture_output=True,text=True)
print(r.stdout[-3000:])
print("ERR",r.stderr[-300:])
