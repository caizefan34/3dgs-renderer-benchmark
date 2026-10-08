import subprocess, os, glob
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print("=== look for P12/C0 room training logs ===")
print(sh("find /root -maxdepth 4 -name '*.log' -path '*c0*room*' 2>/dev/null | head -10"))
print(sh("find /root -maxdepth 4 -iname '*room*25*' 2>/dev/null | head -10"))
print(sh("find /root -maxdepth 3 -iname '*c0*' -type d 2>/dev/null | head -10"))
print("=== n1snap room checkpoints (N per ckpt via a quick probe is slow; just list) ===")
print(sh("ls -la /root/epic05_n1/runs/n1snap_c0f_s42_room/ 2>/dev/null"))
print("=== any results json for room c0 ===")
print(sh("find /root -maxdepth 5 -name 'results*.json' -path '*room*' 2>/dev/null | head"))
print("=== search for memory/GB mentions near room c0 training ===")
print(sh("grep -rls 'GB' /root/epic05_n1 2>/dev/null | head -3"))
