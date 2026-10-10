import subprocess
# check process state and open files, and any python stack via py-spy if available
r=subprocess.run(["bash","-lc",
  "for p in 4104112 4104114; do echo \"== PID $p ==\"; cat /proc/$p/status 2>/dev/null | grep -E 'State|VmRSS|Threads'; echo 'wchan:'; cat /proc/$p/wchan 2>/dev/null; echo; done"],
  capture_output=True,text=True)
print(r.stdout)
# Check if gsplat_cuda.so is loaded
r2=subprocess.run(["bash","-lc","ls -la /proc/4104112/fd 2>/dev/null | grep -iE 'shm|txext|gsplat' | head; echo '---maps---'; grep -c gsplat /proc/4104112/maps 2>/dev/null"],capture_output=True,text=True)
print(r2.stdout)
