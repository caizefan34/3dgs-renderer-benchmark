import subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
key=r"C:\Users\36570\.ssh\mx_public.key"
port="26372"
host="liaoyuanjun@36.140.146.31"
def run(remote, timeout=120):
    cmd=["ssh","-i",key,"-p",port,"-o","StrictHostKeyChecking=no","-o","ConnectTimeout=25","-o","ServerAliveInterval=15","-o","ServerAliveCountMax=4",host,"tr -d '\\r' | bash -s"]
    p=subprocess.run(cmd,input=remote,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
    out=p.stdout
    # strip SSH banner "warring"
    lines=[l for l in out.splitlines() if "warring" not in l.lower()]
    return p.returncode,"\n".join(lines),p.stderr
if __name__=="__main__":
    remote=sys.stdin.read()
    rc,out,err=run(remote,timeout=int(sys.argv[1]) if len(sys.argv)>1 else 120)
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write("\n--STDERR--\n"+err)
    sys.exit(0)


