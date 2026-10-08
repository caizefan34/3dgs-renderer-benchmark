import shutil, hashlib, os
src="/root/epic05_p12/publication_trainer_p12_holdout8.py"
dst="/tmp/holdout8_trainer.py"
shutil.copy(src,dst)
b=open(dst,"rb").read()
print("SHA256", hashlib.sha256(b).hexdigest())
print("BYTES", len(b))
print("LINES", b.count(b"\n"))