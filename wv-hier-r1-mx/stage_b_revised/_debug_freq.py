"""Debug: is per-pixel inclusion p, or is my kron freq computation wrong?
Direct pixel->child/parent mapping, no kron."""
import math, numpy as np
p=0.25; h,w=65,77
cth,ctw=math.ceil(h/4),math.ceil(w/4)
pth,ptw=math.ceil(h/16),math.ceil(w/16)
# explicit pixel -> child index and pixel -> parent index
pi=np.arange(h); pj=np.arange(w)
pix_child_r=np.minimum(pi//4,cth-1)[:,None]*np.ones((1,w),dtype=int)
pix_child_c=np.minimum(pj//4,ctw-1)[None,:]*np.ones((h,w),dtype=int)
pix_par_r=np.minimum(pi//16,pth-1)[:,None]*np.ones((1,w),dtype=int)
pix_par_c=np.minimum(pj//16,ptw-1)[None,:]*np.ones((h,w),dtype=int)
N=60000
rng=np.random.default_rng(0)
sel4=np.zeros((h,w))
for s in range(N):
    child=rng.random((cth,ctw))
    sel_ch=child<p
    sel4+=sel_ch[pix_child_r,pix_child_c]
print("G4 direct per-pixel inclusion:", sel4.mean()/(h*w), "expect", p)
print("  selected children mean:", (rng.random((0,)) , None) and "")
# also count how many children are in-bounds vs total
print("cth,ctw,pth,ptw:",cth,ctw,pth,ptw)
# kron-based reproduction
sel4k=np.zeros((h,w))
for s in range(N):
    child=rng.random((cth,ctw))
    sel_ch=child<p
    sp=np.kron(sel_ch,np.ones((4,4),dtype=bool))[:h,:w]
    sel4k+=sp
print("G4 kron per-pixel inclusion:", sel4k.mean()/(h*w))
print("kron total px shape check:", np.kron(np.ones((cth,ctw),dtype=bool),np.ones((4,4),dtype=bool)).shape)
# Is the issue that kron child (i,j) block does not align with pix_child mapping?
# pix_child uses floor(/4); kron block (i,j)=[4i:4i+4]. For i=16: block [64:68] clipped to [64:65].
# floor(64/4)=16 -> child row 16. aligned. For pixel row 64: pix_child_r=16. kron row 64 in block 16. aligned.
# So mapping identical. Then why 0.30?
print("sum of all child sizes (should=5005):",
      np.kron(np.ones((cth,ctw),dtype=bool),np.ones((4,4),dtype=bool))[:h,:w].sum())
