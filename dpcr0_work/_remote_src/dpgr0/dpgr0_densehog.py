import sys, time, torch
gpu=int(sys.argv[1])
torch.cuda.set_device(gpu); torch.manual_seed(0)
# Dense compute hog: repeated large matmuls to keep SMs busy like a foreign training job
a=torch.randn(2048,2048,device=gpu,dtype=torch.float32)
b=torch.randn(2048,2048,device=gpu,dtype=torch.float32)
print("DENSE HOG gpu",gpu,"start",flush=True)
i=0
while True:
    for _ in range(20):
        c=torch.mm(a,b); a=c*0.999999+1e-6
    torch.cuda.synchronize()
    i+=1
    if i%20==0: print("DENSE churn",i,flush=True)