import sys, time, torch
gpu=int(sys.argv[1])
torch.cuda.set_device(gpu); torch.manual_seed(0)
big=[torch.empty(512,1024,1024,device=gpu,dtype=torch.float32) for _ in range(10)]
for b in big: b.fill_(1.0)
print("HOG holding 20GB on gpu",gpu,flush=True)
i=0
while True:
    junk=[torch.empty(1024,1024,device=gpu,dtype=torch.float32) for _ in range(4)]
    del junk
    torch.cuda.synchronize()
    i+=1
    if i%50==0: print("HOG churn",i,flush=True)
    time.sleep(0.05)