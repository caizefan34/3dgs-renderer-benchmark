import json, os, shutil
# Pick a subset of camera indices spread across the trajectory; copy cameras.json + selected images
DATA=r"C:\Users\36570\3dgs-renderer-benchmark\data"
OUT=r"C:\Users\36570\3dgs-renderer-benchmark\_wv0_datapush"
N_SEL=24
for scene in ("room","garden","bicycle"):
    cj=os.path.join(DATA,"official","mipnerf360",scene,"cameras.json")
    cams=json.load(open(cj))
    n=len(cams)
    idxs=sorted(set(int(i*n/N_SEL) for i in range(N_SEL)))
    # official dir
    od=os.path.join(OUT,"official","mipnerf360",scene)
    os.makedirs(od,exist_ok=True)
    shutil.copyfile(cj, os.path.join(od,"cameras.json"))
    sel_cams=[cams[i] for i in idxs]
    json.dump(sel_cams, open(os.path.join(od,"cameras.json"),"w"))  # subset
    # images
    imgdir=None
    for cand in ("images","images_2","images_4","images_8"):
        d=os.path.join(DATA,"datasets","mipnerf360",scene,cand)
        if os.path.isdir(d): imgdir=d; break
    oimg=os.path.join(OUT,"datasets","mipnerf360",scene,"images")
    os.makedirs(oimg,exist_ok=True)
    copied=0
    for i in idxs:
        name=cams[i]["img_name"]
        for ext in (".JPG",".jpg",".png",".JPG".lower()):
            src=os.path.join(imgdir,name+ext)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(oimg,name+ext)); copied+=1; break
    print(scene, "cams", n, "selected", len(idxs), "images_copied", copied)
print("OUT", OUT)
