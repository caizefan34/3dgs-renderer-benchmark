import glob, os, torch
for f in sorted(glob.glob("/dev/shm/wv0_out/states/*.pt")):
    try:
        s=torch.load(f, weights_only=False)
        ncond=len(s.get("conditions",{}))
        w=s.get("work",{}).get("per_G",{})
        print(f"{os.path.basename(f):28s} N={s.get('N'):>9,} cams={s.get('n_cams')} "
              f"cond={ncond} G16_isects={w.get('16',{}).get('dense_isects',0):,}")
    except Exception as e:
        print(f"{os.path.basename(f):28s} ERROR {e}")
