import json
cs = json.load(open("CVT0_CAMERA_SEMANTICS.json"))
pn = json.load(open("CVT0_PANELS.json"))
print("room cell:", json.dumps(cs["cells"]["room@10000"], indent=0)[:300])
print("room panel_0 cams:", pn["cells"]["room@10000"]["panel_cams"]["panel_0"])
print("room holdout cams:", pn["cells"]["room@10000"]["holdout_cams"])
print("bicycle panel_1 cams:", pn["cells"]["bicycle@10000"]["panel_cams"]["panel_1"])
# verify 3 panels x 8 unique
for cell in pn["cells"]:
    pc = pn["cells"][cell]["panel_cams"]
    allc = pc["panel_0"]+pc["panel_1"]+pc["panel_2"]
    assert len(pc["panel_0"])==8 and len(pc["panel_1"])==8 and len(pc["panel_2"])==8
    assert len(set(allc))==24, (cell, len(set(allc)))
print("all cells: 3 panels x 8 unique = 24 unique OK")
