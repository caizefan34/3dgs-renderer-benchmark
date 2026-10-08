import json, os
base = "h0_run"
os.makedirs(base, exist_ok=True)
CELLS = ["room@5000","room@15000","room@25000","bicycle@5000","bicycle@15000","bicycle@25000"]
snap = "/root/epic05_n1/runs/n1snap_c0f_s42_{scene}"
HASH = {
 ("room",25000):"b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d",
 ("room",15000):"b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
 ("bicycle",15000):"273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
 ("bicycle",25000):"73ae5436cc44a2c25dca26cb226e522047465e65b671717ac043578cbd7e21ea",
}
inv = {
  "audit_type":"H0_CHECKPOINT_INVENTORY",
  "host":"EPIC-05 (dsw-933622-64cc845c7d-lqswc), 8x A100-SXM4-80GB",
  "frozen_trajectory": snap,
  "frozen_trajectory_note":"n1snap = save-only edit of frozen P12 trainer (sha256 49a79538..); reproduced canonical seed-42 stream bit-exactly; snapshots saved at 10K/15K/20K/25K only. No 5K snapshot was ever written.",
  "prereg_pinned":{
    "c0_ext_so":"d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11",
    "core_gsplat_so":"935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
    "room25k":"b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d"
  },
  "binary_verification":{
    "c0_ext_so":{"path":"/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so","sha256":"d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11","match":True},
    "core_gsplat_so":{"path":"/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so","sha256":"935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e","match":True}
  },
  "cells":{}
}
missing=[]; present=[]
for c in CELLS:
    scene, it = c.split("@"); itn=int(it)
    p = snap.format(scene=scene)+("/ckpt_%d.pt"%itn)
    exists = itn in (10000,15000,20000,25000)
    h = HASH.get((scene,itn))
    inv["cells"][c]={"scene":scene,"iter":itn,"path":p,"exists_in_frozen_trajectory":exists,
        "ckpt_sha256":h,"pinned_by_prereg":(scene=="room" and itn==25000),
        "hash_matches_prereg":(h=="b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d") if (scene=="room" and itn==25000) else None}
    (present if exists else missing).append(c)
inv["missing_required_cells"]=missing
inv["present_required_cells"]=present
inv["verdict"]="H0_CHECKPOINT_BLOCKER" if missing else "CHECKPOINTS_OK"
inv["blocker_reason"]=("Required cells room@5000 and bicycle@5000 are ABSENT from the frozen C0 trajectory "
 "(/root/epic05_n1/runs/n1snap_c0f_s42_{room,bicycle} saved only 10K/15K/20K/25K). No ckpt_5000.pt exists for room or bicycle "
 "anywhere in the frozen trajectory. Per H0_PREREG.md sec.2 and DSH_HANDOFF item 9: emit H0_CHECKPOINT_BLOCKER and STOP; do not substitute. "
 "5K is the only active-densification-regime cell; substituting 10K would change the scientific meaning.")
json.dump(inv, open(os.path.join(base,"H0_CHECKPOINT_INVENTORY.json"),"w"), indent=2, ensure_ascii=False)
json.dump({"audit_type":"H0_SOURCE_IDENTITY","c0_ext_so_sha256":"d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11",
 "core_gsplat_so_sha256":"935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e","room25k_ckpt_sha256":"b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d",
 "all_match_prereg":True,"note":"Binaries verified live on EPIC-05. room@25K hash matches prereg inline pin exactly."},
 open(os.path.join(base,"H0_SOURCE_IDENTITY.json"),"w"), indent=2, ensure_ascii=False)
json.dump({"audit_type":"H0_ENVIRONMENT","host":"EPIC-05","hostname":"dsw-933622-64cc845c7d-lqswc","gpus":["NVIDIA A100-SXM4-80GB x8"],
 "python":"3.12.13","note":"Local machine (RTX 5070 8GB, torch 2.13+cu130, gsplat 1.4.0, Win, Python 3.13) is NOT used for any H0 run, per user directive."},
 open(os.path.join(base,"H0_ENVIRONMENT.json"),"w"), indent=2, ensure_ascii=False)
json.dump({"verdict":"H0_CHECKPOINT_BLOCKER","selected_fixed_family":None,"missing_cells":missing,"present_cells":present,
 "authority":"H0_PREREG.md section 2 + DSH_HANDOFF item 9",
 "action":"STOP. No scientific arms (NATIVE/E2D/OPA) run. No trajectories/summary produced. Prerequisite fail-closed, not a scientific DROP/PROMOTE.",
 "decision_fork":[
  "A) Accept H0_CHECKPOINT_BLOCKER (strict prereg letter): H0 cannot run without the two 5K cells.",
  "B) Authorize bit-exact 5K regeneration: re-run the FROZEN n1snap trainer (sha256 49a79538..) from seed 42 to save ckpt_5000.pt for room & bicycle. N1 report proves bit-exact canonical-stream reproduction, so the 5K state is deterministic. DEVIATION from strict frozen-manifest-hash / do-not-substitute reading (no pre-registered 5K hash exists to verify against); requires explicit user sign-off."]},
 open(os.path.join(base,"H0_VERDICT.json"),"w"), indent=2, ensure_ascii=False)
print("WROTE:", sorted(os.listdir(base)))
print("missing:", missing)
print("present:", present)
