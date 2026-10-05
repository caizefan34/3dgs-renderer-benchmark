import json
m = json.load(open("/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH/config/scene_manifest.json"))
print("repo_head", m.get("repo_head"))
for k, v in m["scenes"].items():
    print(k, "N=", v.get("checkpoint_num_points"), "dev=", v.get("dev_camera_ids"))
