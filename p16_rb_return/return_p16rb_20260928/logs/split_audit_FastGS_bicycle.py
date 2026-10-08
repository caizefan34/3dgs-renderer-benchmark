
import json, sys, os
# subprocess puts the SCRIPT's directory (logs/) on sys.path, not cwd
sys.path.insert(0, os.getcwd())
from scene.dataset_readers import sceneLoadTypeCallbacks
info = sceneLoadTypeCallbacks["Colmap"](sys.argv[1], "images", True, int(sys.argv[2]))
def row(c):
    return {"name": c.image_name, "w": c.width, "h": c.height,
            "path_exists": os.path.exists(c.image_path)}
out = {"n_train": len(info.train_cameras), "n_test": len(info.test_cameras),
       "train": [row(c) for c in info.train_cameras],
       "test": [row(c) for c in info.test_cameras],
       "test_names": [c.image_name for c in info.test_cameras]}
print("AUDIT_JSON " + json.dumps(out))
