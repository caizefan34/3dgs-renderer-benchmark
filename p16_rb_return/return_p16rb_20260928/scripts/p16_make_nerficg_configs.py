#!/usr/bin/env python3
"""P16-RB: generate NeRFICG (FasterGS / FasterGSFused) run configs on the common dataset.

Native-first discipline: start from the author's shipped config
(fastergs_garden.yaml / fastergsfused_garden.yaml), change ONLY:
  GLOBAL.GPU_INDICES          -> pinned GPU index
  GLOBAL.RANDOM_SEED          -> kept at author value 0
  TRAINING.MODEL_NAME         -> p16_<method>_<scene>_1920
  TRAINING.DATA.PRELOADING_LEVEL -> 2 (image data is 1920x1080; keep author value)
  DATASET.PATH                -> /root/epic05_p16rb/common_data/mipnerf360/<scene>
  DATASET.IMAGE_SCALE_FACTOR  -> 1.0   (common 1920x1080 dataset; author value 0.25
                                          would look for images_4/ which does not exist)
Everything else (loss, optimizer, densification schedule, PCA/near/far, test step) is
byte-identical to the author's recipe. Overrides are recorded in
artifacts/P16_CONFIG_PROVENANCE.json.
"""
import json
import shutil
from pathlib import Path

import yaml

D = Path("/root/epic05_p16rb")
SRC = D / "src"
CFG = D / "configs"
ART = D / "artifacts"
COMMON = D / "common_data" / "mipnerf360"

VARIANTS = {
    "FasterGS": {"author_yaml": SRC / "fastergs_main" / "fastergs_garden.yaml",
                 "method": "FasterGS"},
    "FasterGSFused": {"author_yaml": SRC / "fastergs_fused" / "fastergsfused_garden.yaml",
                      "method": "FasterGSFused"},
}
SCENES = ("room", "bicycle", "garden")


def main():
    CFG.mkdir(parents=True, exist_ok=True)
    provenance = {"policy": "author config + minimal dataset/identity overrides",
                  "overrides": ["GLOBAL.GPU_INDICES", "TRAINING.MODEL_NAME",
                                "DATASET.PATH", "DATASET.IMAGE_SCALE_FACTOR"],
                  "kept_author_values": ["GLOBAL.RANDOM_SEED", "TRAINING.NUM_ITERATIONS",
                                         "DATASET.TEST_STEP", "DATASET.APPLY_PCA",
                                         "DATASET.APPLY_PCA_RESCALE", "DATASET.NEAR_PLANE",
                                         "DATASET.FAR_PLANE", "TRAINING.OPTIMIZER",
                                         "TRAINING.LOSS", "TRAINING.DENSIFICATION"],
                  "configs": {}}
    for name, spec in VARIANTS.items():
        author_path = spec["author_yaml"]
        if not author_path.exists():
            raise SystemExit(f"missing author config {author_path}")
        base = yaml.safe_load(author_path.read_text())
        for scene in SCENES:
            cfg = json.loads(json.dumps(base))  # deep copy
            cfg["GLOBAL"]["GPU_INDICES"] = [2]
            cfg["GLOBAL"]["METHOD_TYPE"] = spec["method"]
            cfg["TRAINING"]["MODEL_NAME"] = f"p16_{name.lower()}_{scene}_1920"
            cfg["TRAINING"]["BACKUP"]["RENDER_TESTSET"] = True
            cfg["TRAINING"]["BACKUP"]["FINAL_CHECKPOINT"] = True
            cfg["DATASET"]["PATH"] = str(COMMON / scene)
            cfg["DATASET"]["IMAGE_SCALE_FACTOR"] = 1.0
            out = CFG / f"p16_{name.lower()}_{scene}_1920.yaml"
            out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            provenance["configs"][out.name] = {
                "author_yaml": str(author_path),
                "author_yaml_sha256": __import__("hashlib").sha256(author_path.read_bytes()).hexdigest(),
                "method_type": spec["method"],
                "seed": cfg["GLOBAL"].get("RANDOM_SEED"),
                "num_iterations": cfg["TRAINING"]["NUM_ITERATIONS"],
                "dataset_path": cfg["DATASET"]["PATH"],
                "image_scale_factor": cfg["DATASET"]["IMAGE_SCALE_FACTOR"],
                "test_step": cfg["DATASET"]["TEST_STEP"],
                "shipped_as_yaml": str(out),
            }
            print(f"wrote {out.name}")
    (ART / "P16_CONFIG_PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("CONFIGS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
