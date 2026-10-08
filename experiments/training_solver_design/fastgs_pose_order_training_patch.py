"""Make a separate FastGS trainer that changes only camera order."""

import argparse
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one source occurrence, found {source.count(old)}: {old[:80]}")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("author_source", type=Path)
    parser.add_argument("patched_output", type=Path)
    args = parser.parse_args()
    source = args.author_source.read_text(encoding="utf-8")
    source = replace_once(source, "import torch\n", "import torch\nimport json\n")
    helper = '''
def pose_order_stack(cameras):
    order_path = os.environ["FASTGS_CAMERA_ORDER_FILE"]
    with open(order_path, encoding="utf-8") as stream:
        names = json.load(stream)
    by_name = {camera.image_name: camera for camera in cameras}
    if len(by_name) != len(cameras) or len(names) != len(cameras) or set(names) != set(by_name):
        raise ValueError("Camera order must contain each training camera exactly once")
    return [by_name[name] for name in reversed(names)]


'''
    source = replace_once(source, "def training(dataset, opt, pipe,", helper + "def training(dataset, opt, pipe,")
    source = replace_once(source,
                          "    viewpoint_stack = scene.getTrainCameras().copy()\n"
                          "    viewpoint_indices = list(range(len(viewpoint_stack)))\n",
                          "    viewpoint_stack = pose_order_stack(scene.getTrainCameras().copy())\n"
                          "    viewpoint_indices = list(range(len(viewpoint_stack)))\n")
    source = replace_once(source,
                          "        if not viewpoint_stack:\n"
                          "            viewpoint_stack = scene.getTrainCameras().copy()\n"
                          "            viewpoint_indices = list(range(len(viewpoint_stack)))\n",
                          "        if not viewpoint_stack:\n"
                          "            viewpoint_stack = pose_order_stack(scene.getTrainCameras().copy())\n"
                          "            viewpoint_indices = list(range(len(viewpoint_stack)))\n")
    source = replace_once(source, "        viewpoint_cam = viewpoint_stack.pop(rand_idx)\n",
                          "        viewpoint_cam = viewpoint_stack.pop()\n")
    args.patched_output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
