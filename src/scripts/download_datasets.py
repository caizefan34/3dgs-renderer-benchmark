"""
Dataset manifest tool for 3DGS Renderer Benchmark.
Lists planned scenes and downloads entries only after they are published.

Usage:
    python src/scripts/download_datasets.py --list
    python src/scripts/download_datasets.py --dataset garden
    python src/scripts/download_datasets.py --dataset all --output-dir data/scenes
"""
import os, sys, json, argparse, hashlib, urllib.request, shutil
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "scenes")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "scenes", "scenes.json")


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found: {MANIFEST_PATH}")
        print("Run scene generator first or download from GitHub releases.")
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def list_datasets():
    manifest = load_manifest()
    scenes = manifest.get("scenes", [])
    if not scenes:
        print("No datasets configured. Check data/scenes/scenes.json")
        return
    print(f"{'Name':20s} {'Status':10s} {'Source':20s} {'Size':8s} {'Type':15s} {'Gaussians':>10s}")
    print("-" * 90)
    for s in scenes:
        print(f"{s['name']:20s} {s.get('status', 'planned'):10s} {s['source']:20s} {str(s['size_gb'])+'GB':8s} "
              f"{s['type']:15s} {s.get('num_gaussians', '?'):>10,}")
    print()


def download_file(url, dest_path, expected_md5=None):
    """Download with progress."""
    print(f"  Downloading {url}")
    print(f"  -> {dest_path}")

    def reporthook(block, block_size, total_size):
        downloaded = block * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024) if total_size > 0 else 0
        if total > 0:
            pct = min(100, downloaded / total * 100)
            print(f"\r    {downloaded:.1f}/{total:.1f} MB ({pct:.0f}%)", end="", flush=True)
        else:
            print(f"\r    {downloaded:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

    # Verify MD5 if provided
    if expected_md5 and expected_md5 != "TODO":
        md5 = hashlib.md5()
        with open(dest_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5.update(chunk)
        if md5.hexdigest() != expected_md5:
            print(f"  MD5 mismatch! Expected {expected_md5}, got {md5.hexdigest()}")
            return False

    return True


def download_dataset(name, output_dir=None):
    manifest = load_manifest()
    scenes = manifest.get("scenes", [])
    scene = next((s for s in scenes if s["name"] == name), None)
    if not scene:
        print(f"Dataset '{name}' not found in manifest.")
        print("Available datasets:")
        list_datasets()
        return False

    if scene.get("status") != "available":
        status = scene.get("status", "planned")
        print(f"Dataset '{name}' has status '{status}' and no packaged direct download.")
        if scene.get("reference_manifest"):
            print(f"See data/scenes/{scene['reference_manifest']} for official URLs and hashes.")
        else:
            print("Use src/scripts/generate_scene.py for the current synthetic benchmark.")
        return False

    output_dir = output_dir or os.path.join(REPO_ROOT, "data")
    scene_dir = os.path.join(output_dir, name)
    os.makedirs(scene_dir, exist_ok=True)

    ply_path = os.path.join(scene_dir, "scene.ply")
    if os.path.exists(ply_path):
        size_mb = os.path.getsize(ply_path) / (1024 * 1024)
        print(f"  Already exists: {ply_path} ({size_mb:.1f} MB)")
        return True

    url = scene.get("ply_url", "")
    if not url:
        print(f"  No download URL for '{name}'. Set ply_url in scenes.json")
        return False

    md5 = scene.get("ply_md5", None)
    ok = download_file(url, ply_path, expected_md5=md5)
    if ok:
        size_mb = os.path.getsize(ply_path) / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.1f} MB -> {ply_path}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Download 3DGS benchmark datasets")
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset name (e.g., garden, bicycle) or 'all'")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: data/)")
    p.add_argument("--list", action="store_true", help="List available datasets")
    args = p.parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset:
        os.makedirs(args.output_dir, exist_ok=True) if args.output_dir else None
        if args.dataset == "all":
            manifest = load_manifest()
            scenes = manifest.get("scenes", [])
            success = True
            for s in scenes:
                print(f"\nDownloading {s['name']}...")
                success = download_dataset(s["name"], args.output_dir) and success
        else:
            success = download_dataset(args.dataset, args.output_dir)
        if not success:
            raise SystemExit(1)
    else:
        p.print_help()
        print("\nAvailable datasets:")
        list_datasets()


if __name__ == "__main__":
    main()
