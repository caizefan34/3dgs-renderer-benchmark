#!/usr/bin/env python3
"""Read-only FastGS tile-prefix opportunity gate.

Run first with --manifest-only to freeze selected training-camera/tile IDs.
Then run with --execute-ideal --manifest camera_tile_manifest.json.  If and
only if the ideal gate passes, run --certify --manifest camera_tile_manifest.json.
This script never opens an image file and only writes into --outdir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CAMERA_TRAIN_INDICES = [0, 39, 77, 116, 155, 194, 232, 271]
TILE_SIZE = 16
TILES_PER_AXIS = 16
DEPTH_BINS = 64
SOURCE_FILES = [
    "submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/rasterizer_impl.cu",
    "submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu",
    "submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/rasterizer_impl.h",
    "submodules/diff-gaussian-rasterization_fastgs/rasterize_points.cu",
    "submodules/diff-gaussian-rasterization_fastgs/diff_gaussian_rasterization_fastgs/__init__.py",
    "gaussian_renderer/__init__.py",
    "scene/dataset_readers.py",
    "scene/cameras.py",
    "utils/camera_utils.py",
    "scene/gaussian_model.py",
    "train.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: dict) -> str:
    clean = dict(obj)
    clean.pop("manifest_sha256", None)
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_cameras_bin(path: Path) -> dict:
    """Use the author's COLMAP decoder; camera intrinsics are not held-out views."""
    from scene.colmap_loader import read_intrinsics_binary

    return read_intrinsics_binary(str(path))


def parse_images_bin(path: Path, selected_names: set[str] | None = None) -> list[dict]:
    """Read names for sorting; unpack pose only for selected training cameras.

    COLMAP's image record is [id:u32, qvec:4*f64, tvec:3*f64,
    camera_id:u32, name:cstr, npoints:u64, points:(x,y,id)*n].
    Held-out records' qvec/tvec are skipped as opaque bytes.
    """
    rows: list[dict] = []
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id_raw = f.read(4)
            pose_raw = f.read(56)
            camera_id_raw = f.read(4)
            if len(image_id_raw) != 4 or len(pose_raw) != 56 or len(camera_id_raw) != 4:
                raise RuntimeError("truncated COLMAP image record")
            image_id = struct.unpack("<I", image_id_raw)[0]
            camera_id = struct.unpack("<I", camera_id_raw)[0]
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise RuntimeError("truncated COLMAP image name")
                if ch == b"\0":
                    break
                name_bytes.extend(ch)
            full_name = name_bytes.decode("utf-8")
            npts_raw = f.read(8)
            if len(npts_raw) != 8:
                raise RuntimeError("truncated COLMAP point count")
            npts = struct.unpack("<Q", npts_raw)[0]
            f.seek(24 * npts, os.SEEK_CUR)
            basename = Path(full_name).name
            image_name = basename.split(".")[0]
            row = {"image_id": image_id, "image_name": image_name,
                   "full_name": full_name}
            if selected_names and image_name in selected_names:
                qvec = struct.unpack("<4d", pose_raw[:32])
                tvec = struct.unpack("<3d", pose_raw[32:56])
                row.update(camera_id=camera_id, qvec=list(qvec), tvec=list(tvec))
            rows.append(row)
    return rows


def make_manifest(checkout: Path, dataset: Path, ply: Path) -> dict:
    from scene.colmap_loader import qvec2rotmat
    from utils.graphics_utils import focal2fov

    raw_rows = parse_images_bin(dataset / "sparse/0/images.bin")
    sorted_rows = sorted(raw_rows, key=lambda x: x["image_name"])
    train_rows = [(i, r) for i, r in enumerate(sorted_rows) if i % 8 != 0]
    if len(train_rows) != 272:
        raise RuntimeError(f"expected 272 train cameras under eval split; got {len(train_rows)}")
    selected = [(idx, train_rows[idx][0], train_rows[idx][1])
                for idx in CAMERA_TRAIN_INDICES]
    if any("camera_id" in r for _, _, r in selected):
        raise RuntimeError("manifest discovery unexpectedly parsed camera poses")
    selected_names = {r["image_name"] for _, _, r in selected}
    posed_rows = parse_images_bin(dataset / "sparse/0/images.bin", selected_names)
    by_name = {r["image_name"]: r for r in posed_rows if "camera_id" in r}
    intrinsics = read_cameras_bin(dataset / "sparse/0/cameras.bin")

    cams = []
    for train_idx, sorted_idx, name_only in selected:
        r = by_name[name_only["image_name"]]
        intr = intrinsics[r["camera_id"]]
        if intr.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise RuntimeError(f"unsupported camera model: {intr.model}")
        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = float(intr.params[0])
        else:
            fx, fy = float(intr.params[0]), float(intr.params[1])
        ow, oh = int(intr.width), int(intr.height)
        down = (ow / 1600.0) if ow > 1600 else 1.0
        width, height = int(ow / down), int(oh / down)
        nx, ny = (width + TILE_SIZE - 1) // TILE_SIZE, (height + TILE_SIZE - 1) // TILE_SIZE
        txs = sorted({math.floor((k + 0.5) * nx / TILES_PER_AXIS)
                      for k in range(TILES_PER_AXIS)})
        tys = sorted({math.floor((k + 0.5) * ny / TILES_PER_AXIS)
                      for k in range(TILES_PER_AXIS)})
        tiles = [[tx, ty] for ty in tys for tx in txs]
        R = np.asarray(qvec2rotmat(r["qvec"]), dtype=np.float64).T
        cams.append({
            "training_camera_index": train_idx,
            "sorted_all_camera_index": sorted_idx,
            "image_name": r["image_name"],
            "colmap_image_id": r["image_id"],
            "colmap_camera_id": r["camera_id"],
            "full_effective_width": width,
            "full_effective_height": height,
            "tile_grid_width": nx,
            "tile_grid_height": ny,
            "tile_coordinates_xy": tiles,
            "fov_x_radians": float(focal2fov(fx, ow)),
            "fov_y_radians": float(focal2fov(fy, oh)),
            "R_transposed": R.tolist(),
            "T": r["tvec"],
        })
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    manifest = {
        "protocol": "TILE_PREFIX_TRANSMITTANCE_GATE_20260928",
        "git_commit": rev,
        "ply_path": str(ply),
        "ply_sha256": sha256(ply),
        "source_sha256": {f: sha256(checkout / f) for f in SOURCE_FILES},
        "split": {"all_camera_count": len(sorted_rows), "train_camera_count": len(train_rows),
                  "heldout_rule": "sorted image_name; indices divisible by 8 are test",
                  "selected_training_indices": CAMERA_TRAIN_INDICES},
        "render": {"mult": 0.5, "tile_size": TILE_SIZE,
                   "pixel_coordinates": "integer pixel centers, matching FastGS forward.cu",
                   "resolution_rule": "author loadCam resolution=-1; downscale width to 1600 only when wider"},
        "cameras": cams,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def align(offset: int, base_ptr: int, alignment: int = 128) -> int:
    absolute = base_ptr + offset
    return ((absolute + alignment - 1) & ~(alignment - 1)) - base_ptr


def take(raw: bytes, base_ptr: int, offset: int, count: int, dtype: str) -> tuple[np.ndarray, int]:
    dt = np.dtype(dtype)
    start = align(offset, base_ptr)
    end = start + count * dt.itemsize
    if end > len(raw):
        raise RuntimeError(f"buffer too short: need {end} bytes, have {len(raw)}")
    return np.frombuffer(raw, dtype=dt, count=count, offset=start).copy(), end


def geometry_arrays(geom, p: int) -> dict[str, np.ndarray]:
    base = int(geom.data_ptr())
    raw = geom.detach().cpu().numpy().tobytes()
    off = 0
    out = {}
    for name, n, dtype in [
        ("depth", p, "<f4"), ("clamped", p * 3, "|u1"),
        ("radii", p, "<i4"), ("xy", p * 2, "<f4"),
        ("cov3d", p * 6, "<f4"), ("conic_opacity", p * 4, "<f4"),
        ("rgb", p * 3, "<f4"),
    ]:
        a, off = take(raw, base, off, n, dtype)
        out[name] = a
    out["xy"] = out["xy"].reshape(p, 2)
    out["conic_opacity"] = out["conic_opacity"].reshape(p, 4)
    out["rgb"] = out["rgb"].reshape(p, 3)
    return out


def binning_arrays(binning, r: int) -> tuple[np.ndarray, np.ndarray]:
    base = int(binning.data_ptr())
    raw = binning.detach().cpu().numpy().tobytes()
    off = 0
    values, off = take(raw, base, off, r, "<u4")
    _, off = take(raw, base, off, r, "<u4")
    keys, off = take(raw, base, off, r, "<u8")
    return values, keys


def image_arrays(img, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    base = int(img.data_ptr())
    raw = img.detach().cpu().numpy().tobytes()
    n = w * h
    final_t, off = take(raw, base, 0, n, "<f4")
    n_contrib, _ = take(raw, base, off, n, "<u4")
    return final_t.reshape(h, w), n_contrib.reshape(h, w)


def cpu_tile(ids, xy_all, con_all, rgb_all, xvals, yvals):
    """Vectorized per-pixel replay of native forward equations for one tile."""
    xx, yy = np.meshgrid(xvals, yvals)
    valid = (xx >= 0) & (yy >= 0)
    pix_x = xx[valid].astype(np.float32)
    pix_y = yy[valid].astype(np.float32)
    npix = len(pix_x)
    T = np.ones(npix, dtype=np.float32)
    C = np.zeros((npix, 3), dtype=np.float32)
    done = np.zeros(npix, dtype=bool)
    last_contrib = np.zeros(npix, dtype=np.uint32)
    trigger_prefix = np.zeros(npix, dtype=np.int32)
    f32 = np.float32
    thresh = f32(1.0 / 255.0)
    for j, gid in enumerate(ids):
        active = ~done
        if not active.any():
            break
        x, y = xy_all[gid]
        co = con_all[gid]
        dx = f32(x) - pix_x[active]
        dy = f32(y) - pix_y[active]
        t1 = f32(f32(co[0] * dx) * dx)
        t2 = f32(f32(co[2] * dy) * dy)
        t3 = f32(f32(co[1] * dx) * dy)
        inside = f32(f32(t1 + t2) * f32(-0.5)) - t3
        power = inside.astype(np.float32, copy=False)
        eligible = power <= f32(0.0)
        expv = np.exp(power, dtype=np.float32)
        alpha = np.minimum(f32(0.99), f32(co[3]) * expv).astype(np.float32)
        eligible &= alpha >= thresh
        active_ids = np.flatnonzero(active)
        accept_ids = active_ids[eligible]
        if accept_ids.size:
            test_t = f32(T[accept_ids] * f32(f32(1.0) - alpha[eligible]))
            triggers = test_t < f32(1.0e-4)
            if triggers.any():
                tr_ids = accept_ids[triggers]
                done[tr_ids] = True
                trigger_prefix[tr_ids] = j + 1
            keep_ids = accept_ids[~triggers]
            if keep_ids.size:
                keep_alpha = alpha[eligible][~triggers]
                old_t = T[keep_ids].copy()
                C[keep_ids] = f32(C[keep_ids] + f32(rgb_all[gid][None, :] * keep_alpha[:, None]) * old_t[:, None])
                T[keep_ids] = test_t[~triggers]
                last_contrib[keep_ids] = np.uint32(j + 1)
    return {"T": T, "C": C, "n_contrib": last_contrib,
            "done": done, "trigger_prefix": trigger_prefix,
            "valid_x": pix_x.astype(np.int32), "valid_y": pix_y.astype(np.int32)}


def fp32_alpha_lower(co, xy, x0, x1, y0, y1) -> tuple[float, bool]:
    """Corner upper bound on q, with outward numerical margins.

    A continuous minimum check around power>0 catches the only discontinuity
    in the native alpha rule; uncertain entries get lower bound zero.
    """
    cxx, cxy, cyy, opacity = map(float, co)
    det = cxx * cyy - cxy * cxy
    if not (cxx > 0.0 and cyy > 0.0 and det > 0.0 and math.isfinite(det)):
        return 0.0, False
    corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    qvals = []
    for px, py in corners:
        dx, dy = float(xy[0]) - px, float(xy[1]) - py
        a, b, c = cxx * dx * dx, 2.0 * cxy * dx * dy, cyy * dy * dy
        qvals.append((a + b + c, abs(a) + abs(b) + abs(c)))
    # Inflate the largest q for float32 operation/rounding uncertainty.
    qmax = max(q + 16.0 * np.finfo(np.float32).eps * (m + 1.0) for q, m in qvals)
    # q minimum on a rectangle can occur on its interior or an edge.  Use the
    # continuous box minimum as a conservative guard against power>0 rounding.
    # If the box contains the mean, or is within the arithmetic error band,
    # abstain for this entry rather than assume native exp never sees power>0.
    xmin, xmax = x0 - float(xy[0]), x1 - float(xy[0])
    ymin, ymax = y0 - float(xy[1]), y1 - float(xy[1])
    if xmin <= 0.0 <= xmax and ymin <= 0.0 <= ymax:
        return 0.0, True
    # A conservative lower test based on the smallest eigenvalue of the conic.
    near_x = max(xmin, -xmax, 0.0)
    near_y = max(ymin, -ymax, 0.0)
    lam_min = 0.5 * (cxx + cyy - math.sqrt(max(0.0, (cxx-cyy)**2 + 4.0*cxy*cxy)))
    qmin_lower = max(0.0, lam_min) * (near_x * near_x + near_y * near_y)
    qerr = 16.0 * np.finfo(np.float32).eps * (qmax + 1.0)
    if qmin_lower <= qerr:
        return 0.0, True
    if not math.isfinite(opacity) or opacity <= 0.0:
        return 0.0, False
    alpha = min(0.99, opacity * math.exp(-0.5 * qmax))
    # Downward relative and absolute margins, then nextafter toward zero.
    alpha = max(0.0, alpha * (1.0 - 32.0 * np.finfo(np.float32).eps) - 4.0 * np.finfo(np.float32).eps)
    alpha = float(np.nextafter(np.float32(alpha), np.float32(0.0)))
    if alpha < (1.0 / 255.0):
        return 0.0, False
    return alpha, False


def save_view(checkout: Path, model_path: Path, camera: dict, outdir: Path,
              parity_tiles: int = 1) -> dict:
    import torch
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    from gaussian_renderer import render_fastgs

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pcd = GaussianModel(3)
    pcd.load_ply(str(model_path))
    pcd.active_sh_degree = 3
    R = np.asarray(camera["R_transposed"], dtype=np.float64)
    T0 = np.asarray(camera["T"], dtype=np.float64)
    wv = torch.tensor(getWorld2View2(R, T0, np.zeros(3), 1.0)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0,
                               fovX=camera["fov_x_radians"],
                               fovY=camera["fov_y_radians"]).transpose(0, 1).cuda()
    full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam = SimpleNamespace(
        FoVx=camera["fov_x_radians"], FoVy=camera["fov_y_radians"],
        image_width=camera["full_effective_width"], image_height=camera["full_effective_height"],
        world_view_transform=wv, full_proj_transform=full,
        camera_center=wv.inverse()[3, :3], image_name=camera["image_name"],
    )
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    pkg = render_fastgs(cam, pcd, pipe, bg, 0.5)
    output = pkg["render"]
    grad_fn = output.grad_fn
    if grad_fn is None or not hasattr(grad_fn, "saved_tensors"):
        raise RuntimeError("native rasterizer buffers are not exposed through autograd saved_tensors")
    saved = grad_fn.saved_tensors
    if len(saved) != 12:
        raise RuntimeError(f"expected 12 saved tensors, got {len(saved)}")
    geom, binning, img = saved[8], saved[9], saved[10]
    r = int(getattr(grad_fn, "num_rendered"))
    p, h, w = pcd.get_xyz.shape[0], cam.image_height, cam.image_width
    ga = geometry_arrays(geom, p)
    list_ids, keys = binning_arrays(binning, r)
    native_t, native_n = image_arrays(img, w, h)

    selected_tiles = camera["tile_coordinates_xy"]
    tile_ids = np.asarray([ty * ((w + 15) // 16) + tx for tx, ty in selected_tiles], dtype=np.int32)
    key_tiles = (keys >> np.uint64(32)).astype(np.uint32)
    if np.any(key_tiles >= ((w + 15) // 16) * ((h + 15) // 16)):
        raise RuntimeError("sorted native key contains an invalid tile ID")
    # Native point_list is already in exact tile/depth/tie order. Group by the
    # high 32-bit tile ID exactly as identifyTileRanges does in the source.
    offsets = np.searchsorted(key_tiles, tile_ids, side="left")
    ends = np.searchsorted(key_tiles, tile_ids, side="right")
    if np.any(key_tiles[offsets[offsets < r]] != tile_ids[offsets < r]):
        raise RuntimeError("could not reconstruct a native tile range from sorted keys")
    picked = []
    for tx, ty in selected_tiles:
        tile = ty * ((w + 15) // 16) + tx
        a = int(np.searchsorted(key_tiles, tile, side="left"))
        b = int(np.searchsorted(key_tiles, tile, side="right"))
        picked.append((tile, tx, ty, list_ids[a:b].astype(np.int32), keys[a:b].copy()))

    counts = []
    ideal_sum = 0
    parity_records = []
    tile_arrays = []
    for tile, tx, ty, ids, native_keys in picked:
        x0, y0 = tx * 16, ty * 16
        xvals = np.arange(x0, min(x0 + 16, w), dtype=np.float32)
        yvals = np.arange(y0, min(y0 + 16, h), dtype=np.float32)
        # Native rendered tile IDs and sorted key depth order are both exact.
        if len(ids):
            depth_key = (native_keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            if np.any(depth_key[1:] < depth_key[:-1]):
                raise RuntimeError("native point_list keys are not nondecreasing in depth order")
        replay = cpu_tile(ids, ga["xy"], ga["conic_opacity"], ga["rgb"], xvals, yvals)
        n_valid = len(replay["valid_x"])
        all_done = bool(replay["done"].all()) if n_valid else False
        ideal = (len(ids) - int(replay["trigger_prefix"].max())) if all_done else 0
        if ideal < 0:
            raise RuntimeError("invalid ideal removable count")
        ideal_sum += ideal

        pixel_native_n = native_n[y0:min(y0 + 16, h), x0:min(x0 + 16, w)].reshape(-1)
        pixel_native_t = native_t[y0:min(y0 + 16, h), x0:min(x0 + 16, w)].reshape(-1)
        # Save list/raw per-tile bounds and native outputs for all sampled tiles.
        tile_arrays.append({"tile_id": tile, "tx": tx, "ty": ty, "ids": ids,
                            "keys": native_keys, "ideal": ideal,
                            "all_pixels_done": all_done, "prefix": int(replay["trigger_prefix"].max()) if all_done else None,
                            "native_n": pixel_native_n.astype(np.uint32),
                            "native_t": pixel_native_t.astype(np.float32)})
        if len(parity_records) < parity_tiles:
            native_rgb = output[:, y0:min(y0 + 16, h), x0:min(x0 + 16, w)].detach().cpu().numpy()
            cpu_rgb = np.zeros_like(native_rgb)
            cpu_t = np.zeros_like(pixel_native_t, dtype=np.float32)
            cpu_n = np.zeros_like(pixel_native_n, dtype=np.uint32)
            # The selected tiles are interior except if sampling includes the final tile.
            valid_i = 0
            for py in range(y0, min(y0 + 16, h)):
                for px in range(x0, min(x0 + 16, w)):
                    cpu_rgb[:, py-y0, px-x0] = replay["C"][valid_i]
                    cpu_t[valid_i] = replay["T"][valid_i]
                    cpu_n[valid_i] = replay["n_contrib"][valid_i]
                    valid_i += 1
            mismatch_n = np.flatnonzero(cpu_n != pixel_native_n)
            diff_rgb = np.abs(cpu_rgb - native_rgb)
            diff_t = np.abs(cpu_t - pixel_native_t)
            parity_records.append({
                "tile_id": tile, "tx": tx, "ty": ty,
                "pixel_count": int(cpu_n.size),
                "n_contrib_mismatch_count": int(mismatch_n.size),
                "n_contrib_mismatches": [{"flat_pixel": int(i), "cpu": int(cpu_n[i]),
                                            "native": int(pixel_native_n[i])} for i in mismatch_n],
                "max_abs_rgb_error": float(diff_rgb.max(initial=0.0)),
                "rgb_mismatch_count_at_1e-5": int(np.count_nonzero(diff_rgb > 1e-5)),
                "max_abs_final_T_error": float(diff_t.max(initial=0.0)),
                "final_T_mismatch_count_at_1e-6": int(np.count_nonzero(diff_t > 1e-6)),
            })
        counts.append({"tile_id": tile, "tx": tx, "ty": ty,
                       "L": len(ids), "ideal_removable": ideal,
                       "ideal_fraction": (ideal / len(ids)) if len(ids) else 0.0,
                       "all_pixels_done": all_done,
                       "ideal_retained_prefix": tile_arrays[-1]["prefix"]})

    npz_path = outdir / f"native_tiles_{camera['image_name']}.npz"
    flat_ids = np.concatenate([a["ids"] for a in tile_arrays]) if tile_arrays else np.empty(0, np.int32)
    flat_keys = np.concatenate([a["keys"] for a in tile_arrays]) if tile_arrays else np.empty(0, np.uint64)
    ptr = np.cumsum([0] + [len(a["ids"]) for a in tile_arrays]).astype(np.int64)
    tile_n = np.stack([a["native_n"] for a in tile_arrays])
    tile_t = np.stack([a["native_t"] for a in tile_arrays])
    np.savez_compressed(npz_path, tile_ids=np.asarray([a["tile_id"] for a in tile_arrays], np.int32),
                        tile_xy=np.asarray([[a["tx"], a["ty"]] for a in tile_arrays], np.int16),
                        offsets=ptr, gaussian_ids=flat_ids, sorted_keys=flat_keys,
                        native_n_contrib=tile_n, native_final_T=tile_t,
                        projected_xy=ga["xy"], conic_opacity=ga["conic_opacity"], rgb=ga["rgb"],
                        depth=ga["depth"])

    L_sum = int(sum(c["L"] for c in counts))
    return {
        "image_name": camera["image_name"],
        "training_camera_index": camera["training_camera_index"],
        "width": w, "height": h, "tile_grid": [((w + 15) // 16), ((h + 15) // 16)],
        "sampled_tile_count": len(counts), "original_entries": L_sum,
        "ideal_removable": ideal_sum,
        "ideal_fraction": ideal_sum / L_sum if L_sum else 0.0,
        "parity": parity_records,
        "tiles": counts,
        "raw_npz": npz_path.name,
    }


def certify_view(camera: dict, raw_path: Path) -> dict:
    raw = np.load(raw_path)
    xy_all = raw["projected_xy"]
    con_all = raw["conic_opacity"]
    ids_all = raw["gaussian_ids"]
    offsets = raw["offsets"]
    coords = raw["tile_xy"]
    cert_tiles = []
    total_entries = total_cert = corner_evals = zero_guarded = bin_slots = 0
    lower_all = []
    for i, (tx, ty) in enumerate(coords.tolist()):
        ids = ids_all[offsets[i]:offsets[i + 1]]
        length = len(ids)
        total_entries += length
        bin_slots += min(DEPTH_BINS, length)
        x0, y0 = int(tx) * TILE_SIZE, int(ty) * TILE_SIZE
        x1 = min(x0 + TILE_SIZE, camera["full_effective_width"]) - 1
        y1 = min(y0 + TILE_SIZE, camera["full_effective_height"]) - 1
        lower = np.zeros(length, dtype=np.float64)
        uncertain = np.zeros(length, dtype=bool)
        for j, gid in enumerate(ids):
            lower[j], uncertain[j] = fp32_alpha_lower(
                con_all[gid], xy_all[gid], x0, x1, y0, y1)
        lower_all.append(lower)
        corner_evals += 4 * length
        zero_guarded += int(uncertain.sum())
        certified = 0
        if length:
            positions = np.arange(length, dtype=np.int64)
            bins = [np.flatnonzero((positions * DEPTH_BINS // length) == b)
                    for b in range(DEPTH_BINS)]
            bins = [b for b in bins if b.size]
            trans_upper = np.float64(1.0)
            for bin_positions in bins:
                for j in bin_positions:
                    trans_upper = np.nextafter(
                        trans_upper * (1.0 - lower[j]), np.float64(np.inf))
                if trans_upper < 1.0e-4:
                    certified = length - int(bin_positions[-1]) - 1
                    break
        total_cert += certified
        cert_tiles.append({
            "tile_id": int(raw["tile_ids"][i]), "tx": int(tx), "ty": int(ty),
            "L": length, "certified_removable": certified,
            "certified_fraction": certified / length if length else 0.0,
        })
    flat_lower = np.concatenate(lower_all) if lower_all else np.empty(0, dtype=np.float64)
    lower_path = raw_path.with_name(raw_path.stem + "_certificate.npz")
    np.savez_compressed(lower_path, offsets=offsets, alpha_lower=flat_lower)
    return {
        "image_name": camera["image_name"],
        "training_camera_index": camera["training_camera_index"],
        "original_entries": total_entries,
        "certified_removable": total_cert,
        "certified_fraction": total_cert / total_entries if total_entries else 0.0,
        "entry_tile_corner_bound_evaluations": corner_evals,
        "entries_zeroed_for_power_skip_numerical_guard": zero_guarded,
        "tile_bin_storage_slots": bin_slots,
        "tiles": cert_tiles,
        "raw_certificate_npz": lower_path.name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--execute-ideal", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--parity-tiles", type=int, default=1)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.outdir / "camera_tile_manifest.json")
    if args.manifest_only:
        manifest = make_manifest(args.checkout, args.dataset, args.model)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"],
                          "selected_cameras": [c["image_name"] for c in manifest["cameras"]],
                          "tiles_per_camera": [len(c["tile_coordinates_xy"]) for c in manifest["cameras"]]}, indent=2))
        return
    if not (args.execute_ideal or args.certify) or not args.manifest:
        raise SystemExit("use --manifest-only, --execute-ideal --manifest PATH, or --certify --manifest PATH")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if canonical_hash(manifest) != manifest.get("manifest_sha256"):
        raise RuntimeError("camera/tile manifest SHA256 mismatch")
    current_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip()
    if current_rev != manifest["git_commit"]:
        raise RuntimeError("author checkout HEAD changed since manifest freeze")
    if sha256(args.model) != manifest["ply_sha256"]:
        raise RuntimeError("30K PLY changed since manifest freeze")
    if args.execute_ideal:
        views = []
        for camera in manifest["cameras"]:
            print(f"Rendering selected train camera {camera['training_camera_index']}: {camera['image_name']}", flush=True)
            views.append(save_view(args.checkout, args.model, camera, args.outdir, args.parity_tiles))
        total_entries = sum(v["original_entries"] for v in views)
        total_ideal = sum(v["ideal_removable"] for v in views)
        ideal_cams = sum(v["ideal_fraction"] >= 0.20 for v in views)
        gate = {
            "manifest_sha256": manifest["manifest_sha256"],
            "git_commit": manifest["git_commit"], "ply_sha256": manifest["ply_sha256"],
            "source_sha256": manifest["source_sha256"], "views": views,
            "pooled": {"original_entries": total_entries,
                       "ideal_removable": total_ideal,
                       "ideal_fraction": total_ideal / total_entries if total_entries else 0.0,
                       "ideal_cameras_at_20pct": ideal_cams},
            "ideal_gate_pass": (total_ideal / total_entries >= 0.40 if total_entries else False) and ideal_cams >= 4,
        }
        out = args.outdir / "ideal_gate_results.json"
        out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"results": str(out), "pooled": gate["pooled"],
                          "ideal_gate_pass": gate["ideal_gate_pass"]}, indent=2))
        return
    ideal_path = args.outdir / "ideal_gate_results.json"
    if not ideal_path.exists():
        raise RuntimeError("ideal result must exist in the output directory before certification")
    ideal_result = json.loads(ideal_path.read_text(encoding="utf-8"))
    if ideal_result.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("ideal result uses a different frozen manifest")
    if not ideal_result.get("ideal_gate_pass"):
        raise RuntimeError("frozen ideal gate failed; certification is prohibited")
    certified_views = []
    for camera, ideal_view in zip(manifest["cameras"], ideal_result["views"]):
        certified_views.append(certify_view(camera, args.outdir / ideal_view["raw_npz"]))
    total_entries = sum(v["original_entries"] for v in certified_views)
    total_cert = sum(v["certified_removable"] for v in certified_views)
    cert_cams = sum(v["certified_fraction"] >= 0.125 for v in certified_views)
    result = {
        "manifest_sha256": manifest["manifest_sha256"],
        "ideal_gate_pass": True, "views": certified_views,
        "pooled": {"original_entries": total_entries,
                   "certified_removable": total_cert,
                   "certified_fraction": total_cert / total_entries if total_entries else 0.0,
                   "certified_cameras_at_12_5pct": cert_cams,
                   "entry_tile_corner_bound_evaluations": sum(v["entry_tile_corner_bound_evaluations"] for v in certified_views),
                   "tile_bin_storage_slots": sum(v["tile_bin_storage_slots"] for v in certified_views)},
        "certified_gate_pass": (total_cert / total_entries >= 0.25 if total_entries else False) and cert_cams >= 4,
    }
    out = args.outdir / "certificate_gate_results.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(out), "pooled": result["pooled"],
                      "certified_gate_pass": result["certified_gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
