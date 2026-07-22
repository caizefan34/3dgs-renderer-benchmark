"""Deterministic, common-representation compression baselines for 3DGS PLY files."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_FLOAT_TYPES = {"float", "float32"}


@dataclass(frozen=True)
class PlyLayout:
    count: int
    properties: list[str]
    data_offset: int

    @property
    def dtype(self) -> np.dtype:
        return np.dtype([(name, "<f4") for name in self.properties])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_memmap(array) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_ply_layout(path: Path) -> PlyLayout:
    properties: list[str] = []
    count = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        if handle.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("compression baselines require binary_little_endian PLY")
        in_vertex = False
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            line = raw.decode("ascii").strip()
            if line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
                elif count is not None and int(parts[2]):
                    raise ValueError("PLY elements after vertex are not supported")
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                if len(parts) != 3 or parts[1] not in _FLOAT_TYPES:
                    raise ValueError("canonical compression track requires float32 vertex properties")
                properties.append(parts[2])
            elif line == "end_header":
                break
        offset = handle.tell()
    if not count or not properties:
        raise ValueError("PLY has no vertex payload")
    expected = offset + count * 4 * len(properties)
    if path.stat().st_size != expected:
        raise ValueError("PLY payload size does not match its float32 vertex declaration")
    return PlyLayout(count, properties, offset)


def read_binary_ply(path: Path) -> tuple[PlyLayout, np.memmap]:
    layout = read_ply_layout(path)
    records = np.memmap(
        path, dtype=layout.dtype, mode="r", offset=layout.data_offset,
        shape=(layout.count,),
    )
    return layout, records


def _write_header(handle, layout: PlyLayout) -> None:
    lines = [
        "ply", "format binary_little_endian 1.0",
        f"element vertex {layout.count}",
        *(f"property float {name}" for name in layout.properties),
        "end_header", "",
    ]
    handle.write("\n".join(lines).encode("ascii"))


def _fixed_zip_entry(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with source.open("rb") as input_handle, archive.open(info, "w") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, 8 * 1024 * 1024)


def _write_archive(path: Path, metadata: dict, arrays: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        info = zipfile.ZipInfo("metadata.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        for name in sorted(arrays):
            _fixed_zip_entry(archive, arrays[name], name)


def _quantize(values: np.ndarray, minimum: np.ndarray, scale: np.ndarray, maximum: int, dtype) -> np.ndarray:
    safe_scale = np.where(scale == 0, 1.0, scale)
    return np.rint((values - minimum) / safe_scale).clip(0, maximum).astype(dtype)


def _encode_block_float(records, layout: PlyLayout, temp: Path, block_size: int) -> tuple[dict, dict[str, Path]]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    block_count = (layout.count + block_size - 1) // block_size
    quantized_path = temp / "quantized.npy"
    minima_path = temp / "minima.npy"
    scales_path = temp / "scales.npy"
    quantized = np.lib.format.open_memmap(
        quantized_path, mode="w+", dtype="<u2", shape=(layout.count, len(layout.properties))
    )
    minima = np.lib.format.open_memmap(
        minima_path, mode="w+", dtype="<f4", shape=(block_count, len(layout.properties))
    )
    scales = np.lib.format.open_memmap(
        scales_path, mode="w+", dtype="<f4", shape=(block_count, len(layout.properties))
    )
    for block in range(block_count):
        start, end = block * block_size, min(layout.count, (block + 1) * block_size)
        values = np.column_stack([records[name][start:end] for name in layout.properties])
        low = values.min(axis=0).astype("<f4")
        scale = ((values.max(axis=0) - low) / 65535.0).astype("<f4")
        minima[block], scales[block] = low, scale
        quantized[start:end] = _quantize(values, low, scale, 65535, "<u2")
    quantized.flush(); minima.flush(); scales.flush()
    for array in (quantized, minima, scales):
        _close_memmap(array)
    metadata = {"codec": "block-float", "block_size": block_size, "quant_bits": 16}
    return metadata, {"quantized.npy": quantized_path, "minima.npy": minima_path, "scales.npy": scales_path}


def _encode_tile_codebook(records, layout: PlyLayout, temp: Path, resolution: int) -> tuple[dict, dict[str, Path]]:
    if resolution < 1 or resolution > 32:
        raise ValueError("tile_resolution must be between 1 and 32")
    for name in ("x", "y", "z"):
        if name not in layout.properties:
            raise ValueError("tile-codebook requires x/y/z properties")
    xyz = np.column_stack([records[name] for name in ("x", "y", "z")])
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    span = np.where(high == low, 1.0, high - low)
    coords = np.floor((xyz - low) / span * resolution).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    tile_ids = coords[:, 0] + resolution * (coords[:, 1] + resolution * coords[:, 2])
    order = np.argsort(tile_ids, kind="stable")
    tile_count = resolution ** 3
    counts = np.bincount(tile_ids, minlength=tile_count)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype("<i8")

    high_props = [
        name for name in layout.properties
        if name in {"x", "y", "z", "opacity"}
        or name.startswith("scale_") or name.startswith("rot_") or name.startswith("f_dc_")
    ]
    low_props = [name for name in layout.properties if name not in high_props]
    q16_path, q8_path = temp / "quantized16.npy", temp / "quantized8.npy"
    minima16_path, scales16_path = temp / "minima16.npy", temp / "scales16.npy"
    minima8_path, scales8_path = temp / "minima8.npy", temp / "scales8.npy"
    offsets_path = temp / "tile_offsets.npy"
    q16 = np.lib.format.open_memmap(q16_path, mode="w+", dtype="<u2", shape=(layout.count, len(high_props)))
    q8 = np.lib.format.open_memmap(q8_path, mode="w+", dtype="u1", shape=(layout.count, len(low_props)))
    min16 = np.lib.format.open_memmap(minima16_path, mode="w+", dtype="<f4", shape=(tile_count, len(high_props)))
    scale16 = np.lib.format.open_memmap(scales16_path, mode="w+", dtype="<f4", shape=(tile_count, len(high_props)))
    min8 = np.lib.format.open_memmap(minima8_path, mode="w+", dtype="<f4", shape=(tile_count, len(low_props)))
    scale8 = np.lib.format.open_memmap(scales8_path, mode="w+", dtype="<f4", shape=(tile_count, len(low_props)))
    np.save(offsets_path, offsets, allow_pickle=False)

    for tile in range(tile_count):
        start, end = int(offsets[tile]), int(offsets[tile + 1])
        if start == end:
            min16[tile] = 0; scale16[tile] = 0; min8[tile] = 0; scale8[tile] = 0
            continue
        indices = order[start:end]
        for props, quantized, minima, scales, maximum, dtype in (
            (high_props, q16, min16, scale16, 65535, "<u2"),
            (low_props, q8, min8, scale8, 255, "u1"),
        ):
            if not props:
                continue
            values = np.column_stack([records[name][indices] for name in props])
            tile_low = values.min(axis=0).astype("<f4")
            tile_scale = ((values.max(axis=0) - tile_low) / maximum).astype("<f4")
            minima[tile], scales[tile] = tile_low, tile_scale
            quantized[start:end] = _quantize(values, tile_low, tile_scale, maximum, dtype)
    for array in (q16, q8, min16, scale16, min8, scale8):
        array.flush()
        _close_memmap(array)
    metadata = {
        "codec": "tile-codebook", "tile_resolution": resolution,
        "high_precision_properties": high_props, "low_precision_properties": low_props,
        "high_quant_bits": 16, "low_quant_bits": 8,
    }
    arrays = {
        "quantized16.npy": q16_path, "quantized8.npy": q8_path,
        "minima16.npy": minima16_path, "scales16.npy": scales16_path,
        "minima8.npy": minima8_path, "scales8.npy": scales8_path,
        "tile_offsets.npy": offsets_path,
    }
    return metadata, arrays


def _load_archive(archive_path: Path, temp: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        archive.extractall(temp)
    arrays = {
        path.name: np.load(path, mmap_mode="r", allow_pickle=False)
        for path in temp.glob("*.npy")
    }
    return metadata, arrays


def decode_ply(archive_path: Path, output_path: Path, chunk_size: int = 65536) -> None:
    archive_path, output_path = Path(archive_path), Path(output_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata, arrays = _load_archive(archive_path, Path(temp_dir))
        layout = PlyLayout(metadata["count"], metadata["properties"], 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            _write_header(handle, layout)
            output_dtype = layout.dtype
            codec = metadata["codec"]
            if codec == "block-float":
                block_size = metadata["block_size"]
                q, minima, scales = arrays["quantized.npy"], arrays["minima.npy"], arrays["scales.npy"]
                for start in range(0, layout.count, chunk_size):
                    end = min(layout.count, start + chunk_size)
                    block_ids = np.arange(start, end) // block_size
                    values = minima[block_ids] + q[start:end].astype(np.float32) * scales[block_ids]
                    rows = np.empty(end - start, dtype=output_dtype)
                    for index, name in enumerate(layout.properties):
                        rows[name] = values[:, index]
                    handle.write(rows.tobytes())
            elif codec == "tile-codebook":
                offsets = arrays["tile_offsets.npy"]
                high_props, low_props = metadata["high_precision_properties"], metadata["low_precision_properties"]
                property_index = {name: index for index, name in enumerate(layout.properties)}
                for tile in range(len(offsets) - 1):
                    tile_start, tile_end = int(offsets[tile]), int(offsets[tile + 1])
                    for start in range(tile_start, tile_end, chunk_size):
                        end = min(tile_end, start + chunk_size)
                        matrix = np.empty((end - start, len(layout.properties)), dtype="<f4")
                        if high_props:
                            values = arrays["minima16.npy"][tile] + arrays["quantized16.npy"][start:end].astype(np.float32) * arrays["scales16.npy"][tile]
                            for index, name in enumerate(high_props):
                                matrix[:, property_index[name]] = values[:, index]
                        if low_props:
                            values = arrays["minima8.npy"][tile] + arrays["quantized8.npy"][start:end].astype(np.float32) * arrays["scales8.npy"][tile]
                            for index, name in enumerate(low_props):
                                matrix[:, property_index[name]] = values[:, index]
                        rows = np.empty(end - start, dtype=output_dtype)
                        for index, name in enumerate(layout.properties):
                            rows[name] = matrix[:, index]
                        handle.write(rows.tobytes())
            else:
                raise ValueError(f"unsupported codec: {codec}")
        for array in arrays.values():
            _close_memmap(array)


def encode_ply(
    source_path: Path, archive_path: Path, codec: str,
    *, block_size: int = 4096, tile_resolution: int = 8,
) -> dict:
    source_path, archive_path = Path(source_path).resolve(), Path(archive_path).resolve()
    layout, records = read_binary_ply(source_path)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        if codec == "block-float":
            codec_metadata, arrays = _encode_block_float(records, layout, temp, block_size)
            parameters = {"block_size": block_size, "quant_bits": 16}
        elif codec == "tile-codebook":
            codec_metadata, arrays = _encode_tile_codebook(records, layout, temp, tile_resolution)
            parameters = {"tile_resolution": tile_resolution, "geometry_bits": 16, "sh_rest_bits": 8}
        else:
            raise ValueError(f"unsupported codec: {codec}")
        archive_metadata = {
            "format_version": 1, "count": layout.count, "properties": layout.properties,
            **codec_metadata,
        }
        _write_archive(archive_path, archive_metadata, arrays)
    _close_memmap(records)
    encode_ms = (time.perf_counter() - started) * 1000.0
    artifact_sha = _sha256(archive_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        validation_path = Path(temp_dir) / "decoded.ply"
        decode_started = time.perf_counter()
        decode_ply(archive_path, validation_path)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        decoded_layout = read_ply_layout(validation_path)
        if decoded_layout.count != layout.count or decoded_layout.properties != layout.properties:
            raise ValueError("decoded PLY layout mismatch")
    source_bytes, compressed_bytes = source_path.stat().st_size, archive_path.stat().st_size
    return {
        "schema_version": "1.0", "status": "artifact_ready",
        "source": {
            "path": str(source_path), "sha256": _sha256(source_path), "bytes": source_bytes,
            "gaussian_count": layout.count, "properties": layout.properties,
        },
        "codec": {
            "id": codec, "version": "1", "parameters": parameters,
            "common_representation_compatible": True, "retraining_required": False,
        },
        "compressed_artifact": {
            "path": str(archive_path), "sha256": artifact_sha, "bytes": compressed_bytes,
            "compression_ratio": source_bytes / compressed_bytes,
        },
        "decode": {
            "output_format": "binary_little_endian_ply", "artifact_sha256": artifact_sha,
            "cpu_only": True, "peak_decode_vram_mb": 0.0,
        },
        "timings_ms": {"encode": encode_ms, "decode_validation": decode_ms},
        "provenance": {
            "benchmark_commit": _git_commit(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(os.sys.argv),
        },
    }
