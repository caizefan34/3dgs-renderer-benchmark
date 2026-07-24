#!/usr/bin/env python
"""Encode or decode deterministic 3DGS compression baseline artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compression_artifact import decode_ply, encode_ply  # noqa: E402
from schema_validation import validate_schema  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    encode = subparsers.add_parser("encode")
    encode.add_argument("--input", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--manifest", type=Path, required=True)
    encode.add_argument("--codec", choices=["block-float", "tile-codebook"], required=True)
    encode.add_argument("--block-size", type=int, default=4096)
    encode.add_argument("--tile-resolution", type=int, default=8)
    decode = subparsers.add_parser("decode")
    decode.add_argument("--input", type=Path, required=True)
    decode.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.operation == "decode":
        decode_ply(args.input, args.output)
        return 0

    manifest = encode_ply(
        args.input, args.output, args.codec,
        block_size=args.block_size, tile_resolution=args.tile_resolution,
    )
    schema = json.loads(
        (ROOT / "benchmark" / "schemas" / "compression-artifact.schema.json").read_text(encoding="utf-8")
    )
    validate_schema(manifest, schema)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "codec": args.codec,
        "compressed_bytes": manifest["compressed_artifact"]["bytes"],
        "compression_ratio": manifest["compressed_artifact"]["compression_ratio"],
        "encode_ms": manifest["timings_ms"]["encode"],
        "decode_validation_ms": manifest["timings_ms"]["decode_validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
