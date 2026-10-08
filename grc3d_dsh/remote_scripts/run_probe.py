#!/usr/bin/env python3
"""EPIC-05 probe wrapper: bind the frozen GMAIN strong baseline FIRST, then run
the package's own probe CLI unchanged (the package has no EPIC bootstrap).

Usage: python scripts/run_probe.py --gpu 3 --backend gsplat --bundle ... [probe args]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gx_common as G  # noqa: E402

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--gpu", type=int, default=None)
known, rest = ap.parse_known_args()
gpu = known.gpu if known.gpu is not None else int(os.environ.get("GRC3D_GPU", "3"))

# Compat shim (runner-side only, package file untouched): grc3d/probe.py calls
# hashlib.file_digest, added in Python 3.11; this env is Python 3.10.
import hashlib  # noqa: E402
if not hasattr(hashlib, "file_digest"):
    def _file_digest(fileobj, digest, /, *, _bufsize=2 ** 18):
        h = hashlib.new(digest)
        for chunk in iter(lambda: fileobj.read(_bufsize), b""):
            h.update(chunk)
        return h
    hashlib.file_digest = _file_digest
    print(f"[run_probe] shim: hashlib.file_digest backported for "
          f"python {sys.version.split()[0]}", flush=True)

t0 = __import__("time").time()
t, ident = G.boot_gmain(gpu)
G.add_grc3d_path()
import gsplat  # noqa: E402
print(f"[run_probe] gmain tree={gsplat.__file__} so={ident['so_sha256'][:16]}... "
      f"gpu={gpu}", flush=True)

sys.argv = [sys.argv[0]] + rest
from grc3d.probe import main  # noqa: E402

main()
print(f"[run_probe] wall_s={__import__('time').time() - t0:.1f}", flush=True)
