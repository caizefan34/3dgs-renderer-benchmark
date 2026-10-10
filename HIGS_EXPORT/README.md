# HiGS original source handoff exporter

This archive addresses a packaging error: `TRAINABLE_HIGS_INSTRUMENT_ORACLE_R14_MX.zip` contains report documents, not CUDA/Python source code.

On **MX**, run:

```bash
python3 export_higs_sources.py
```

The script reads the reported HiGS source tree and relevant R8/handoff documents, copies relevant `.cu/.cuh/.cpp/.h/.py` into a **new** directory, records checksums, and creates:

`/mnt/storage_pool/liaoyuanjun/HIGS_SOURCE_HANDOFF_ACTUAL.tar.gz`

Upload the generated `.tar.gz` into the conversation. If the script reports `MISSING_ROOT` or missing required files, provide the actual source directory. No original files are modified; binaries are SHA256-hashed rather than copied. The exporter also does not rebuild or import HiGS.

## Expected contents

- Actual `IntersectTile.cu` including the CUDA counting functions, not snippets quoted by another report
- Actual `setup.py`
- CUDA binding code (e.g. `ext.cpp`)
- Python frontend containing `rasterize_gaussian_higs_frozen`
- Original R8 execution script if available
- `SOURCE_MANIFEST.json`, `SHA256SUMS.txt`

Until those are included, a source-specific CUDA patch or trustworthy independent HiGS counting oracle cannot be validated.
