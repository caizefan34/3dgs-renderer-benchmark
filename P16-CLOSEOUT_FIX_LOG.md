# P16-RB Closeout Fix Log — 2026-09-28

**Scope:** statistical-text corrections in the P16-RB closeout artifacts only.
No experiment was launched, re-run, or re-aggregated. No raw JSON record, timing value,
quality metric, or verdict was modified. `P16_VERDICT.json` is byte-identical to before
(verdict value `RESET_SCENE_DEPENDENT`, `strong_methods = []` — unchanged).

## 1. Why

Two statistics in the closeout prose did not match the canonical census
(`artifacts/P16_BOTTLENECK_CENSUS.json`; 12 modern cells = 4 methods x 3 scenes):

1. The quality-box admission fraction was written against a denominator of **nine** modern
   cells; the correct denominator is **twelve** (FasterGSFused/room remains the single
   qualifying cell at 1.73x — that finding is unchanged).
2. The bottleneck-classification summary claimed **"9 of 12 cells"** as the
   execution-dominated group; the correct prereg §15 classification is
   **7/12 EXECUTION_DOMINATED, 2/12 MIXED, 3/12 CONVERGENCE_DOMINATED**
   (FasterGS room/bicycle/garden, FasterGSFused room/bicycle/garden, FastGS garden =
   EXECUTION_DOMINATED; FastGS room/bicycle = MIXED; SADGS room/bicycle/garden =
   CONVERGENCE_DOMINATED — exactly as the §8 census table already showed).

## 2. Text changes

### 2.1 `p16_rb/P16_REPORT.md` and `p16_rb_return/return_p16rb_20260928/artifacts/P16_REPORT.md`

The two copies are byte-identical; the same three edits were applied to both.

**Line 12 — modern cell count:**

```text
OLD: - Nine method x scene cells were measured (4 modern methods + P12 C0, 3 scenes), all at common resolution (1920x1080, resolution gate PASS everywhere) with one common metric pipeline on the same test split.
NEW: - Twelve method x scene cells were measured (4 modern methods + P12 C0, 3 scenes), all at common resolution (1920x1080, resolution gate PASS everywhere) with one common metric pipeline on the same test split.
```

**Line 14 — quality-box admission fraction:**

```text
OLD: - Under the prereg §10 C0-matched quality box, exactly **one of nine** cells qualifies: FasterGSFused on room (1.73x, quality inside box). Every other modern cell misses at least one box constraint, most by structural PSNR margins.
NEW: - Under the prereg §10 C0-matched quality box, exactly **one of twelve** cells qualifies: FasterGSFused on room (1.73x, quality inside box). Every other modern cell misses at least one box constraint, most by structural PSNR margins.
```

**Line 150 — bottleneck classification (replaced fragment only; everything after
"raster backward:" is unchanged):**

```text
OLD: - **The field is execution-dominated wherever it trains the full recipe:** 9 of 12 cells at 0.71-0.89 exec share, dominated by raster backward: FasterGS 4.5-8.7 ms/iter (+1.3-4.7 ms separate optimizer), FasterGSFused 5.2-11.1 ms/iter with the optimizer fused into backward (separate optimizer = 0), SADGS 16.3-21.7 ms/iter (+2.6-9.5 ms optimizer). FastGS room/bicycle sit just under the 0.70 exec threshold (MIXED) because its loss (~1.10 ms) + sort + other costs are proportionally larger at 6-7 ms totals.
NEW: - **The field is execution-dominated wherever it trains the full recipe:** the prereg §15 classification is 7 of 12 cells EXECUTION_DOMINATED, 2 of 12 MIXED, and 3 of 12 CONVERGENCE_DOMINATED; exec shares of 0.71-0.89 are dominated by raster backward: FasterGS 4.5-8.7 ms/iter (+1.3-4.7 ms separate optimizer), FasterGSFused 5.2-11.1 ms/iter with the optimizer fused into backward (separate optimizer = 0), SADGS 16.3-21.7 ms/iter (+2.6-9.5 ms optimizer). FastGS room/bicycle sit just under the 0.70 exec threshold (MIXED) because its loss (~1.10 ms) + sort + other costs are proportionally larger at 6-7 ms totals.
```

All other report lines are unchanged; the §8 census table already carried the correct
per-cell classes and was not touched. Report file: 25052 -> 25165 bytes; sha256
`fb741b33eab7d7a60143f97aac8215a7bf3c99f2eb0289e755c45bf7e8d6ecac` ->
`6ef8a51105ccd98defaf5e47b2fc0cd48a71174771767aa43b507e96208c3a80`.

### 2.2 `README_RETURN.md` (3 byte-identical copies, same one-line change)

- `p16_rb_return/return_p16rb_20260928/README_RETURN.md`
- `p16_rb_return/return_p16rb_20260928/bundle/README_RETURN.md`
- `p16_rb/bundle/README_RETURN.md`

**Line 6:**

```text
OLD: **Bottom line:** exactly one of nine modern cells is quality-admissible and faster on the triad — FasterGSFused/room at 1.7319x (PSNR 31.7282, inside the prereg §10 C0 box). No admissible method exceeds 1.73x on any scene, so **P12 C0 remains the conservative strong baseline** for future ">3x" claims.
NEW: **Bottom line:** exactly one of twelve modern cells is quality-admissible and faster on the triad — FasterGSFused/room at 1.7319x (PSNR 31.7282, inside the prereg §10 C0 box). No admissible method exceeds 1.73x on any scene, so **P12 C0 remains the conservative strong baseline** for future ">3x" claims.
```

3707 -> 3709 bytes; sha256 `db7074407c5720435e3a58a5700c818db6408bc82b5f6afa0495aa11b3b5dfe4` ->
`7785e38491c03f5a96490b5e48f7bf79fc577ae80b9fe10fe1d34101dd890413` (all three copies).

## 3. Manifest changes

`p16_rb_return/return_p16rb_20260928/artifacts/MANIFEST_P16.json` — regenerated by
recomputing **every** entry from the actual package files (same generation method and
serialization as the original `p16_package_return.py` output, key order preserved):

| field | old | new |
|---|---|---|
| `generated` | `2026-09-28T12:15:29+0800` | `2026-09-28T21:30:51+0800` |
| `root` | `/root/epic05_p16rb` | unchanged |
| `files["artifacts/P16_REPORT.md"].sha256` | `fb741b33eab7d7a60143f97aac8215a7bf3c99f2eb0289e755c45bf7e8d6ecac` | `6ef8a51105ccd98defaf5e47b2fc0cd48a71174771767aa43b507e96208c3a80` |
| `files["artifacts/P16_REPORT.md"].bytes` | 25052 | 25165 |

All other **61 entries are byte-identical** to the previous manifest (verified by
recomputation against the files). `bundle/MANIFEST.json` needed no change: it covers the
8 bundle reference files only and lists neither `P16_REPORT.md` nor `README_RETURN.md`.

## 4. Package rebuild

`p16rb_return_20260928.tar.gz` rebuilt from `p16_rb_return/return_p16rb_20260928/` with the
same archive layout as the original `p16_return_tar.sh` flow (top-level directory
`return_p16rb_20260928/`; identical 522-file set):

| item | old | new |
|---|---|---|
| tarball sha256 | `fe9e9a8db68ba28bcd30838af4621d5160b927d28d048a3654d2330e5fa8c021` | `2795a7ef152c122ccb3691bd1c5e76683355261c7fa9808ba535f076d22bc0bc` |
| tarball bytes | 1133099 | 1036938 |

(The new tarball is smaller only because the rebuild compresses with gzip level 9 instead of
GNU tar's default level; content, not container compression, defines the package.)
`p16rb_return_20260928.tar.gz.sha256` was updated to the new hash in the original
`sha256sum` line format, keeping the box-path convention:
`2795a7ef152c122ccb3691bd1c5e76683355261c7fa9808ba535f076d22bc0bc  /root/epic05_p16rb/p16rb_return_20260928.tar.gz`

## 5. Verification (all checks pass)

Pre-edit baseline: original tarball matched its sidecar; the extracted tree matched the
tarball byte-for-byte (522/522 files); all 62 manifest entries matched the package files.

Post-edit:
- New tarball hash matches the new sidecar.
- New tarball contents are identical to the extracted tree (522/522 files, every hash equal);
  readable with the system `tar` (bsdtar) as well.
- All 62 `MANIFEST_P16.json` entries match the actual package files (0 mismatches).
- Both `P16_REPORT.md` copies and all three `README_RETURN.md` copies are byte-identical
  to each other.
- No occurrence of `one of nine`, `Nine method`, or `9 of 12 cells` remains in the report
  or the READMEs.
- Untouched (hash-verified against the manifest): every `runs/`, `curves/`, `profiles/`
  JSON and every other artifact JSON, including `P16_VERDICT.json`
  (`RESET_SCENE_DEPENDENT`, `strong_methods = []`).

## 6. Complete list of files changed

| file | change |
|---|---|
| `p16_rb/P16_REPORT.md` | 3 text corrections (L12, L14, L150) |
| `p16_rb_return/return_p16rb_20260928/artifacts/P16_REPORT.md` | same 3 corrections (package copy) |
| `p16_rb_return/return_p16rb_20260928/README_RETURN.md` | 1 text correction (L6) |
| `p16_rb_return/return_p16rb_20260928/bundle/README_RETURN.md` | same correction (package copy) |
| `p16_rb/bundle/README_RETURN.md` | same correction (working copy) |
| `p16_rb_return/return_p16rb_20260928/artifacts/MANIFEST_P16.json` | regenerated (report entry + `generated`) |
| `p16rb_return_20260928.tar.gz` | rebuilt |
| `p16rb_return_20260928.tar.gz.sha256` | updated to new tarball hash |
