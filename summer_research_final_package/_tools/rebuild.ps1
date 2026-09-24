# Final clean rebuild of the package directory structure.
$root = "C:\Users\36570\3dgs-renderer-benchmark\summer_research_final_package"

# ---- 0. Remove the whole package tree (content was already mirrored into it; sources live in ../). ----
if (Test-Path $root) { Remove-Item $root -Recurse -Force }

# ---- 1. Recreate canonical structure ----
$dirs = @(
  "00_README_FIRST","01_PROJECT_OVERVIEW","02_TIMELINE","03_FINAL_MATRIX","04_CRITICAL_FINDINGS",
  "05_PROTOCOL_AND_DATASETS","06_C25_BACKDROP_DEPTH","07_TILE_GEOMETRY_HARDWARE",
  "08_C42","09_C1_FORWARD_SORT","10_C51_SELECTIVE_BACKWARD","11_C49_ATTRIBUTE",
  "12_CANDIDATE_C","13_TRAINABLE_HIGS","14_NEGATIVE_RESULTS","15_REPRODUCIBILITY",
  "16_RAW_DATA","17_APPENDICES"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $root $d) | Out-Null }

# ---- 2. Move already-produced digest/README content into canonical places ----
# The drafts folder was regenerated; collect from wherever subagents left files.
$srcBase = "C:\Users\36570\3dgs-renderer-benchmark"

# Copy source-level digest files if they exist at the repo root under "reports" etc. is NOT needed:
# the digests were created in package dirs directly. Gather any *.md already in package paths.
# (Simplest: keep _drafts as the master digest area, and copy to sections.)
foreach ($k in @(
  @{ s = "s1_protocol.md";      d = "05_PROTOCOL_AND_DATASETS" },
  @{ s = "s2_c42.md";           d = "08_C42" },
  @{ s = "s3_tile_geom.md";     d = "07_TILE_GEOMETRY_HARDWARE" },
  @{ s = "s4_forward_sort.md";  d = "09_C1_FORWARD_SORT" },
  @{ s = "s5_backdrop_c25.md";  d = "06_C25_BACKDROP_DEPTH" },
  @{ s = "s5_backward_c25.md";  d = "06_C25_BACKDROP_DEPTH" },
  @{ s = "s5b_c51.md";          d = "10_C51_SELECTIVE_BACKWARD" },
  @{ s = "s7_c49.md";           d = "11_C49_ATTRIBUTE" },
  @{ s = "s8_candidate_c.md";   d = "12_CANDIDATE_C" },
  @{ s = "s9_higs.md";          d = "13_TRAINABLE_HIGS" }
)) {
  foreach ($cand in @((Join-Path $srcBase "summer_research_final_package\glare_drafts"),$k.s)) {
    $full = Join-Path $srcBase $cand
    if (Test-Path $full) {
      Copy-Item $full (Join-Path (Join-Path $root $k.d) "EVIDENCE_DIGEST.md") -Force
      Write-Host "copied $k.s -> $($k.d)"
      break
    }
  }
}

# ---- 3. Copy the root-level curated artifacts into place ----
foreach ($f in @("00_README_FIRST.md","FINAL_REPORT_DRAFT_BRIEF.md","MANIFEST.md","RESEARCH_TIMELINE.md","VERSION_INFO.json")) {
  $p = Join-Path $srcBase "summer_research_final_package\gla_root"
  if (-not $p) {}
}
# The canonical artifacts are already at package root; we removed the tree so re-copy from the source ones:
$root_files_src = $srcBase
foreach ($f in @("00_README_FIRST.md","MANIFEST.md","RESEARCH_TIMELINE.md","FINAL_RESULT_MATRIX.csv","version-info.json")) {
  $p = Join-Path $srcBase $f
  if (Test-Path $p) { Copy-Item $p (Join-Path $root $f) -Force; Write-Host "copied root $f" }
}

# ---- 4. Move _drafts content into 17_APPENDICES/DRAFTS and remove empty scaffolding ----
$drafts = Join-Path $srcBase "summer_research_final_package\glare_drafts"
if (Test-Path $drafts) {
  Copy-Item (Join-Path $drafts "*") (Join-Path $root "17_APPENDICES") -Recurse -Force -ErrorAction SilentlyContinue
}

# Copy canonical ROOT .md files into 00_README_FIRST etc.
Copy-Item (Join-Path $root "00_README_FIRST.md") (Join-Path $root "00_README_FIRST\README_FIRST.md") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "RESEARCH_TIMELINE.md") (Join-Path $root "02_TIMELINE\RESEARCH_TIMELINE.md") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "FINAL_RESULT_MATRIX.csv") (Join-Path $root "03_FINAL_MATRIX\FINAL_RESULT_MATRIX.csv") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "version-info.json") (Join-Path $root "17_APPENDICES\version-info.json") -Force -ErrorAction SilentlyContinue

Write-Host "Rebuild complete."
Get-ChildItem $root | Select-Object -ExpandProperty Name