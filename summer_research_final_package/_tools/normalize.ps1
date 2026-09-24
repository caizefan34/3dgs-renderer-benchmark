# Normalize the package: merge duplicate dirs, remove empties, enforce canonical layout.
$root = "C:\Users\36570\3dgs-renderer-benchmark\summer_research_final_package"
Set-Location $root

function Get-FileHash-Id($path) {
    try { (Get-FileHash -Algorithm SHA256 -Path $path).Hash } catch { "ERR" }
}

# --- Map of canonical -> sources (merge order: keep FIRST non-conflicting file) ---
$merge = @{
  "06_BACKDROP_C25"      = @("06_Backdrop_C25","06_BACKDROP_C25","06_BACKWARD_C25","06_BACKDROP_PROFILING_C25","06_BACKWARD_PROFILING_C25","06_C25_BACKDROP_DEPTH","10_BACKDROP_PROFILING_C25","10_BACKWARD_PROFILING_C25","10_BACKDROP_C25","10_BACKDROP_BACKDROP")
  "07_TILE_GEOMETRY_HARDWARE" = @("07_TILE_GEOMETRY_HARDWARE","07_TileGeometry_Hardware")
  "08_C42"               = @("08_C42")
  "09_C1_FORWARD_SORT"   = @("09_C1_FORWARD_SORT","09_FORWARD_SORT","09_ForwardSort")
  "10_C51_SELECTIVE_BACKDROP" = @("10_C51_SELECTIVE_BACKDROP","10_C51_SELECTIVE_BACKWARD","11_C51_SELECTIVE_BACKDROP","11_C51_SELECTIVE_BACKWARD")
  "11_C49_ATTRIBUTE_DECOMPOSED" = @("11_C49_ATTRIBUTE","11_C49_ATTRIBUTE_DECOMPOSED","11_C49_ATTRIBUTE_DECOUPLED","12_ATTRIBUTE_DECOMPOSED_C49","12_ATTRIBUTE_DECOUPLED_C49")
  "12_CANDIDATE_C"       = @("12_CANDIDATE_C","13_CANDIDATE_C")
  "13_TRAINABLE_HIGS"    = @("13_HIGS_TRAINABLE","13_TRAINABLE_HIGS","14_HIGS_TRAINABLE","14_TRAINABLE_HIGS")
  "05_PROTOCOL_AND_DATASETS" = @("05_PROTOCOL_AND_DATASETS","05_PROTOCOL","05_ExperimentalProtocol")
  "01_PROJECT_OVERVIEW"  = @("01_PROJECT_OVERVIEW")
  "02_TIMELINE"          = @("02_TIMELINE","02_RESEARCH_TIMELINE")
  "03_FINAL_MATRIX"      = @("03_FINAL_MATRIX","03_FINAL_RESULT_MATRIX")
}

foreach ($canonical in $merge.Keys) {
  New-Item -ItemType Directory -Force -Path (Join-Path $root $canonical) | Out-Null
  $seen = @{}
  foreach ($src in ($merge[$canonical] | Select-Object -Unique)) {
    $srcDir = Join-Path $root $src
    if ((Test-Path $srcDir) -and (Test-Path (Join-Path $srcDir "..\dummy")) -eq $false) {
      # merge files (flat files directly under srcDir)
      Get-ChildItem $srcDir -File -ErrorAction SilentlyContinue | ForEach-Object {
        $dest = Join-Path $root (Join-Path $canonical $_.Name)
        if (-not (Test-Path $dest)) {
          Copy-Item $_.FullName $dest -Force
        } else {
          # if identical, skip; else rename with suffix
          if ((Get-FileHash-Id $_.FullName) -ne (Get-FileHash-Id $dest)) {
            Copy-Item $_.FullName ("{0}.dup" -f $dest) -Force
          }
        }
      }
    }
  }
}

# --- Delete ALL non-canonical top-level directories (after merge; canonical kept) ---
$canonicalSet = @($merge.Keys) + @("_tools","17_APPENDICES")
Get-ChildItem $root -Directory | Where-Object {
  $_.Name -notin $canonicalSet
} | ForEach-Object {
  Write-Host ("REMOVE: " + $_.Name)
  Remove-Item $_.FullName -Recurse -Force
}

# --- Remove empty dirs everywhere ---
Get-ChildItem $root -Directory -Recurse | Where-Object { (Get-ChildItem $_.FullName -Force | Measure-Object).Count -eq 0 } | ForEach-Object {
  Write-Host ("EMPTY: " + $_.FullName.Replace($root,''))
  Remove-Item $_.FullName -Force
}

Write-Host "Normalization done."
