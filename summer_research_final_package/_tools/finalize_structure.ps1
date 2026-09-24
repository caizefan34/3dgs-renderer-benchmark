$ErrorActionPreference = 'Stop'
$root = "C:\Users\36570\3dgs-renderer-benchmark"
$pkg = Join-Path $root "summer_research_final_package"

function Move-Content([string]$src, [string]$dst) {
  if (-not (Test-Path $src)) { return }
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Get-ChildItem $src -Force | ForEach-Object {
    $target = Join-Path $dst $_.Name
    if ($_.PSIsContainer) {
      if (-not (Test-Path $target)) { Move-Item $_.FullName $target -Force }
      else {
        # merge dir
        Get-ChildItem $_.FullName -Force | ForEach-Object {
          $t2 = Join-Path $target $_.Name
          if (-not (Test-Path $t2)) { Move-Item $_.FullName $t2 -Force }
          else { Move-Item $_.FullName "$t2.conflict" -Force }
        }
        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
      }
    } else {
      if (-not (Test-Path $target)) { Move-Item $_.FullName $target -Force }
      else { Move-Item $_.FullName "$target.conflict" -Force }
    }
  }
}

# 1) Rename/merge existing dirs to canonical names
Move-Content (Join-Path $pkg "01_Project_Overview")      (Join-Path $pkg "01_PROJECT_OVERVIEW")
Move-Content (Join-Path $pkg "03_FinAlmost_MATRIX")      (Join-Path $pkg "03_RESULTS_MATRIX")
Move-Content (Join-Path $pkg "05_Protocol_And_DataSets") (Join-Path $pkg "05_PROTOCOL_AND_DATASETS")
Move-Content (Join-Path $pkg "06_Backdrop_C25")          (Join-Path $pkg "10_BACKDROP_PROFILING_C25")
Move-Content (Join-Path $pkg "07_THE_GEOMETRY_HARDWARE") (Join-Path $pkg "07_TILE_GEOMETRY_HARDWARE")
Move-Content (Join-Path $pkg "09_C1_Forward_Sort")       (Join-Path $pkg "09_FORWARD_SORT")
Move-Content (Join-Path $pkg "10_C51_Selective_Backdrop")(Join-Path $pkg "11_C51_SELECTIVE_BACKDROP")
Move-Content (Join-Path $pkg "11_C49_Attribute_Decomposed")(Join-Path $pkg "12_C49_ATTRIBUTE_DECOMPOSED")
Move-Content (Join-Path $pkg "12_Candidate_C")           (Join-Path $pkg "13_CANDIDATE_C")
Move-Content (Join-Path $pkg "13_Trainable_HIGS")        (Join-Path $pkg "14_TRAINABLE_HIGS")

# 2) Copy the root docs into their sections
Copy-Item (Join-Path $pkg "RESEARCH_TIMELINE.md") (Join-Path $pkg "02_TIMELINE\RESEARCH_TIMELINE.md") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $pkg "FINAL_RESULT_MATRIX.csv") (Join-Path $pkg "03_RESULTS_MATRIX\FINAL_RESULT_MATRIX.csv") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $pkg "00_README_FIRST.md") (Join-Path $pkg "00_README_FIRST\README_FIRST.md") -Force -ErrorAction SilentlyContinue

# 3) Restore repo mirrors (text files only) into the section folders
$repo = @{
  "02_REPORTS_COLLECTION" = "reports"
}
foreach ($k in $repo.Keys) {
  $src = Join-Path $root $repo[$k]
  $dst = Join-Path $pkg $k
  if (Test-Path $src) {
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem $src -Recurse -File -Include *.md,*.json,*.csv,*.yaml,*.yml,*.toml,*.txt -EA SilentlyContinue | ForEach-Object {
      $rel = $_.FullName.Substring($src.Length).TrimStart('\')
      $target = Join-Path $dst $rel
      New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
      if (-not (Test-Path $target)) { Copy-Item $_.FullName $target -Force }
    }
  }
}

# 4) Copy raw-data samples into 17_RAW_DATA (only the key summary files, not huge artifacts)
$rawSrc = Join-Path $root "results"
if (Test-Path $rawSrc) {
  $rawDst = Join-Path $pkg "17_RAW_DATA"
  New-Item -ItemType Directory -Force -Path $rawDst | Out-Null
  Get-ChildItem $rawSrc -Recurse -File -EA SilentlyContinue | Where-Object {
    $_.Extension -in '.json','.csv','.md','.yaml','.yml','.txt' -and $_.Length -lt 5MB
  } | ForEach-Object {
    $rel = $_.FullName.Substring($rawSrc.Length).TrimStart('\')
    $target = Join-Path $rawDst $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    if (-not (Test-Path $target)) { Copy-Item $_.FullName $target -Force }
  }
}

Write-Host "Structure finalization complete."
