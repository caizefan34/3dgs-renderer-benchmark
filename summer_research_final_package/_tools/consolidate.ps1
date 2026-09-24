# Consolidate the package into the canonical directory structure.
$root = "C:\Users\36570\3dgs-renderer-benchmark\summer_research_final_package"

# canonical name -> list of old/current dirs whose files should land there
$map = [ordered]@{
  "00_README_FIRST"           = @("00_README_FIRST")
  "01_PROJECT_OVERVIEW"       = @("01_PROJECT_OVERVIEW")
  "02_TIMELINE"               = @("02_TIMELINE","02_RESEARCH_TIMELINE")
  "03_FINAL_MATRIX"           = @("03_FINAL_MATRIX","03_FINAL_RESULT_MATRIX")
  "04_CRITICAL_FINDINGS"      = @("04_CRITICAL_FINDINGS","04_CriticalFindings","04_C_RITICAL_FINDINGS")
  "05_PROTOCOL_AND_DATASETS"  = @("05_PROTOCOL_AND_DATASETS","05_PROTOCOL","05_ExperimentalProtocol")
  "06_C25_BACKDROP_DEPTH"     = @("06_C25_BACKDROP_DEPTH","06_BACKDROP_C25","06_BACKWARD_C25","06_BACKDROP_PROFILING_C25","06_BACKWARD_PROFILING_C25","10_BACKDROP_PROFILING_C25","10_BACKWARD_PROFILING_C25")
  "07_TILE_GEOMETRY_HARDWARE" = @("07_TILE_GEOMETRY_HARDWARE","07_TILE_GEOMETRY","07_TileGeometry_Hardware")
  "08_C42"                    = @("08_C42")
  "09_C1_FORWARD_SORT"        = @("09_C1_FORWARD_SORT","09_FORWARD_SORT")
  "10_C51_SELECTIVE_BACKWARD" = @("10_C51_SELECTIVE_BACKWARD","11_C51_SELECTIVE_BACKWARD","11_C51_SelectiveBackward")
  "11_C49_ATTRIBUTE"          = @("11_C49_ATTRIBUTE","12_ATTRIBUTE_DECOMPOSED_C49","12_ATTRIBUTE_DECOUPLED_C49","12_AttributeDecomposed_C49")
  "12_CANDIDATE_C"            = @("12_CANDIDATE_C","13_CANDIDATE_C")
  "13_HIGS_TRAINABLE"         = @("13_HIGS_TRAINABLE","14_HIGS_TRAINABLE","14_TRAINABLE_HIGS","14_TrainableHiGS")
  "14_NEGATIVE_RESULTS"       = @("14_NEGATIVE_RESULTS","15_NEGATIVE_RESULTS","15_NegativeResults")
  "15_REPRODUCIBILITY"        = @("15_REPRODUCIBILITY","16_REPRODUCIBILITY","16_Reproducibility")
  "16_RAW_DATA"               = @("16_RAW_DATA","17_RAW_DATA","17_RawData")
  "17_APPENDICES"             = @("17_APPENDICES")
}

foreach ($c in $map.Keys) {
  $target = Join-Path $root $c
  New-Item -ItemType Directory -Force -Path $target | Out-Null
  foreach ($src in ($map[$c] | Select-Object -Unique)) {
    $srcDir = Join-Path $root $src
    if ((Test-Path $srcDir) -and $src -ne $c) {
      Get-ChildItem -Path $srcDir -Force | ForEach-Object {
        $dst = Join-Path $target $_.Name
        if ($_.PSIsContainer) {
          if (-not (Test-Path $dst)) { Copy-Item -Path $_.FullName -Destination $dst -Recurse -Force }
        } else {
          Copy-Item -Path $_.FullName -Destination $dst -Force
        }
      }
    }
  }
}

# Remove old empty dirs and the temp scaffolding dirs
$trash = @("_drafts","_tools","02_RESEARCH_TIMELINE","02_Timeline","03_FinalMatrix","03_FINAL_RESULT_MATRIX","04_CriticalFindings","04_C_RITICAL_FINDINGS","05_ExperimentalProtocol","05_Protocol","06_BACKDROP_PROFILING_C25","06_BACKWARD_C25","06_BACKWARD_PROFILING_C25","07_TileGeometry_Hardware","10_BACKDROP_PROFILING_C25","10_BACKWARD_PROFILING_C25","11_C51_SelectiveBackward","12_ATTRIBUTE_DECOUPLED_C49","12_AttributeDecomposed_C49","13_CANDIDATE_C","14_HIGS_TRAINABLE","14_TrainableHIGS","14_HIGS","15_NegativeResults","16_REPRODUCABILITY","17_RawData","8_Backdrop_C25")
foreach ($t in $trash) {
  $p = Join-Path $root $t
  if ((Test-Path $p) -and ((Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)) {
    Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
  }
}

Write-Host "Consolidation done."
