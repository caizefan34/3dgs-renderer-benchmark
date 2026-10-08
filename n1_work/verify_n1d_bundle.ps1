$ErrorActionPreference = "Stop"
$root = "C:\Users\36570\3dgs-renderer-benchmark\N1D_RETURN"
$manifest = Get-Content "$root\artifacts\MANIFEST_N1D.json" -Raw | ConvertFrom-Json
$ok = 0; $bad = @(); $skip = 0
foreach ($group in @($manifest.required_artifacts, $manifest.supporting_evidence, $manifest.contamination_csvs, $manifest.scripts)) {
  foreach ($e in $group) {
    if ($e.sha256 -eq "self-referential") { $skip++; continue }
    if ($e.PSObject.Properties.Name -contains "missing") { $bad += "$($e.file): MARKED MISSING IN MANIFEST"; continue }
    # artifacts live under artifacts\, csvs+scripts under root
    $p = Join-Path "$root\artifacts" $e.file
    if (-not (Test-Path $p)) { $p = Join-Path $root"\" $e.file }
    if (-not (Test-Path $p)) { $bad += "$($e.file): MISSING LOCALLY"; continue }
    $h = (Get-FileHash $p -Algorithm SHA256).Hash.ToLower()
    if ($h -eq $e.sha256) { $ok++ } else { $bad += "$($e.file): HASH MISMATCH" }
  }
}
Write-Output "verified_ok=$ok skipped_self=$skip failures=$($bad.Count)"
if ($bad.Count -gt 0) { $bad | ForEach-Object { Write-Output "  FAIL $_" } }
$req = @("N1D_SOURCE_IDENTITY.json","N1D_PARTNER_LIST.json","N1D_ACTION_RESULTS.parquet","N1D_ACTION_TIMING.json","N1D_CONTAMINATION.json","MANIFEST_N1D.json")
$missing = $req | Where-Object { -not (Test-Path "$root\artifacts\$_") }
Write-Output "required_artifacts_missing=$($missing.Count) $($missing -join ', ')"
