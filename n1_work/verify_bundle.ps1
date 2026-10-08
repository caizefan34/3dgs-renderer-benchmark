$ErrorActionPreference = "Stop"
$root = "C:\Users\36570\3dgs-renderer-benchmark\N1_RETURN"
$manifest = Get-Content "$root\artifacts\MANIFEST_N1.json" -Raw | ConvertFrom-Json
$ok = 0; $bad = @(); $skip = 0
foreach ($group in @($manifest.required_artifacts, $manifest.supporting_evidence)) {
  foreach ($e in $group) {
    if ($e.sha256 -eq "self-referential") { $skip++; continue }
    $p = Join-Path "$root\artifacts" $e.file
    if (-not (Test-Path $p)) { $bad += "$($e.file): MISSING LOCALLY"; continue }
    $h = (Get-FileHash $p -Algorithm SHA256).Hash.ToLower()
    if ($h -eq $e.sha256) { $ok++ } else { $bad += "$($e.file): HASH MISMATCH" }
  }
}
Write-Output "verified_ok=$ok skipped_self=$skip failures=$($bad.Count)"
if ($bad.Count -gt 0) { $bad | ForEach-Object { Write-Output "  FAIL $_" } }
# required artifacts present?
$req = @("N1_SOURCE_IDENTITY.json","N1_CHECKPOINT_IDENTITY.json","N1_PROBE_LIST.json","N1_ACTION_RESULTS.parquet","N1_ACTION_TIMING.json","N1_GRADIENT_CONFLICT.parquet","N1_SKETCH_PREDICTIVITY.json","N1_ACTION_HETEROGENEITY.json","N1_BUDGET_FRONTIER.json","N1_CROSS_SCENE_POLICY.json","N1_CONTAMINATION.json","N1_VERDICT.json","N1_REPORT.md","MANIFEST_N1.json")
$missing = $req | Where-Object { -not (Test-Path "$root\artifacts\$_") }
Write-Output "required_artifacts_missing=$($missing.Count) $($missing -join ', ')"
$v = Get-Content "$root\artifacts\N1_VERDICT.json" -Raw | ConvertFrom-Json
Write-Output "VERDICT: $($v.verdict)"
