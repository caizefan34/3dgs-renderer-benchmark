function Invoke-Remote([string]$script) {
  $lfs = ($script -replace "`r","")
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($lfs))
  ssh EPIC-05 "echo $b64 | base64 -d | bash"
}
function Send-RemoteFile([string]$localPath, [string]$remotePath) {
  $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($localPath))
  ssh EPIC-05 "echo $b64 | base64 -d > $remotePath"
}
