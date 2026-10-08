function Invoke-Mx([string]$Local,[int]$Retries=10){
  $key="C:\Users\36570\.ssh\mx_public.key"
  for($i=1;$i -le $Retries;$i++){
    $out = Get-Content -Raw $Local | ssh -i $key -p 26372 -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=15 liaoyuanjun@36.140.146.31 "tr -d '\r' | bash -s" 2>&1
    if($LASTEXITCODE -eq 0 -and ($out -notmatch 'Connection closed')){ return $out }
    Start-Sleep -Seconds (5 + $i*2)
  }
  Write-Output "MX-FAILED: $out"; return $out
}