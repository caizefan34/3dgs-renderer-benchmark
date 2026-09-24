# Long-term SSH retry: every 5 minutes for up to 4 hours
$maxAttempts = 48
$scriptPath = "C:\Users\36570\3dgs-renderer-benchmark\experiments\r3\run_r3_1_autonomous.sh"
$scriptContent = Get-Content -Raw $scriptPath
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($scriptContent))

for ($i = 1; $i -le $maxAttempts; $i++) {
    $time = Get-Date -Format 'HH:mm:ss'
    Write-Host "[$time] Attempt $i/$maxAttempts..."
    try {
        $result = ssh -o ConnectTimeout=15 -o ServerAliveInterval=5 mx "echo CONNECTED" 2>&1
        if ($result -match "CONNECTED") {
            Write-Host "SSH CONNECTED! Uploading autonomous pipeline..."
            $scriptBytes = [Convert]::FromBase64String($encoded)
            $scriptText = [Text.Encoding]::UTF8.GetString($scriptBytes)
            $scriptText | ssh -o ConnectTimeout=15 mx "cat > /tmp/run_r3_1_autonomous.sh && sed -i 's/\r$//' /tmp/run_r3_1_autonomous.sh && chmod +x /tmp/run_r3_1_autonomous.sh && nohup bash /tmp/run_r3_1_autonomous.sh > /mnt/storage_pool/liaoyuanjun/r3_1_pipeline_nohup.log 2>&1 & echo LAUNCHED"
            Write-Host "Pipeline launched on remote!"
            # Verify it's running
            Start-Sleep -Seconds 5
            $verify = ssh -o ConnectTimeout=15 mx "ps aux | grep run_r3_1_autonomous | grep -v grep | wc -l" 2>&1
            Write-Host "Pipeline processes running: $verify"
            break
        }
    } catch {}
    Write-Host "  Failed, waiting 300s..."
    Start-Sleep -Seconds 300
}
Write-Host "Retry loop finished."
