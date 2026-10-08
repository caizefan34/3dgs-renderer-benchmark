$arxivQueries = @(
    '"virtual optimizer state"',
    '"logical optimizer state"',
    '"lazy adam"',
    'all:"lazy adam" AND all:"gaussian splatting"',
    'all:"gaussian splatting" AND all:optimizer AND all:invisible',
    'all:"gaussian splatting" AND all:replay AND all:optimizer',
    'all:"gaussian splatting" AND all:"bias correction" AND all:sparse',
    'all:adam AND all:"deferred update"',
    'all:adam AND all:materialize AND all:deferred',
    'all:"per-parameter" AND all:timestamp AND all:optimizer',
    'all:"amortized optimizer"',
    'all:"optimizer state" AND all:"gaussian splatting" AND all:visible'
)
foreach ($q in $arxivQueries) {
    $enc = [System.Uri]::EscapeDataString($q)
    $xml = curl.exe -s --max-time 25 "https://export.arxiv.org/api/query?search_query=$enc&max_results=5"
    $total = 0
    if ($xml -match '<opensearch:totalResults>(\d+)</opensearch:totalResults>') { $total = $Matches[1] }
    $titles = @()
    foreach ($line in ($xml -split "`n")) {
        if ($line -match '<title>(.*?)</title>' -and $line -notmatch 'arXiv Query') {
            $titles += ($Matches[1] -replace '\s+', ' ').Trim()
        }
    }
    Write-Output ("[{0,5} results] {1}" -f $total, $q)
    foreach ($t in ($titles | Select-Object -First 5)) { Write-Output ("          - " + $t) }
    Start-Sleep -Seconds 3
}
