$ErrorActionPreference = "SilentlyContinue"
$arxivQueries = @(
    '"logical optimizer state"',
    '"lazy adam"',
    'all:"lazy adam" AND all:"gaussian splatting"',
    'all:"gaussian splatting" AND all:"bias correction" AND all:sparse',
    'all:adam AND all:"deferred update"',
    'all:adam AND all:materialize AND all:deferred',
    'all:"per-parameter" AND all:timestamp AND all:optimizer',
    'all:"optimizer state" AND all:"gaussian splatting" AND all:visible',
    'all:"gaussian splatting" AND all:"optimizer" AND all:"skip"',
    'all:"visibility" AND all:"adam" AND all:"gaussian splatting"',
    'all:"selective adam"',
    'all:"sparse adam" AND all:"gaussian splatting"',
    'all:"optimizer" AND all:"replay" AND all:"skip" AND all:"adam"'
)
foreach ($q in $arxivQueries) {
    $enc = [System.Uri]::EscapeDataString($q)
    $xml = curl.exe -s --max-time 25 "https://export.arxiv.org/api/query?search_query=$enc&max_results=6"
    $total = "?"
    $m = [regex]::Match(($xml -join "`n"), '<opensearch:totalResults>(\d+)</opensearch:totalResults>')
    if ($m.Success) { $total = $m.Groups[1].Value }
    $titles = [regex]::Matches(($xml -join "`n"), '<title>(.*?)</title>') |
        ForEach-Object { ($_.Groups[1].Value -replace '\s+', ' ').Trim() } |
        Where-Object { $_ -notlike "arXiv Query*" }
    Write-Output ("[{0,5}] {1}" -f $total, $q)
    foreach ($t in ($titles | Select-Object -First 6)) { Write-Output ("        - " + $t) }
    Start-Sleep -Seconds 3
}
