import urllib.request, urllib.parse, re, time

def sweep(q, n=6):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": q, "start": 0, "max_results": n})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(3)
    return ""

queries = [
    'all:"sparse adam" AND all:"gaussian"',
    'all:"FastGS"',
    'all:"GS-Scale" OR all:"GS Scale"',
    'all:"deterministic" AND all:"Gaussian splatting"',
    'all:"exact" AND all:"optimizer" AND all:"Gaussian splatting"',
    'all:"topology" AND all:"Gaussian splatting" AND all:"densification"',
    'all:"reproducibility" AND all:"Gaussian splatting"',
    'all:"checkpoint" AND all:"Gaussian splatting" AND all:"optimizer state"',
]
for q in queries:
    print("\n##### QUERY:", q)
    out = sweep(q)
    titles = re.findall(r"<entry>.*?<title>(.*?)</title>", out, re.S)
    total = re.search(r"opensearch:totalResults[^>]*>(\d+)<", out)
    print("totalResults:", total.group(1) if total else "?")
    for t in titles[:6]:
        print("  -", " ".join(t.split())[:140])
    time.sleep(3)
