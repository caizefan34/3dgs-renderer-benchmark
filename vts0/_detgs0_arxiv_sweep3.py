import urllib.request, urllib.parse, re, time

def sweep(q, n=8):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": q, "start": 0, "max_results": n})
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(3)
    return ""

# Final coverage sweeps for the exact-combination claim
queries = [
    'all:"exact" AND all:"training" AND all:"Gaussian splatting" AND all:"reproducib*"',
    'all:"densification" AND all:"deterministic" AND all:"Gaussian"',
    'all:"Gaussian splatting" AND all:"bit-exact"',
    'all:"consistent" AND all:"optimizer" AND all:"densification" AND all:"3D"',
    'all:" Sparse Adam" ',
    'all:"A Step to Decouple Optimization"',
]
for q in queries:
    print("\n##### QUERY:", q)
    out = sweep(q)
    titles = re.findall(r"<entry>.*?<title>(.*?)</title>", out, re.S)
    total = re.search(r"opensearch:totalResults[^>]*>(\d+)<", out)
    print("totalResults:", total.group(1) if total else "?")
    for t in titles[:8]:
        print("  -", " ".join(t.split())[:130])
    time.sleep(3)
