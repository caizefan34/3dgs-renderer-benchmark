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

queries = [
    'all:"deferred" AND all:"optimizer" AND all:"Gaussian"',
    'all:"lazy" AND all:"optimizer" AND all:"Gaussian splatting"',
    'all:"virtual" AND all:"optimizer state"',
    'all:"per-Gaussian" AND all:"step" AND all:"Adam"',
    'all:"dynamic" AND all:"Gaussian splatting" AND all:"densification" AND all:"consistent"',
    'all:"bitwise" OR all:"bit-exact" AND all:"Gaussian splatting"',
    'all:"state consistency" AND all:"neural"',
    'all:"memory efficient" AND all:"Adam" AND all:"sparse" AND all:"3D"',
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
