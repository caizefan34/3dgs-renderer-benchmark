import urllib.request, urllib.parse, re, time, json
QUERIES = [
 "Gaussian Splatting optimizer memory",
 "8-bit Adam optimizer quantization",
 "block-wise quantization adaptive optimizer",
 "Gaussian Splatting mixed precision training",
 "AdamW Gaussian Splatting",
 "sparse Adam Gaussian Splatting speedup",
 "compressed optimizer state deep learning",
 "attribute-aware neural network compression",
]
def search(q, n=5):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": "all:" + q, "start": 0, "max_results": n})
    try:
        data = urllib.request.urlopen(url, timeout=8).read().decode("utf-8","ignore")
    except Exception as e:
        return [("ERR", str(e)[:60], "")]
    entries = re.findall(r"<entry>(.*?)</entry>", data, re.S)
    out=[]
    for e in entries:
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        p = re.search(r"<published>(.*?)</published>", e, re.S)
        i = re.search(r"<id>(.*?)</id>", e, re.S)
        title = re.sub(r"\s+"," ",t.group(1)).strip() if t else "?"
        pub = p.group(1)[:10] if p else "?"
        iid = i.group(1) if i else "?"
        out.append([title, pub, iid])
    return out
res={}
for q in QUERIES:
    r=search(q)
    res[q]=r
    print("Q:", q, "->", len(r), flush=True)
    for title,pub,iid in r:
        print(f"   [{pub}] {title[:95]}", flush=True)
    time.sleep(0.5)
json.dump(res, open("amps_gs/AMPS0/raw/arxiv_prior_art.json","w",encoding="utf-8"), indent=1, ensure_ascii=False)
print("SAVED", flush=True)
