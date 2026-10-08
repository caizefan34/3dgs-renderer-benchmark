import urllib.request, urllib.parse, json, time, ssl
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def q(query, n=8):
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":query,"max_results":n,"sortBy":"submittedDate","sortOrder":"descending"})
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    try:
        data=urllib.request.urlopen(req, timeout=20, context=ctx).read().decode()
    except Exception as e:
        return [("ERR",str(e))]
    entries=[]
    import re
    titles=re.findall(r"<entry>.*?<title>(.*?)</title>", data, re.S)
    ids=re.findall(r"<entry>.*?<id>(.*?)</id>", data, re.S)
    pub=re.findall(r"<entry>.*?<published>(.*?)</published>", data, re.S)
    for i in range(min(len(titles),len(ids))):
        entries.append((titles[i].strip().replace("\n"," "), ids[i].strip(), pub[i][:10]))
    return entries
queries={
 "associative_scan_backward": 'abs:"associative scan" AND abs:"gaussian splatting"',
 "prefix_suffix_state_recovery": 'abs:"prefix" AND abs:"suffix" AND abs:"splatting" AND abs:"backward"',
 "bucket_checkpoint_replay": 'abs:"bucket" AND abs:"checkpoint" AND abs:"replay" AND abs:"splatting"',
 "compositing_associative": 'abs:"associative" AND abs:"compositing" AND abs:"gaussian"',
 "workload_balanced_fasterGS": 'abs:"workload" AND abs:"FasterGS"',
 "segment_scan_differentiable_render": 'abs:"segment" AND abs:"scan" AND abs:"differentiable rendering"',
}
out={}
for k,query in queries.items():
    out[k]=q(query)
    print("=====",k)
    for t,i,p in out[k]:
        print(f"  [{p}] {t[:90]}  | {i}")
    time.sleep(3)
json.dump(out, open("/tmp/arxiv_prior_art.json","w"), indent=1)
