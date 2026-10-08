import urllib.request, urllib.parse, json, time, ssl, re
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def q(query, n=6):
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":query,"max_results":n,"sortBy":"submittedDate","sortOrder":"descending"})
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    try:
        data=urllib.request.urlopen(req, timeout=20, context=ctx).read().decode()
    except Exception as e:
        return [("ERR",str(e),"","")]
    titles=re.findall(r"<entry>.*?<title>(.*?)</title>", data, re.S)
    ids=re.findall(r"<entry>.*?<id>(.*?)</id>", data, re.S)
    summ=re.findall(r"<entry>.*?<summary>(.*?)</summary>", data, re.S)
    pub=re.findall(r"<entry>.*?<published>(.*?)</published>", data, re.S)
    res=[]
    for i in range(min(len(titles),len(ids))):
        s=summ[i].strip().replace("\n"," ")[:240] if i<len(summ) else ""
        res.append((titles[i].strip().replace("\n"," "), ids[i].strip(), pub[i][:10] if i<len(pub) else "", s))
    return res
queries={
 "workload_balanced_3dgs": 'abs:"workload-balanced" AND abs:"Gaussian"',
 "load_balancing_splats": 'ti:"load balancing" AND abs:"Gaussian Splatting"',
 "associative_scan_rendering": 'abs:"associative scan" AND abs:"rendering"',
 "prefix_scan_rendering_backward": 'abs:"prefix scan" AND abs:"rendering" AND abs:"gradient"',
 "checkpoint_replay_splatting": 'abs:"checkpoint" AND abs:"replay" AND abs:"Gaussian Splatting"',
 "bucketed_backward_splatting": 'abs:"bucket" AND abs:"backward" AND abs:"Gaussian Splatting"',
}
out={}
for k,query in queries.items():
    out[k]=q(query)
    print("=====",k)
    for t,i,p,s in out[k]:
        print(f"  [{p}] {t[:85]}")
        print(f"        {i} :: {s[:150]}")
    time.sleep(3)
json.dump(out, open("/tmp/arxiv_prior_art3.json","w"), indent=1)
print("SAVED")
