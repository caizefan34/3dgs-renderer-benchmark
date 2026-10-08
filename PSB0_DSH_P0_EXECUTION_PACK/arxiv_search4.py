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
        s=summ[i].strip().replace("\n"," ")[:260] if i<len(summ) else ""
        res.append((titles[i].strip().replace("\n"," "), ids[i].strip(), pub[i][:10] if i<len(pub) else "", s))
    return res
queries={
 "exact_collision_scan_backward": 'abs:"scan" AND abs:"backward" AND abs:"compositing" AND abs:"Gaussian"',
 "associative_transform_splatting": 'abs:"associative" AND abs:"splatting"',
 "workload_balanced_fasterGS_2026": 'abs:"workload" AND abs:"FasterGS"',
 "local_compositing_load_balance": 'abs:"local compositing" OR abs:"associative" AND abs:"load balancing" AND abs:"Gaussian"',
 "hierarchical_backward_gs": 'abs:"hierarchical" AND abs:"backward" AND abs:"Gaussian Splatting"',
}
out={}
for k,query in queries.items():
    out[k]=q(query)
    print("=====",k)
    for t,i,p,s in out[k]:
        print(f"  [{p}] {t[:88]}")
        print(f"        {i} :: {s[:170]}")
    time.sleep(3)
json.dump(out, open("/tmp/arxiv_prior_art4.json","w"), indent=1)
print("SAVED")
