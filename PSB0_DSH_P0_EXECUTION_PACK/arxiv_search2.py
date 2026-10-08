import urllib.request, urllib.parse, json, time, ssl, re
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def q(query, n=6):
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":query,"max_results":n,"sortBy":"submittedDate","sortOrder":"descending"})
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    try:
        data=urllib.request.urlopen(req, timeout=20, context=ctx).read().decode()
    except Exception as e:
        return [("ERR",str(e),"")]
    titles=re.findall(r"<entry>.*?<title>(.*?)</title>", data, re.S)
    ids=re.findall(r"<entry>.*?<id>(.*?)</id>", data, re.S)
    summ=re.findall(r"<entry>.*?<summary>(.*?)</summary>", data, re.S)
    pub=re.findall(r"<entry>.*?<published>(.*?)</published>", data, re.S)
    res=[]
    for i in range(min(len(titles),len(ids))):
        s=summ[i].strip().replace("\n"," ")[:220] if i<len(summ) else ""
        res.append((titles[i].strip().replace("\n"," "), ids[i].strip(), pub[i][:10] if i<len(pub) else "", s))
    return res
queries={
 "taming_3dgs": 'abs:"Taming 3DGS" OR ti:"Taming 3DGS"',
 "fasterGS": 'ti:"FasterGS"',
 "associative_bucket": 'abs:"associative" AND abs:"bucket" AND abs:"3DGS"',
 "parallel_prefix_compositing": 'abs:"parallel prefix" AND abs:"compositing"',
 "scan_state_recovery_render": 'abs:"scan" AND abs:"state recovery" AND abs:"rendering"',
 "differentiable_render_backward_scan": 'abs:"backward" AND abs:"scan" AND abs:"alpha compositing"',
 "workload_balanced_splatting": 'abs:"workload-balanced" AND abs:"splatting"',
}
out={}
for k,query in queries.items():
    out[k]=q(query)
    print("=====",k)
    for t,i,p,s in out[k]:
        print(f"  [{p}] {t[:85]}")
        print(f"        {i}  :: {s[:160]}")
    time.sleep(3)
json.dump(out, open("/tmp/arxiv_prior_art2.json","w"), indent=1)
print("SAVED")
