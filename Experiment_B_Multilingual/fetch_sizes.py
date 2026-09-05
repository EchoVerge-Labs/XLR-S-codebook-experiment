import json, urllib.request, concurrent.futures as cf, os
R="results"
rows=json.load(open(f"{R}/fleurs_languages.json"))
def get(r):
    u=f"https://datasets-server.huggingface.co/size?dataset=google%2Ffleurs&config={r['config']}"
    for _ in range(3):
        try:
            d=json.load(urllib.request.urlopen(u,timeout=90))
            for s in d["size"]["splits"]:
                if s["split"]=="test":
                    r["test_rows"]=s["num_rows"]
                    r["test_mb"]=round(s["num_bytes_parquet_files"]/1e6,1)
                    return r
        except Exception as e:
            r["err"]=repr(e)[:60]
    return r
with cf.ThreadPoolExecutor(16) as ex: rows=list(ex.map(get,rows))
json.dump(rows,open(f"{R}/fleurs_languages.json","w"),indent=1,ensure_ascii=False)
ok=[r for r in rows if r.get("test_rows")]
print(f"got sizes for {len(ok)}/{len(rows)}")
print("test_rows: min %d  median %d  max %d"%(min(r['test_rows'] for r in ok),
      sorted(r['test_rows'] for r in ok)[len(ok)//2], max(r['test_rows'] for r in ok)))
print(f">=500 clips: {sum(r['test_rows']>=500 for r in ok)}")
print(f"total test MB if all: {sum(r['test_mb'] for r in ok):,.0f}")
