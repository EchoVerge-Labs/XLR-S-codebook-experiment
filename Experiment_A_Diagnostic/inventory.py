import os, collections, csv
BASE = os.path.expanduser("~/google-drive/Pre Processed Data")
rows_out=[]
for lang, f in [("Sinhala","results/_inventory_sinhala.tsv"),("Tamil","results/_inventory_tamil.tsv")]:
    per = collections.defaultdict(lambda: [0,0.0])   # genre -> [nfiles, seconds]
    for line in open(f):
        sz, p = line.rstrip("\n").split("\t", 1)
        rel = os.path.relpath(p, os.path.join(BASE, lang))
        genre = rel.split(os.sep)[0]
        d = (int(sz) - 44) / 32000.0
        per[genre][0] += 1
        per[genre][1] += d
    tot_n = sum(v[0] for v in per.values()); tot_s = sum(v[1] for v in per.values())
    print(f"\n{'='*74}\n  {lang}\n{'='*74}")
    print(f"{'genre':<24}{'files':>9}{'hours':>10}{'% files':>9}{'mean dur(s)':>13}")
    print("-"*74)
    for g in sorted(per, key=lambda g: -per[g][0]):
        n,s = per[g]
        print(f"{g:<24}{n:>9,}{s/3600:>10.2f}{100*n/tot_n:>8.1f}%{s/n:>13.2f}")
        rows_out.append(dict(language=lang, genre=g, files=n, hours=round(s/3600,3), mean_dur_s=round(s/n,2)))
    print("-"*74)
    print(f"{'TOTAL':<24}{tot_n:>9,}{tot_s/3600:>10.2f}{100.0:>8.1f}%{tot_s/tot_n:>13.2f}")
    rows_out.append(dict(language=lang, genre="TOTAL", files=tot_n, hours=round(tot_s/3600,3), mean_dur_s=round(tot_s/tot_n,2)))
with open("results/data_inventory.csv","w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=["language","genre","files","hours","mean_dur_s"]); w.writeheader(); w.writerows(rows_out)
print("\n-> results/data_inventory.csv")
