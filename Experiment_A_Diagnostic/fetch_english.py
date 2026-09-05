import io, os, sys, tarfile, urllib.request
URL="https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main/audio/en/dev/en_dev_0.tar"
OUT=os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic/data/english_cv")
TARGET=1200
req=urllib.request.Request(URL, headers={"User-Agent":"curl/8"})
r=urllib.request.urlopen(req, timeout=120)
print("HTTP", r.status, "len", r.headers.get("Content-Length"))
n=0
tf=tarfile.open(fileobj=r, mode="r|")   # streaming, sequential
for m in tf:
    if not m.isfile(): continue
    name=os.path.basename(m.name)
    if not name.lower().endswith((".mp3",".wav")): continue
    data=tf.extractfile(m).read()
    open(os.path.join(OUT,name),"wb").write(data)
    n+=1
    if n>=TARGET: break
print("extracted",n,"clips ->",OUT)
