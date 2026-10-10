"""Download ShanghaiT2DM (figshare 20444397, CC BY 4.0, 3.4 MB) and unpack the T2DM visits to data/shanghai/t2dm/.
The figshare redirect URL expires in 10 s, so fetch and download immediately; retry on failure."""
import os, ssl, subprocess, zipfile
D = "data/shanghai"; os.makedirs(D, exist_ok=True)
for k in range(5):
    loc = subprocess.run(["curl", "-sI", "--cacert", "/etc/ssl/cert.pem", "https://ndownloader.figshare.com/files/38259264"], capture_output=True, text=True).stdout
    url = next((l.split(" ", 1)[1].strip() for l in loc.splitlines() if l.lower().startswith("location")), None)
    if url and subprocess.run(["curl", "-s", "-m", "60", "-o", f"{D}/data.zip", url]).returncode == 0 and zipfile.is_zipfile(f"{D}/data.zip"):
        break
else:
    raise SystemExit("download failed; fetch https://figshare.com/articles/dataset/20444397 manually")
z = zipfile.ZipFile(f"{D}/data.zip"); os.makedirs(f"{D}/t2dm", exist_ok=True)
for i in z.infolist():
    b = os.path.basename(i.filename)
    if "T2DM" in i.filename and b and not b.startswith("~$") and b.endswith((".xls", ".xlsx")):
        open(f"{D}/t2dm/{b}", "wb").write(z.read(i))
print("ok", len(os.listdir(f"{D}/t2dm")), "visit files")
