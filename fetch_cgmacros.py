"""Fetch only the CSVs (not the photos) from the CGMacros PhysioNet zip via HTTP range requests. CC BY-NC-SA 4.0."""
import io, os, ssl, sys, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
U = "https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
OUT = "data/cgmacros"
CTX = ssl.create_default_context(cafile="/etc/ssl/cert.pem")  # macOS system CAs; python.org build ships none


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.size = int(urllib.request.urlopen(urllib.request.Request(U, method="HEAD"), context=CTX).headers["Content-Length"]); self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else self.pos + o if w == 1 else self.size + o
        return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b""
        for _ in range(5):
            try:
                r = urllib.request.Request(U, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                b = urllib.request.urlopen(r, timeout=60, context=CTX).read(); break
            except Exception: pass
        self.pos += len(b); return b


def get(name):
    dst = os.path.join(OUT, os.path.basename(name))
    if os.path.exists(dst): return
    z = zipfile.ZipFile(RangeFile())
    open(dst + ".part", "wb").write(z.read(name)); os.rename(dst + ".part", dst); print("got", dst, flush=True)

os.makedirs(OUT, exist_ok=True)
names = [n for n in zipfile.ZipFile(RangeFile()).namelist() if n.lower().endswith(".csv") and "Gut" not in n and "Microbes" not in n]
print(len(names), "csv files", flush=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(get, names))
print("done")
