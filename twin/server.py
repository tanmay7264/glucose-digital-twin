"""Stdlib HTTP server: dashboard + JSON + live what-if re-simulation. No framework needed."""
import json, pickle, os
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from . import model as M
from .data import MEALS, meal_entry, fhir_bundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = pickle.load(open(os.path.join(ROOT, "out/state.pkl"), "rb"))
RNG = np.random.default_rng(1)


def plan_of(d):
    if d.get("meal") and d["meal"] in MEALS:
        c, sl = meal_entry(d["meal"])
        c = float(d.get("carbs") or c)
    else:
        c, sl = 0.0, 0.3
    return dict(c=c, sl=sl, off=int(d.get("meal_offset", 0)) // 5, walk=int(d.get("walk_min", 0)) // 5, woff=int(d.get("walk_offset", 30)) // 5)


def whatif(pid, i, plans, S=60, n=36):
    st = STATE[pid]
    v = max(j for j, vv in enumerate(st["versions"]) if vv[0] <= i)
    th, cov = st["versions"][v][1], st["versions"][v][2]
    smp = np.vstack([np.log(th), RNG.multivariate_normal(np.log(th), cov, S - 1)])
    smp = np.exp(smp)
    thv = (smp[:, 0], smp[:, 1], smp[:, 2])
    out = []
    for pl in plans:
        pl = plan_of(pl)
        s = np.tile(st["states"][i], (S, 1))
        size = np.exp(RNG.normal(0, 0.2, S))  # portion-size uncertainty
        g = [s[:, 0].copy()]
        for j in range(n):
            if j == pl["off"] and pl["c"] > 0:
                s[:, 2] += pl["c"] * (1 - pl["sl"]) * size
                s[:, 4] += pl["c"] * pl["sl"] * size
            e = 1.0 if pl["woff"] <= j < pl["woff"] + pl["walk"] else 0.0
            s = step_(s, thv, st["gb"], e)
            g.append(s[:, 0].copy())
        g = np.array(g)
        q = np.percentile(g, [5, 50, 95], axis=1)
        peaks = g.max(0)
        out.append(dict(p5=np.round(q[0], 0).tolist(), p50=np.round(q[1], 0).tolist(), p95=np.round(q[2], 0).tolist(),
                        peak=float(np.median(peaks)), peak_lo=float(np.percentile(peaks, 5)), peak_hi=float(np.percentile(peaks, 95)),
                        min_above=int((q[1] > M.THR).sum() * 5), p_exceed=float((peaks > M.THR).mean())))
    return out


step_ = M.step


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.path = self.path.split("?")[0]
        if self.path in ("/", "/index.html"):
            self._send(open(os.path.join(ROOT, "web/index.html"), "rb").read(), "text/html")
        elif self.path.startswith(("/vendor/", "/human.js")) and ".." not in self.path:
            f = os.path.join(ROOT, "web", self.path.lstrip("/"))
            if not os.path.isfile(f):
                return self.send_error(404)
            self._send(open(f, "rb").read(), "text/javascript")
        elif self.path == "/api/results":
            self._send(open(os.path.join(ROOT, "out/results.json"), "rb").read())
        elif self.path.startswith("/api/fhir/") and self.path[10:] in STATE:
            p = STATE[self.path[10:]]["p"]
            self._send(json.dumps(fhir_bundle(p), indent=1))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/whatif":
            return self.send_error(404)
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            self._send(json.dumps(whatif(d["pid"], int(d["i"]), d["plans"])))
        except Exception as e:  # bad pid/index from client
            self.send_error(400, str(e))

    def log_message(self, *a):
        pass


def serve(port=8765):
    port = int(os.environ.get("PORT", port))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"   # hosted platforms set PORT
    print(f"Digital twin dashboard: http://localhost:{port}")
    ThreadingHTTPServer((host, port), H).serve_forever()
