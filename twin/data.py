"""Data layer: Indian meal library, synthetic T2D generator (stress-test only), CGMacros loader, FHIR export.

The synthetic generator is DELIBERATELY structurally different from the twin's ODE (Erlang-3 gut, saturating
uptake, renal spill, delayed secretion, circadian SI, unlogged snacks, sensor lag) so the twin cannot just
re-learn the generator. Even so: synthetic metrics are stress-test numbers, never headline accuracy.
"""
import csv
import os
import numpy as np

STEP = 5  # minutes per grid step

# name: (carbs g, protein g, fat g, fiber g). Approximate per-serving values; verify against IFCT 2017 before real use.
MEALS = {
    "Idli x3 + sambar": (42, 9, 3, 4), "Masala dosa": (60, 8, 12, 4), "Poha (1 plate)": (50, 6, 8, 3),
    "Upma (1 bowl)": (45, 7, 9, 3), "Aloo paratha x2": (68, 9, 18, 5), "Moong dal chilla x2": (28, 16, 8, 6),
    "Roti x2 + dal + sabzi": (55, 16, 10, 9), "White rice 1 cup + dal + sabzi": (70, 14, 8, 6),
    "Brown rice + dal + sabzi": (60, 15, 8, 9), "Ragi roti x2 + dal": (50, 13, 6, 9),
    "Jowar roti x2 + sabzi": (45, 10, 6, 8), "Chole bhature": (90, 17, 28, 10), "Veg biryani (1 plate)": (80, 12, 18, 5),
    "Curd rice": (45, 8, 6, 1), "Biscuits + chai": (25, 3, 6, 1), "Banana": (27, 1, 0, 3),
    "Sprouts chaat": (20, 10, 2, 7), "Sweets x2 (mithai)": (35, 3, 12, 0),
}
BREAKFAST = ["Idli x3 + sambar", "Masala dosa", "Poha (1 plate)", "Upma (1 bowl)", "Aloo paratha x2", "Moong dal chilla x2"]
MAIN = ["Roti x2 + dal + sabzi", "White rice 1 cup + dal + sabzi", "Brown rice + dal + sabzi", "Ragi roti x2 + dal",
        "Jowar roti x2 + sabzi", "Chole bhature", "Veg biryani (1 plate)", "Curd rice"]
SNACK = ["Biscuits + chai", "Banana", "Sprouts chaat", "Sweets x2 (mithai)"]


def slow_frac(carbs, protein, fat, fiber):
    """Heuristic share of carbs absorbed slowly (fat/protein/fibre slow gastric emptying). Assumption, not fitted."""
    x = fat + protein + 3 * fiber
    return float(np.clip((x / (carbs + x + 1e-9) - 0.15) * 2, 0.05, 0.8))


def meal_entry(name):
    c, p, f, fi = MEALS[name]
    return c, slow_frac(c, p, f, fi)


def _erlang_ra(carbs, lam, lag=5, n=500):
    t = np.arange(n) - lag
    t = np.maximum(t, 0)
    ra = carbs * lam * (lam * t) ** 2 / 2 * np.exp(-lam * t)
    ra[: lag] = 0
    return ra


def simulate(cfg, extra_walks=()):
    """Ground-truth blood glucose, 1-min resolution. cfg from gen_patient. extra_walks: [(start_min, dur_min)]."""
    n, ph = cfg["n"], cfg["phys"]
    ra = np.zeros(n + 600)
    for m in cfg["meals"]:
        lam = ph["kg0"] * (1 - 0.55 * m["sl_true"])
        r = _erlang_ra(m["carbs_true"], lam)
        ra[m["t"]: m["t"] + 500] += r[: len(ra[m["t"]: m["t"] + 500])]
    wk = cfg["wk"].copy()
    for s, d in extra_walks:
        wk[s: s + d] = 1.0
    gb, u0, beta = ph["gb"], 0.006, ph["beta"]
    egp0 = gb / (1 + gb / 800) * u0
    G = Gd = gb
    I = Ie = 0.0
    out = np.empty(n)
    for t in range(n):
        hr = (t / 60) % 24
        si = ph["si"] * (1 - 0.25 * np.exp(-((hr - 7) / 2.5) ** 2))
        if cfg["drift_t"] is not None and t >= cfg["drift_t"]:
            si *= cfg["drift_mult"]
        Gprev = G
        Gd += (G - Gd) / 15
        I += -0.05 * I + beta * max(Gd - 90, 0) * 0.05 + 0.3 * beta * max(G - Gprev, 0)
        Ie += 0.03 * (I - Ie)
        U = G / (1 + G / 800) * (u0 + si * Ie + 0.0045 * wk[t])
        egp = egp0 * max(0.0, 1 - 0.006 * Ie) * (1 + 0.03 * max(gb - G, 0))
        G += ra[t] * 6.0 + egp - U - 0.004 * max(G - 180, 0)
        out[t] = G
    return out


def gen_patient(pid, seed, days=10, drift_day=None):
    r = np.random.default_rng(seed)
    n = days * 1440
    phys = dict(si=r.uniform(0.0002, 0.0006), beta=r.uniform(0.15, 0.45), gb=r.uniform(105, 150),
                kg0=r.uniform(0.035, 0.07))
    meals, wk, steps = [], np.zeros(n), np.zeros(n)
    for d in range(days):
        slots = [(450, 40, BREAKFAST, 1.0), (795, 45, MAIN, 1.0), (1200, 50, MAIN, 1.0), (990, 30, SNACK, 0.5)]
        for mu, sd, pool, pr in slots:
            if r.random() > pr:
                continue
            t = d * 1440 + int(mu + r.normal(0, sd))
            name = pool[r.integers(len(pool))]
            c, sl = meal_entry(name)
            logged = pool is not SNACK or r.random() < 0.5
            meals.append(dict(t=t, name=name, carbs_true=c * r.uniform(0.8, 1.25), carbs_log=c, sl_true=float(np.clip(sl + r.normal(0, .1), 0, .9)),
                              sl_log=sl, logged=bool(logged)))
        base = d * 1440
        steps[base + 420: base + 1320] = r.poisson(6, 900)
        dinner = [m["t"] for m in meals if m["t"] // 1440 == d and m["t"] % 1440 > 1100]
        if dinner and r.random() < 0.35:
            s = dinner[0] + 30
            wk[s: s + 18] = 1.0
        if r.random() < 0.4:
            s = base + int(r.uniform(600, 1000))
            wk[s: s + int(r.uniform(20, 40))] = 1.0
    steps += wk * r.uniform(90, 110, n)
    cfg = dict(n=n, phys=phys, meals=meals, wk=wk, drift_t=None if drift_day is None else drift_day * 1440, drift_mult=0.3)
    G = simulate(cfg)
    ig, k = np.empty(n), G[0]
    for t in range(n):  # interstitial lag
        k += (G[t] - k) / 9
        ig[t] = k
    e, bias = 0.0, 1 + r.normal(0, 0.04)
    cgm = []
    for t in range(0, n, STEP):
        e = 0.7 * e + r.normal(0, 3)
        cgm.append(round(ig[t] * bias + e))
    cgm = np.array(cgm, float)
    for _ in range(days // 2):  # sensor dropouts
        s = r.integers(0, len(cgm) - 20)
        cgm[s: s + r.integers(6, 18)] = np.nan
    for _ in range(days // 3):  # compression lows at night
        s = int(r.integers(0, days)) * 288 + int(r.integers(10, 60))
        cgm[s: s + 4] -= 25
    st5 = steps.reshape(-1, STEP).mean(1)
    age, bmi = r.uniform(35, 65), r.uniform(22, 32)
    return dict(
        id=pid, src="synthetic", cgm=cgm, steps=st5,
        meals=[(m["t"] // STEP, m["carbs_log"], m["sl_log"], m["name"]) for m in meals if m["logged"]],
        static=dict(age=round(age), sex=["F", "M"][r.integers(2)], bmi=round(bmi, 1), waist=round(bmi * 3.4 + r.normal(0, 3)),
                    hba1c=round((np.nanmean(G) + 46.7) / 28.7, 1)),
        gen=cfg)


def load_cgmacros(path, bio=None, sensor="dexcom"):
    """One CGMacros-0XX.csv (1-min rows) -> 5-min grid aligned to midnight of day 1. Dexcom G6 preferred, Libre fallback.
    METs column is METs x10; proxied to steps/min (assumption). Meal = row with Carbs>0 scaled by Amount Consumed."""
    from datetime import datetime
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    f = lambda v: float(v) if v not in (None, "", "NaN") else np.nan
    ts = [datetime.fromisoformat(r["Timestamp"].strip()) for r in rows]
    t0 = ts[0].replace(hour=0, minute=0, second=0)
    mins = np.array([(t - t0).total_seconds() // 60 for t in ts], int)
    n = int(mins[-1] // STEP) + 1
    dx = np.array([f(r.get("Dexcom GL")) for r in rows]); lb = np.array([f(r.get("Libre GL")) for r in rows])
    g = dx if (sensor == "dexcom" and np.isfinite(dx).mean() > 0.5) else lb
    if sensor == "libre15":  # mimic ShanghaiT2DM cadence: Libre every 15 min, linearly interpolated to the 5-min grid, no activity
        g = np.where(mins % 15 == 0, lb, np.nan)
    act = np.array([f(r.get("METs")) for r in rows]) / 10
    cgm, steps = np.full(n, np.nan), np.zeros(n)
    idx = mins // STEP
    for i in np.unique(idx):
        m = idx == i
        if np.isfinite(g[m]).any():
            cgm[i] = np.nanmean(g[m])
    if sensor == "libre15":
        ok = np.isfinite(cgm); gi = np.arange(n); cgm = np.interp(gi, gi[ok], cgm[ok], left=np.nan, right=np.nan)
        j = np.searchsorted(gi[ok], gi); bad = (j > 0) & (j < ok.sum()) & (gi[ok][np.minimum(j, ok.sum() - 1)] - gi[ok][np.maximum(j - 1, 0)] > 9); cgm[bad] = np.nan
        steps[i] = 0 if sensor == "libre15" else np.clip((np.nan_to_num(act[m], nan=1.0).mean() - 1.5) * 40, 0, 100)  # METs -> steps/min proxy
    meals = []
    for k, r in enumerate(rows):
        c = f(r.get("Carbs"))
        if np.isfinite(c) and c > 0:
            c *= np.nan_to_num(f(r.get("Amount Consumed ", r.get("Amount Consumed"))), nan=100) / 100
            meals.append((int(mins[k] // STEP), c, slow_frac(c, np.nan_to_num(f(r.get("Protein"))), np.nan_to_num(f(r.get("Fat"))),
                                                             np.nan_to_num(f(r.get("Fiber")))), r.get("Meal Type") or "meal"))
    sub = int(os.path.basename(path)[9:12])
    st = dict(age=None, sex=None, bmi=None, waist=None, hba1c=None)
    if bio:
        b = next((r for r in bio if int(r["subject"]) == sub), None)
        if b:
            st = dict(age=int(b["Age"]), sex=b["Gender"], bmi=round(float(b["BMI"]), 1), waist=None, hba1c=float(b["A1c PDL (Lab)"]))
    return dict(id=f"C{sub:03d}", src="cgmacros", cgm=cgm, steps=steps, meals=meals, static=st)


def load_cgmacros_dir(d, min_a1c=5.7, sensor="dexcom"):
    """All participants with HbA1c >= min_a1c (prediabetes + T2D = the dev cohort in the plan), common length."""
    import glob
    bio = list(csv.DictReader(open(os.path.join(d, "bio.csv"), encoding="utf-8-sig")))
    ps = [load_cgmacros(f, bio, sensor) for f in sorted(glob.glob(os.path.join(d, "CGMacros-*.csv")))]
    ps = [p for p in ps if (p["static"]["hba1c"] or 0) >= min_a1c and np.isfinite(p["cgm"]).mean() > 0.7 and len(p["cgm"]) >= 2900]
    n = min(len(p["cgm"]) for p in ps)
    for p in ps:
        p["cgm"], p["steps"] = p["cgm"][:n], p["steps"][:n]
        p["meals"] = [m for m in p["meals"] if m[0] < n]
    return ps


# Rough carbs per 100 g/ml for ShanghaiT2DM free-text diet logs (keyword -> g carb). Crude estimates, NOT IFCT/CFCT-verified.
CARB100 = [("fried rice", 30), ("porridge", 10), ("congee", 10), ("rice noodle", 25), ("noodle", 25), ("dumpling", 25), ("bun", 45),
           ("bread", 47), ("steamed", 30), ("pancake", 40), ("rice", 27), ("coarse grain", 25), ("oat", 12), ("potato", 17), ("yam", 15),
           ("corn", 20), ("pumpkin", 6), ("lotus", 16), ("apple", 13), ("pear", 12), ("orange", 11), ("banana", 22), ("grape", 17),
           ("watermelon", 6), ("fruit", 12), ("milk", 5), ("yogurt", 9), ("soy", 3), ("soup", 3), ("cake", 50), ("biscuit", 65),
           ("cookie", 65), ("sugar", 100), ("juice", 11)]


def _carbs_of(text):
    import re
    tot = 0.0
    for line in str(text).replace("\r", "").split("\n"):
        m = re.match(r"(.+?)\s*([\d.]+)\s*(g|ml)\b", line.strip().lower())
        if not m:
            continue
        name, q = m.group(1), float(m.group(2))
        tot += q / 100 * next((c for k, c in CARB100 if k in name), 4)  # default 4 g/100 g: vegetables, meat, egg, tofu
    return tot


def _rows(f):
    if f.endswith(".xls"):
        import xlrd
        s = xlrd.open_workbook(f).sheet_by_index(0)
        from datetime import datetime, timedelta
        return [[(datetime(1899, 12, 30) + timedelta(days=c)) if k == 0 and isinstance(c, float) else c for k, c in enumerate(s.row_values(r))] for r in range(1, s.nrows)]
    import openpyxl
    return [list(r) for r in openpyxl.load_workbook(f, read_only=True).active.iter_rows(min_row=2, values_only=True)]


def load_shanghai_dir(d, min_days=6.5, min_meals=12):
    """ShanghaiT2DM (figshare 20444397, CC BY 4.0): 15-min Libre CGM + free-text diet logs. One record per patient (longest
    eligible visit). Visits with any insulin entry are excluded (the twin has no insulin input). 15-min CGM is linearly
    interpolated to the 5-min grid across gaps <=45 min. Carbs are keyword estimates; steps/HbA1c unavailable."""
    import glob
    best = {}
    for f in sorted(glob.glob(os.path.join(d, "*.xls*"))):
        R = [r for r in _rows(f) if r[0] not in ("", None)]
        if not R or any(r[k] not in ("", None) for r in R for k in (6, 8, 9, 10)):
            continue
        t0 = R[0][0].replace(hour=0, minute=0, second=0)
        m = np.array([(r[0] - t0).total_seconds() / 60 for r in R])
        y = np.array([float(r[1]) if r[1] not in ("", None) else np.nan for r in R])
        n = int(m[-1] // STEP) + 1
        ok = np.isfinite(y)
        grid = np.arange(n) * STEP
        cgm = np.interp(grid, m[ok], y[ok], left=np.nan, right=np.nan)
        nxt = np.searchsorted(m[ok], grid)
        mo = m[ok]
        bad = np.array([(0 < j < len(mo)) and (mo[j] - mo[j - 1] > 45) for j in nxt])
        cgm[bad] = np.nan
        meals = []
        for r in R:
            txt = r[4]
            if txt in ("", None) or "not available" in str(txt).lower():
                continue
            c = min(_carbs_of(txt), 150.0)
            if c >= 5:
                meals.append((int(round((r[0] - t0).total_seconds() / 60 / STEP)), c, 0.3, "logged meal"))
        meals = [x for x in meals if x[0] < n]
        pid = os.path.basename(f)[:4]
        if n / 288 >= min_days and len(meals) >= min_meals and np.isfinite(cgm).mean() > 0.6 and (pid not in best or n > len(best[pid]["cgm"])):
            best[pid] = dict(id="S" + pid, src="shanghai", cgm=cgm, steps=np.zeros(n), meals=meals,
                             static=dict(age=None, sex=None, bmi=None, waist=None, hba1c=None))
    return list(best.values())


def fhir_bundle(p):
    """FHIR R4 Bundle: Patient + Conditions (ICD-10) + Observations (LOINC). Static fields from the dataset; mock IDs."""
    g = p["cgm"][np.isfinite(p["cgm"])]
    pid = f"patient-{p['id']}"
    ent = [{"resource": {"resourceType": "Patient", "id": pid, "gender": {"F": "female", "M": "male"}.get(p["static"]["sex"], "unknown")}},
           {"resource": {"resourceType": "Condition", "subject": {"reference": f"Patient/{pid}"},
                         "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "E11", "display": "Type 2 diabetes mellitus"}]}}}]
    def obs(loinc, name, val, unit):
        return {"resource": {"resourceType": "Observation", "status": "final", "subject": {"reference": f"Patient/{pid}"},
                             "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": name}]},
                             "valueQuantity": {"value": round(float(val), 1), "unit": unit}}}
    if p["static"].get("hba1c"):
        ent.append(obs("4548-4", "Hemoglobin A1c", p["static"]["hba1c"], "%"))
    ent.append(obs("2339-0", "Glucose, mean CGM", g.mean(), "mg/dL"))
    return {"resourceType": "Bundle", "type": "collection", "entry": ent,
            "meta": {"tag": [{"display": "Synthetic/mock data. Not for clinical use."}]}}
