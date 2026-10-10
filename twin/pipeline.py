"""Per-patient twin pipeline + patient-level evaluation. Writes out/results.json (dashboard) and out/state.pkl (what-if server)."""
import json, pickle, sys, time
import numpy as np
from scipy.stats import rankdata
from . import model as M
from .data import gen_patient, simulate, MEALS, meal_entry, STEP

REAL_NOTE = ("REAL CGMacros data (PhysioNet, CC BY-NC-SA 4.0): US adults with HbA1c >= 5.7 (prediabetes + T2D, >=10 days of data), Dexcom G6, meal macros from photos. "
             "Small cohort, single dataset, no external validation yet, no Indian data.")
CAL = 4 * 288           # calibrate on first 4 days, test on the rest
rng_ = np.random.default_rng(0)


def clarke(ref, pred):
    """Clarke error grid zone (A..E) for scalars."""
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= 7 / 5 * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or (175 / 3 <= ref <= 70 and pred >= 6 / 5 * ref):
        return "D"
    return "B"


def auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def avg_prec(y, s):
    y = np.asarray(y, bool)[np.argsort(-s)]
    return float((np.cumsum(y) / np.arange(1, len(y) + 1))[y].mean()) if y.any() else float("nan")


def alerts(pexc, now, thr, refractory=24):
    out, last = [], -10 ** 9
    for i in range(len(pexc)):
        if pexc[i] >= thr and now[i] <= M.THR and i - last >= refractory:
            out.append(i)
            last = i
    return out


def ffill(x, lim=3):
    y, last, age = x.copy(), np.nan, 99
    for i in range(len(y)):
        if np.isfinite(y[i]):
            last, age = y[i], 0
        else:
            age += 1
            y[i] = last if age <= lim else np.nan
    return y


def pop_prior(patients):
    """Empirical-Bayes prior: population mean/SD of per-patient log-params (weak-prior fits), HbA1c slope kept on SI."""
    X, A = [], []
    for p in patients:
        gb = float(np.nanpercentile(p['cgm'][:CAL], 15)); mu, sd = M.prior_for(p['static'])
        X.append(M.fit(p, 0, CAL, mu, sd, gb)[1]); A.append(p['static'].get('hba1c') or 7.0)
    X = np.array(X); return X.mean(0), X.std(0) + 0.05, float(np.mean(A))


def process(p, recal=True, pop=None, shift=12.0):
    cg, n = p["cgm"], len(p["cgm"])
    gb = float(np.nanpercentile(cg[:CAL], 15))
    mu, sd = M.prior_for(p["static"])
    if pop is not None:  # shrink toward population; keep HbA1c-driven SI shift relative to cohort mean
        mu = pop[0] + np.array([-0.35 * ((p["static"].get("hba1c") or pop[2]) - pop[2]), 0, 0]); sd = pop[1]
    th, x, cov = M.fit(p, 0, CAL, mu, sd, gb)

    def refit(i, th_now):
        m = np.log(th_now)
        r = M.fit(p, max(0, i - 864), i, m, np.full(3, 0.5), gb)
        return (r[0], r[2]) if r[0] is not None else None

    states, nis, versions, events = M.stream(p, th, cov, gb, CAL, refit if recal else None, shift)
    F = M.forecast_all(states, versions, gb)
    B = M.base_feats(p, states, gb)
    models = M.fit_resid(p, B, F, states, 0, CAL)
    Fh, corr = M.apply_resid(models, B, F, states)
    sig = M.sigma_h(p, Fh, 0, CAL)
    return dict(B=B, p=p, gb=gb, th=th, cov=cov, states=states, nis=nis, versions=versions, events=events, F=F, Fh=Fh, corr=corr,
                sig=sig, pexc=M.p_exceed(Fh, sig), pexc_ode=M.p_exceed(F, np.interp(np.arange(1, 25), (0,) + M.H, [4, 14, 24, 40])))


def pool_residuals(rs, lam=50.0):
    """One residual model per horizon trained on the calibration windows of ALL patients (test days never touched)."""
    models = {}
    for h in M.H:
        X, y = [], []
        for r in rs:
            idx = np.arange(24, CAL - h)
            t = r["p"]["cgm"][idx + h] - r["F"][idx, h - 1]
            ok = np.isfinite(t)
            X.append(M.hfeat(r["B"], r["F"], r["states"], h)[idx][ok]); y.append(t[ok])
        models[h] = M.ridge(np.vstack(X), np.concatenate(y), lam)
    for r in rs:
        r["Fh"], r["corr"] = M.apply_resid(models, r["B"], r["F"], r["states"])
        r["sig"] = M.sigma_h(r["p"], r["Fh"], 0, CAL)
        r["pexc"] = M.p_exceed(r["Fh"], r["sig"])
    return models


def _hgb():
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    return HGB(max_iter=150, learning_rate=0.06, max_depth=4, min_samples_leaf=80, l2_regularization=5)


def _apply(r, models):
    r["Fh"], r["corr"] = M.apply_resid(models, r["B"], r["F"], r["states"])
    r["sig"] = M.sigma_h(r["p"], r["Fh"], 0, CAL)
    r["pexc"] = M.p_exceed(r["Fh"], r["sig"])


def cv_hgb(rs, k=5):
    """Gradient-boosted residual models with patient-level K-fold CV: each patient's forecasts come from models trained on OTHER
    patients' full recordings. Returns ({fold: models}, {pid: fold}); rs forecasts are overwritten with the held-out versions."""
    fold = {r["p"]["id"]: i % k for i, r in enumerate(rs)}
    out = {}
    for f in range(k):
        tr = [r for r in rs if fold[r["p"]["id"]] != f]
        models = {}
        for h in M.H:
            X, y = [], []
            for r in tr:
                idx = np.arange(24, len(r["p"]["cgm"]) - h)
                t = r["p"]["cgm"][idx + h] - r["F"][idx, h - 1]
                ok = np.isfinite(t)
                X.append(M.hfeat(r["B"], r["F"], r["states"], h)[idx][ok]); y.append(t[ok])
            models[h] = _hgb().fit(np.vstack(X), np.concatenate(y))
        out[f] = models
        for r in rs:
            if fold[r["p"]["id"]] == f:
                _apply(r, models)
    return out, fold


def alert_eval(rs, fa_budget=2.0, lead_ok=3):
    """Alert policy with ONE pooled threshold (chosen on all patients' calibration windows for <= fa_budget false alerts per patient-day;
    per-patient thresholds from 4 days were noisy and fired late). Lead time is measured per excursion: minutes between the EARLIEST alert
    in the preceding 2 h and the up-crossing of 180 mg/dL."""
    thr_grid = np.arange(0.05, 0.96, 0.05)
    def run(r, lo, hi, thr):
        cg = r["p"]["cgm"]; now = ffill(cg); idx = np.arange(lo, hi)
        return [int(idx[i]) for i in alerts(r["pexc"][idx], now[idx], thr)]
    def false_per_day(thr, lo_f, hi_f):
        f = d = 0
        for r in rs:
            cg = r["p"]["cgm"]; hi = hi_f(len(cg)); a = run(r, lo_f, hi, thr); d += (hi - lo_f) / 288
            f += sum(1 for i in a if np.isfinite(cg[i + 1: i + 25]).any() and not np.nanmax(cg[i + 1: i + 25]) > M.THR)
        return f / d
    def score(thr):
        ev = warned = ok = fa = na = 0; days = 0; leads = []
        for r in rs:
            cg = r["p"]["cgm"]; n = len(cg); a = run(r, CAL, n - 24, thr); days += (n - 24 - CAL) / 288; na += len(a)
            fa += sum(1 for i in a if np.isfinite(cg[i + 1: i + 25]).any() and not np.nanmax(cg[i + 1: i + 25]) > M.THR)
            for c in range(CAL, n - 1):
                if np.isfinite(cg[c]) and np.isfinite(cg[c - 1]) and cg[c] > M.THR >= cg[c - 1]:
                    ev += 1; pre = [c - i for i in a if 0 < c - i <= 24]
                    if pre:
                        warned += 1; leads.append(max(pre) * STEP); ok += max(pre) >= lead_ok
        return dict(thr=float(thr), alerts_per_day=na / days, false_per_day=fa / days, events=ev, sensitivity=warned / max(ev, 1),
                    sensitivity_15min=ok / max(ev, 1), median_lead_min=float(np.median(leads)) if leads else None,
                    mean_lead_min=float(np.mean(leads)) if leads else None, patient_days=days)
    thr = next((t for t in thr_grid if false_per_day(t, 24, lambda n: CAL - 24) <= fa_budget), 0.95)
    for r in rs:
        r["thr"] = float(thr); r["alerts"] = run(r, CAL, len(r["p"]["cgm"]) - 24, thr)
    out = score(thr)
    out["curve"] = [score(t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    out["note"] = f"one pooled threshold ({thr:.2f}) set on calibration windows for <= {fa_budget} false alerts/patient-day; lead = earliest alert before each up-crossing of 180"
    return out


def evaluate(rs):
    """Pooled patient-level test-period metrics. Baselines: persistence, linear extrapolation, ODE+UKF, hybrid."""
    rows = {h: {k: [] for k in ("ref", "pers", "lin", "ode", "hyb", "pid")} for h in M.H}
    ex = {k: [] for k in ("y", "pers", "lin", "mealrule", "ode", "hyb")}
    cal_pts, per_patient = [], {}
    al = dict(n_alert=0, fa=0, ev=0, hit=0, lead=[], days=0)
    # threshold from calibration window: lowest thr with <=2 false alerts/patient-day
    thr_grid = np.arange(0.1, 0.96, 0.05)
    for r in rs:
        p = r["p"]; cg = p["cgm"]; n = len(cg); now = ffill(cg)
        lin = np.clip(now[:, None] + np.r_[np.zeros(3), now[3:] - now[:-3]][:, None] / 3 * np.arange(1, 25), 40, 400)
        lin_bad = ~np.isfinite(np.r_[np.full(3, np.nan), now[3:] - now[:-3]])
        lin[lin_bad] = np.nan
        t_idx = np.arange(CAL, n - 24)
        for h in M.H:
            ok = np.isfinite(cg[t_idx + h]) & np.isfinite(now[t_idx]) & np.isfinite(lin[t_idx, h - 1])
            ti = t_idx[ok]
            rw = rows[h]
            rw["ref"].append(cg[ti + h]); rw["pers"].append(now[ti]); rw["lin"].append(lin[ti, h - 1])
            rw["ode"].append(r["F"][ti, h - 1]); rw["hyb"].append(r["Fh"][ti, h - 1]); rw["pid"].append(np.full(len(ti), p["id"]))
        fut = np.array([np.nanmax(cg[i + 1: i + 25]) if np.isfinite(cg[i + 1: i + 25]).sum() > 18 else np.nan for i in t_idx])
        ok = np.isfinite(fut) & np.isfinite(now[t_idx]) & (now[t_idx] <= M.THR) & np.isfinite(lin[t_idx, 23])
        ti = t_idx[ok]
        ex["y"].append(fut[ok] > M.THR); ex["pers"].append(now[ti]); ex["lin"].append(np.nanmax(lin[ti], 1))
        ex["mealrule"].append(M.base_feats(p, r["states"], r["gb"])[ti, 9]); ex["ode"].append(r["pexc_ode"][ti]); ex["hyb"].append(r["pexc"][ti])
        cal_pts.append((r["pexc"][ti], fut[ok] > M.THR))

    res = {"horizons": {}}
    for h in M.H:
        rw = {k: np.concatenate(v) for k, v in rows[h].items()}
        d = {}
        for k in ("pers", "lin", "ode", "hyb"):
            e = rw[k] - rw["ref"]
            z = [clarke(a, b) for a, b in zip(rw["ref"][::3], rw[k][::3])]
            d[k] = dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), ab=float(np.mean([c in "AB" for c in z]) * 100))
        d["n"] = int(len(rw["ref"]))
        res["horizons"][str(h * STEP)] = d
    y = np.concatenate(ex["y"])
    res["excursion"] = {k: dict(auroc=auroc(y, np.concatenate(ex[k])), auprc=avg_prec(y, np.concatenate(ex[k]))) for k in ("pers", "lin", "mealrule", "ode", "hyb")}
    res["excursion"]["prevalence"] = float(y.mean())
    pp, yy = np.concatenate([c[0] for c in cal_pts]), np.concatenate([c[1] for c in cal_pts])
    res["brier"] = float(((pp - yy) ** 2).mean())
    bins = np.minimum((pp * 10).astype(int), 9)
    res["reliability"] = [dict(p=float(pp[bins == b].mean()), obs=float(yy[bins == b].mean()), n=int((bins == b).sum())) for b in range(10) if (bins == b).sum() >= 20]
    res["alerts"] = alert_eval(rs)
    for r in rs:  # per patient rmse60 for the table
        p = r["p"]; cg = p["cgm"]; t = np.arange(CAL, len(cg) - 12); ok = np.isfinite(cg[t + 12])
        per_patient[p["id"]] = dict(rmse60_hyb=float(np.sqrt(np.mean((cg[t + 12][ok] - r["Fh"][t, 11][ok]) ** 2))),
                                    rmse60_pers=float(np.sqrt(np.nanmean((cg[t + 12] - ffill(cg)[t]) ** 2))))
    res["per_patient"] = per_patient
    return res


def counterfactual_check(rs, max_meals=5):
    """Synthetic only: does the twin's predicted effect of a 15-min post-meal walk match the generator's true effect?"""
    out = []
    for r in rs:
        p = r["p"]; cfg = p["gen"]; gb = r["gb"]
        meals = [m for m in cfg["meals"] if m["logged"] and CAL + 12 < m["t"] // STEP < len(p["cgm"]) - 40 and 17 * 60 < m["t"] % 1440 < 22 * 60][:max_meals]
        for m in meals:
            i = m["t"] // STEP
            v = max(j for j, vv in enumerate(r["versions"]) if vv[0] <= i)
            th = tuple(r["versions"][v][1])
            e = np.zeros(36); e[6:9] = 1.0
            s0 = r["states"][i]
            peak = lambda ee: float(M.rollout(s0, th, gb, 36, ee).max())
            dtw = peak(None) - peak(e)
            g0 = simulate(cfg)[m["t"]: m["t"] + 180].max()
            g1 = simulate(cfg, [(m["t"] + 30, 15)])[m["t"]: m["t"] + 180].max()
            out.append(dict(pid=p["id"], twin=dtw, true=float(g0 - g1)))
    a = np.array([[o["twin"], o["true"]] for o in out])
    return dict(n=len(out), mean_twin=float(a[:, 0].mean()), mean_true=float(a[:, 1].mean()), mae=float(np.abs(a[:, 0] - a[:, 1]).mean()),
                corr=float(np.corrcoef(a.T)[0, 1]) if len(out) > 2 else None,
                note="synthetic ground truth; effect of a 15-min walk starting 30 min after dinner on 3-h peak glucose (mg/dL)")


def agp(cg):
    d = cg[: len(cg) // 288 * 288].reshape(-1, 288)
    d = d.reshape(len(d), 96, 3)
    d = np.nanmean(d, 2)
    return {k: np.nan_to_num(np.round(np.nanpercentile(d, q, 0), 0), nan=120).tolist() for k, q in (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))}


def tir(cg):
    g = cg[np.isfinite(cg)]
    return dict(low=float((g < 70).mean() * 100), inr=float(((g >= 70) & (g <= 180)).mean() * 100), high=float((g > 180).mean() * 100),
                mean=float(g.mean()), gmi=float(3.31 + 0.02392 * g.mean()))


def dump(rs, metrics, cf, path="out/results.json"):
    j = lambda a, d=0: [None if not np.isfinite(v) else round(float(v), d) for v in a]
    pts = []
    for r in rs:
        p, n, S = r["p"], len(r["p"]["cgm"]), r["states"]
        th_now = np.array([tuple(r["versions"][min(range(len(r["versions"])), key=lambda k: abs(r["versions"][k][0] - 10 ** 9))][1])])
        # mechanistic attribution of the 60-min forecast (mg/dL): meals on board, insulin action, ML correction
        v_of = np.zeros(n, int)
        for k, vv in enumerate(r["versions"]):
            v_of[vv[0]:] = k
        drv_meal, drv_ins = np.zeros(n), np.zeros(n)
        for k, vv in enumerate(r["versions"]):
            sl = slice(vv[0], r["versions"][k + 1][0] if k + 1 < len(r["versions"]) else n)
            th = tuple(vv[1]); s = S[sl].copy()
            s0 = s.copy(); s0[:, 2:] = 0
            s1 = s.copy(); s1[:, 1] = 0
            f = M.rollout(s, th, r["gb"], 12)[:, -1]
            drv_meal[sl] = f - M.rollout(s0, th, r["gb"], 12)[:, -1]
            drv_ins[sl] = f - M.rollout(s1, th, r["gb"], 12)[:, -1]
        pts.append(dict(
            id=p["id"], src=p["src"], static=p["static"], gb=round(r["gb"]), thr=round(r["thr"], 2),
            cgm=j(p["cgm"]), gf=j(S[:, 0]), p=j(r["pexc"], 2), sig=j(r["sig"], 1),
            fh={str(h): j(r["Fh"][:, h - 1]) for h in M.HS if h}, cob=j(S[:, 2:].sum(1), 0), x=j(S[:, 1] * 1000, 1),
            drv=dict(meal=j(drv_meal), ins=j(drv_ins), ml=j(r["corr"][:, 11])),
            meals=[[m[0], round(m[1]), m[3]] for m in p["meals"]], alerts=r["alerts"],
            versions=[dict(i=int(v[0]), th=list(map(float, v[1])), sd=list(map(float, np.sqrt(np.diag(v[2]))))) for v in r["versions"]],
            events=r["events"], agp=agp(p["cgm"]), tir=tir(p["cgm"]), steps=j(p["steps"])))
    meta = dict(cal_end=CAL, n=len(rs[0]["p"]["cgm"]), step_min=STEP, horizons=[h for h in M.HS if h],
                data_note=(REAL_NOTE if rs[0]["p"]["src"] == "cgmacros" else "SYNTHETIC patients from a generator structurally different from the twin. Numbers are stress-test results, "
                           "NOT clinical evidence. Swap in CGMacros/ShanghaiT2DM via run.py --cgmacros for real-data validation."),
                meals={k: dict(carbs=v[0], protein=v[1], fat=v[2], fiber=v[3], slow=round(meal_entry(k)[1], 2)) for k, v in MEALS.items()})
    json.dump(dict(meta=meta, patients=pts, metrics=metrics, counterfactual=cf), open(path, "w"), separators=(",", ":"))


PORT = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13]   # residual features available in every dataset (no steps, no HbA1c)


class Portable:
    def __init__(self, inner): self.inner = inner
    def predict(self, X): return self.inner.predict(X[:, PORT + [X.shape[1] - 1]])


def train_portable(cg_dir="data/cgmacros", shift=20.0):
    """Residual models for transfer: trained on CGMacros *Libre-15-min* (same cadence/sensor family/no-activity as Shanghai), all days of
    all patients, gradient boosting. Chosen by leave-patients-out CV on CGMacros only; Shanghai is never used for selection."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    from .data import load_cgmacros_dir
    ps = load_cgmacros_dir(cg_dir, sensor="libre15")
    pop = pop_prior(ps)
    rs = [process(p, True, pop, shift) for p in ps]
    models = {}
    for h in M.H:
        X, y = [], []
        for r in rs:
            idx = np.arange(24, len(r["p"]["cgm"]) - h)
            t = r["p"]["cgm"][idx + h] - r["F"][idx, h - 1]
            ok = np.isfinite(t)
            X.append(M.hfeat(r["B"], r["F"], r["states"], h)[idx][ok]); y.append(t[ok])
        X, y = np.vstack(X), np.concatenate(y)
        models[h] = Portable(HGB(max_iter=150, learning_rate=0.06, max_depth=4, min_samples_leaf=80, l2_regularization=5).fit(X[:, PORT + [X.shape[1] - 1]], y))
    return models, pop


def external_validation(models, pop, shift, path="data/shanghai/t2dm"):
    """Frozen CGMacros components (population prior, pooled residual models) applied to ShanghaiT2DM. Only the per-patient ODE
    parameters are fitted, on each patient's first 3 days. No Shanghai data touches any tuned component."""
    global CAL
    import os
    from .data import load_shanghai_dir
    if not os.path.isdir(path):
        return None
    ps = load_shanghai_dir(path)
    old, CAL = CAL, 3 * 288
    try:
        rs = [process(p, True, pop, shift) for p in ps]
        for r in rs:
            r["Fh"], r["corr"] = M.apply_resid(models, r["B"], r["F"], r["states"])
            r["sig"] = M.sigma_h(r["p"], r["Fh"], 0, CAL)
            r["pexc"] = M.p_exceed(r["Fh"], r["sig"])
        m = evaluate(rs)
    finally:
        CAL = old
    m["n_patients"] = len(rs)
    m["note"] = "ShanghaiT2DM: 15-min Libre (interpolated to 5 min), carbs = keyword estimates from free-text diet logs, no insulin-treated visits, no steps, calibrate on first 3 days."
    return m


def main(patients=None, n=12, pooled=True, shrink=True, shift=20.0):
    t = time.time()
    patients = patients or [gen_patient(f"P{k:02d}", k, drift_day=7 if k in (3, 8, 11) else None) for k in range(1, n + 1)]
    pop = pop_prior(patients) if shrink else None
    rs = [process(p, True, pop, shift) for p in patients]
    models = pool_residuals(rs) if pooled else None
    print(f"twin built for {len(rs)} patients in {time.time() - t:.0f}s")
    metrics = evaluate(rs)
    fold_models = None
    if pooled:
        r0 = metrics
        fold_models, pid_fold = cv_hgb(rs)
        metrics = evaluate(rs)
        metrics["internal_ridge"] = {k: r0[k] for k in ("horizons", "excursion", "alerts", "brier")}
    # drift ablation: patients that triggered >=1 recalibration, re-run with recalibration disabled, compare 60-min RMSE after first trigger
    tr = [r for r in rs if r["events"]]
    if tr:
        nr = [process(r["p"], False, pop, shift) for r in tr]
        if fold_models:
            for r in nr:
                _apply(r, fold_models[pid_fold[r["p"]["id"]]])
        def post(r, i0):
            cg = r["p"]["cgm"]; t = np.arange(i0, len(cg) - 12); e = cg[t + 12] - r["Fh"][t, 11]
            return float(np.sqrt(np.nanmean(e ** 2)))
        w = [post(r, r["events"][0]["i"]) for r in tr]; wo = [post(r0, r["events"][0]["i"]) for r, r0 in zip(tr, nr)]
        metrics["drift_ablation"] = dict(patients=[r["p"]["id"] for r in tr], rmse60_after_drift_with_recal=float(np.mean(w)),
                                         rmse60_after_drift_without=float(np.mean(wo)), recalibrations=sum(len(r["events"]) for r in tr))
    if models and patients[0]["src"] == "cgmacros":
        v1 = external_validation(models, pop, shift)           # v1: Dexcom-trained ridge (first attempt, kept for transparency)
        pm, ppop = train_portable(shift=shift)
        ext = external_validation(pm, ppop, shift)
        if ext:
            ext["v1_ridge"] = {k: v1[k] for k in ("horizons", "excursion", "alerts")}
            metrics["external"] = ext
    cf = counterfactual_check(rs) if patients[0]["src"] == "synthetic" else None
    dump(rs, metrics, cf)
    pickle.dump({r["p"]["id"]: dict(gb=r["gb"], states=r["states"], versions=r["versions"], p=dict(id=r["p"]["id"], static=r["p"]["static"], cgm=r["p"]["cgm"], steps=r["p"]["steps"], meals=r["p"]["meals"], src=r["p"]["src"])) for r in rs},
                open("out/state.pkl", "wb"))
    return rs, metrics, cf


if __name__ == "__main__":
    rs, m, cf = main()
    print(json.dumps(m, indent=1)); print(cf)
