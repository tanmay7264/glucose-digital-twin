"""Hybrid glucose twin: Bergman-style minimal model (+ two-pool gut) -> per-patient calibration -> UKF state
update -> ridge residual correction. numpy/scipy only.

State s = [G, X, F1, F2, S1, S2]: glucose (mg/dL), remote insulin action (1/min), fast gut pool (2 comps, g),
slow gut pool (2 comps, g). Constants marked ASSUMED are literature-style fixed values, not fitted.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

DT, SUB = 5.0, 5
P2 = 0.02      # ASSUMED remote-insulin rate (1/min)
CG = 7.6       # ASSUMED mg/dL per g carb (0.9 bioavail * 1000 / (1.7 dL/kg * 70 kg))
KEX = 0.012    # ASSUMED extra insulin-independent uptake per unit exercise (1/min); least certain constant
KSLOW = 0.4    # slow pool absorbs at 0.4 x fast rate
R_OBS = 8.0 ** 2
H = (6, 12, 24)            # 30/60/120 min
HS = (0, 3, 6, 9, 12, 18, 24)  # horizons shipped to the dashboard
THR = 180.0


def step(s, th, gb, e=0.0):
    """Advance state(s) one 5-min grid step. s (...,6); th=(SI,SG,k) scalars or arrays over leading dims."""
    SI, SG, k = th
    G, X, F1, F2, S1, S2 = np.moveaxis(s, -1, 0)
    h, ks = DT / SUB, KSLOW * k
    for _ in range(SUB):
        ra = k * F2 + ks * S2
        dG = -(SG + KEX * e + X) * (G - gb) + CG * ra
        dX = -P2 * X + P2 * SI * np.maximum(G - gb, 0)
        G, X, F1, F2, S1, S2 = (G + h * dG, X + h * dX, F1 - h * k * F1, F2 + h * k * (F1 - F2),
                                S1 - h * ks * S1, S2 + h * ks * (S1 - S2))
    return np.stack([G, X, F1, F2, S1, S2], -1)


def rollout(s, th, gb, n=24, e=None):
    """Mean forecast G(t+1..t+n) from states s (N,6) (or (6,)), future meals unknown. e: optional (n,) activity."""
    s, out = np.array(s, float), []
    for j in range(n):
        s = step(s, th, gb, 0.0 if e is None else e[j])
        out.append(s[..., 0])
    return np.stack(out, -1)


def ex_of(steps):
    return np.clip(np.nan_to_num(steps) / 100.0, 0, 1)


# ---------------------------------------------------------------- calibration
def prior_for(static):
    """Prior on log[SI, SG, k]. HbA1c shifts SI (higher HbA1c -> more resistant). Hypothesis-driven, not fitted on Indian data."""
    a1c = static.get("hba1c") or 7.0
    return np.log([0.0004, 0.015, 0.035]) + np.array([-0.35 * (a1c - 7.0), 0, 0]), np.array([0.8, 0.7, 0.6])


def fit(p, lo, hi, mu, sd, gb, n=48, pre=6):
    """Segment ('multiple shooting') fit: 4 h windows starting 30 min before each logged meal in [lo,hi).
    Returns (theta=exp(x), x, cov) - cov is the Gaussian posterior in log space (prior included)."""
    cg, ex = p["cgm"], ex_of(p["steps"])
    segs = [m[0] - pre for m in p["meals"] if lo <= m[0] - pre and m[0] - pre + n <= hi and np.isfinite(cg[m[0] - pre])
            and np.isfinite(cg[m[0] - pre: m[0] - pre + n]).mean() > 0.8]
    segs = sorted(set(segs))
    if len(segs) < 5:
        return np.exp(mu), mu, np.diag(sd ** 2)
    N = len(segs)
    Y = np.array([cg[t: t + n] for t in segs])
    EX = np.array([ex[t: t + n] for t in segs])
    AF, AS = np.zeros((N, n)), np.zeros((N, n))
    for j, t0 in enumerate(segs):
        for (ti, c, sl, *_r) in p["meals"]:
            if t0 <= ti < t0 + n:
                AF[j, ti - t0] += c * (1 - sl)
                AS[j, ti - t0] += c * sl
    mask = np.isfinite(Y)

    def res(x):
        th = np.exp(x)
        s = np.zeros((N, 6))
        s[:, 0], s[:, 1] = Y[:, 0], th[0] * np.maximum(Y[:, 0] - gb, 0)
        out = np.empty((N, n))
        for j in range(n):
            s[:, 2] += AF[:, j]
            s[:, 4] += AS[:, j]
            out[:, j] = s[:, 0]
            s = step(s, tuple(th), gb, EX[:, j])
        return np.r_[((out - np.where(mask, Y, 0)) / 8.0)[mask], (x - mu) / sd]

    sol = least_squares(res, mu, loss="soft_l1", f_scale=2.0, bounds=(np.log([1e-5, 0.001, 0.01]), np.log([2e-3, 0.05, 0.12])))
    J = sol.jac
    return np.exp(sol.x), sol.x, np.linalg.inv(J.T @ J + 1e-6 * np.eye(3))


# ---------------------------------------------------------------- UKF
class UKF:
    n, lam = 6, 1.0

    def __init__(self, th, gb, y0):
        self.th, self.gb = tuple(th), gb
        self.x = np.array([y0, th[0] * max(y0 - gb, 0), 0, 0, 0, 0.])
        self.P = np.diag([R_OBS, 0.005 ** 2, 1, 1, 1, 1.])
        self.Q = np.diag([3.0 ** 2, 0.0008 ** 2, .5 ** 2, .5 ** 2, .5 ** 2, .5 ** 2])
        self.wm = np.r_[self.lam / (self.n + self.lam), np.full(2 * self.n, 0.5 / (self.n + self.lam))]
        self.wc = self.wm.copy()
        self.wc[0] += 1 - 1 + 2

    def predict(self, e):
        n = self.n
        try:
            L = np.linalg.cholesky((n + self.lam) * self.P)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(self.P)
            L = v * np.sqrt(np.maximum(w, 1e-9) * (n + self.lam))
        pts = self.x + np.vstack([np.zeros(n), L.T, -L.T])
        pts = step(pts, self.th, self.gb, e)
        pts[:, 2:] = np.maximum(pts[:, 2:], 0)
        self.x = self.wm @ pts
        d = pts - self.x
        self.P = (self.wc[:, None] * d).T @ d + self.Q
        self.P = (self.P + self.P.T) / 2

    def update(self, y):
        S = self.P[0, 0] + R_OBS
        K = self.P[:, 0] / S
        v = y - self.x[0]
        self.x = self.x + K * v
        self.P = self.P - np.outer(K, K) * S
        self.x[2:] = np.maximum(self.x[2:], 0)
        return v, S

    def meal(self, carbs, sl):
        f, s = carbs * (1 - sl), carbs * sl
        self.x[2] += f
        self.x[4] += s
        self.P[2, 2] += (0.25 * f) ** 2   # ~25% portion-size uncertainty
        self.P[4, 4] += (0.25 * s) ** 2


def stream(p, th, cov, gb, cal_end, refit=None, shift=12.0):
    """Online assimilation. Returns states (n,6), nis (n,), versions [(start_i, th, cov)], events (drift/recalibration)."""
    cg, n, ex = p["cgm"], len(p["cgm"]), ex_of(p["steps"])
    by_i = {}
    for m in p["meals"]:
        by_i.setdefault(m[0], []).append(m)
    i0 = int(np.argmax(np.isfinite(cg)))
    ukf = UKF(th, gb, cg[i0])
    states, nis, ver = np.tile(ukf.x, (n, 1)), np.full(n, np.nan), np.zeros(n, int)
    versions, events, last_fit = [(0, th, cov)], [], cal_end
    f60, err = np.full(n, np.nan), np.full(n, np.nan)   # drift monitor: signed error of the 60-min forecast, known 60 min later
    for i in range(i0 + 1, n):
        ukf.predict(ex[i - 1])
        if np.isfinite(cg[i]):
            v, S = ukf.update(cg[i])
            nis[i] = v * v / S
        for m in by_i.get(i, []):
            ukf.meal(m[1], m[2])
        states[i], ver[i] = ukf.x, len(versions) - 1
        f60[i] = rollout(ukf.x, ukf.th, gb, 12)[-1]
        err[i] = cg[i] - f60[i - 12] if i >= 12 else np.nan
        if i == cal_end:
            base = np.nanmean(err[i - 288: i])
        if refit and i > cal_end and i - last_fit >= 288:
            roll = np.nanmean(err[i - 287: i + 1]) - base   # mg/dL shift of 24 h mean forecast error vs calibration window
            if abs(roll) > shift:
                last_fit = i
                new = refit(i, versions[-1][1])
                events.append(dict(i=i, ratio=float(roll), old=list(map(float, versions[-1][1])),
                                   new=None if new is None else list(map(float, new[0]))))
                if new is not None:
                    versions.append((i + 1, new[0], new[1]))
                    ukf.th = tuple(new[0])
    return states, nis, versions, events


def forecast_all(states, versions, gb):
    n = len(states)
    F = np.empty((n, 24))
    b = [v[0] for v in versions] + [n]
    for j, (s0, th, _) in enumerate(versions):
        F[s0: b[j + 1]] = rollout(states[s0: b[j + 1]], tuple(th), gb)
    return F


# ---------------------------------------------------------------- ML residual
def base_feats(p, states, gb):
    n, Gf = len(states), states[:, 0]
    sl = Gf - np.r_[np.full(3, Gf[0]), Gf[:-3]]
    cs = np.r_[0, np.cumsum(np.nan_to_num(p["steps"]))]
    st30 = (cs[np.arange(n) + 1] - cs[np.maximum(np.arange(n) - 5, 0)]) / 6
    hr = (np.arange(n) * 5 / 60) % 24
    last, carb = np.full(n, -1e6), np.zeros(n)
    for m in p["meals"]:
        last[m[0]] = m[0]
        carb[m[0]] += m[1]
    since = (np.arange(n) - np.maximum.accumulate(last)) * 5 / 60
    cc = np.r_[0, np.cumsum(carb)]
    c2 = cc[np.arange(n) + 1] - cc[np.maximum(np.arange(n) - 24, 0)]
    a1c = (p["static"].get("hba1c") or 7.0) - 7
    return np.column_stack([Gf - gb, states[:, 1] * 100, states[:, 2] + states[:, 3], states[:, 4] + states[:, 5], sl,
                            st30 / 100, np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), np.minimum(since, 6) / 6,
                            c2 / 50, np.full(n, a1c), Gf - np.r_[np.full(6, Gf[0]), Gf[:-6]], Gf - np.r_[np.full(12, Gf[0]), Gf[:-12]],
                            sl - (np.r_[np.full(3, Gf[0]), Gf[:-3]] - np.r_[np.full(6, Gf[0]), Gf[:-6]])])


def ridge(X, y, lam=20.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return dict(mu=mu, sd=sd, w=w, b=y.mean())


def ridge_pred(m, X):
    if hasattr(m, "predict"):
        return m.predict(X)
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def hfeat(B, F, states, h):
    return np.column_stack([B, F[:, h - 1] - states[:, 0]])


def fit_resid(p, B, F, states, lo, hi):
    models = {}
    for h in H:
        idx = np.arange(max(lo, 24), hi - h)
        y = p["cgm"][idx + h] - F[idx, h - 1]
        ok = np.isfinite(y)
        models[h] = ridge(hfeat(B, F, states, h)[idx][ok], y[ok])
    return models


def apply_resid(models, B, F, states):
    corr = np.column_stack([np.zeros(len(F))] + [ridge_pred(models[h], hfeat(B, F, states, h)) for h in H])
    full = np.stack([np.interp(np.arange(1, 25), (0,) + H, c) for c in corr])
    return F + full, full


def sigma_h(p, Fh, lo, hi):
    """Empirical forecast error SD per horizon (1..24) from the calibration window."""
    sg = []
    for h in H:
        idx = np.arange(max(lo, 24), hi - h)
        e = p["cgm"][idx + h] - Fh[idx, h - 1]
        sg.append(max(float(np.sqrt(np.nanmean(e ** 2))), 6.0))
    return np.interp(np.arange(1, 25), (0,) + H, [4.0] + sg)


def p_exceed(Fh, sig):
    return norm.cdf((Fh - THR) / sig).max(1)
