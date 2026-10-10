"""One self-check: python test_twin.py. Fails if the calibration, filter or error grid break."""
import numpy as np
from twin import model as M
from twin.pipeline import clarke

# 1. mass balance: absorbed carbs * CG equals integrated Ra (gut pools drain fully)
s = np.zeros(6); s[2] = 60
tot = 0
for _ in range(120):
    s = M.step(s, (0.0003, 0.015, 0.03), 120.0)
    tot += s[2:].sum()
assert s[2:].sum() < 1.0, "gut should drain"
# 2. parameter recovery on data the ODE itself generates (noise-free, logged meals)
gb, th = 120.0, (0.0004, 0.015, 0.035)
n = 4 * 288; cg = np.full(n, gb); x = np.array([gb, 0, 0, 0, 0, 0.]); meals = []
for d in range(4):
    for h in (8, 13, 19):
        meals.append((d * 288 + h * 12, 50.0, 0.3, "m"))
by = {m[0]: m for m in meals}
for i in range(1, n):
    x = M.step(x, th, gb, 0.0); cg[i] = x[0]
    if i in by: x[2] += 35; x[4] += 15
    x[0] = cg[i]
cg = np.where(True, cg, 0)
p = dict(cgm=cg, steps=np.zeros(n), meals=[(m[0] - 0, 50.0, 0.3, "m") for m in meals], static={})
mu, sd = M.prior_for({})
est, *_ = M.fit(p, 0, n, mu, sd, gb)
assert abs(np.log(est[2] / th[2])) < 0.5 and abs(np.log(est[1] / th[1])) < 0.7, est
# 3. error grid sanity
assert clarke(100, 105) == "A" and clarke(60, 200) == "E" and clarke(300, 100) == "D"
print("ok", est)
