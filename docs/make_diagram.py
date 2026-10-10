"""Generates docs/architecture.svg (and .html wrapper for PDF export). Run: python3 docs/make_diagram.py"""
W, H = 1600, 1040
o = []
def box(x, y, w, h, title, lines=(), fill="#ffffff", stroke="#9aa7b8", tc="#1c2430", r=10, tsize=15):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    o.append(f'<text x="{x+12}" y="{y+22}" font-size="{tsize}" font-weight="700" fill="{tc}">{title}</text>')
    for i, l in enumerate(lines):
        o.append(f'<text x="{x+12}" y="{y+42+i*17}" font-size="12.5" fill="#46515f">{l}</text>')
def arrow(x1, y1, x2, y2, label="", col="#5a6b85", dash=False, lx=0, ly=-6):
    d = ' stroke-dasharray="6 4"' if dash else ""
    o.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{col}" stroke-width="2" fill="none" marker-end="url(#a)"{d}/>')
    if label: o.append(f'<text x="{(x1+x2)/2+lx}" y="{(y1+y2)/2+ly}" font-size="11.5" fill="{col}" text-anchor="middle">{label}</text>')
def band(x, y, w, h, label, fill):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" opacity="0.55"/>')
    o.append(f'<text x="{x+w-14}" y="{y+22}" font-size="13" font-weight="700" fill="#5a6b85" letter-spacing="1" text-anchor="end">{label}</text>')

o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
o.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5a6b85"/></marker></defs>')
o.append(f'<rect width="{W}" height="{H}" fill="#f6f7f9"/>')
o.append('<text x="30" y="40" font-size="24" font-weight="800" fill="#1c2430">Glucose Digital Twin: architecture</text>')
o.append('<text x="30" y="62" font-size="13" fill="#6b7686">T2D / prediabetes post-meal excursion twin. Mechanistic core + UKF state estimation + boosted ML residual. Decision support for lifestyle, reviewed by a clinician.</text>')

# --- layer 1: data
band(20, 80, 1560, 150, "1  DATA IN", "#e8eef7")
box(40, 112, 270, 105, "CGMacros (PhysioNet)", ["Dexcom G6 + Libre, meal macros,", "Fitbit METs, HbA1c. 29 pts, HbA1c &gt;= 5.7", "Development + training  (CC BY-NC-SA)"], "#fff")
box(330, 112, 270, 105, "ShanghaiT2DM (figshare)", ["15-min Libre, free-text diet logs", "38 non-insulin T2D patients", "EXTERNAL validation only (CC BY 4.0)"], "#fff")
box(620, 112, 250, 105, "Synthetic generator", ["Structurally different from the twin", "Stress tests + counterfactual check", "Never a headline number"], "#fff7ec", "#e0b27a")
box(890, 112, 300, 105, "Static / clinical context", ["HbA1c, age, BMI (prior on insulin", "sensitivity); Indian meal library", "(18 items, carbs + slow fraction)"], "#fff")
box(1210, 112, 350, 105, "Interoperability (mock / roadmap)", ["FHIR R4 Bundle out (LOINC 4548-4, 2339-0,", "ICD-10 E11).  ABHA / ABDM consented pull:", "architecture only, no real PHI"], "#f1faf6", "#7cc6a0")

# --- layer 2: ingestion
band(20, 245, 1560, 105, "2  INGESTION (twin/data.py)", "#eef1f6")
box(40, 275, 1520, 62, "Align to a 5-min grid  |  15-min Libre linearly interpolated (gaps &gt; 45 min stay missing)  |  meal text/macros to carbs + slow-absorption fraction", ["Activity: METs to steps/min proxy (CGMacros) or none (Shanghai).  Future meal logs are never used as features (leakage guard)."], "#fff")

# --- layer 3: twin engine
band(20, 365, 1560, 385, "3  TWIN ENGINE (twin/model.py, twin/pipeline.py)", "#e6f4f4")
box(40, 398, 300, 150, "A  Calibrate (per patient)", ["Bergman-style minimal model + 2-pool gut", "State [G, X, F1, F2, S1, S2]", "Fit SI, SG, k on 4-h post-meal segments", "Empirical-Bayes prior (cohort + HbA1c)", "Posterior covariance kept"], "#fff", "#0f8b8d")
box(375, 398, 300, 150, "B  Assimilate (online)", ["Unscented Kalman filter, 1 step / reading", "Meal log = known input + 25% size", "uncertainty. Fills sensor gaps; the", "twin state updates at every sample"], "#fff", "#0f8b8d")
box(710, 398, 300, 150, "C  Forecast + risk", ["ODE roll-out 30 / 60 / 120 min", "+ gradient-boosted residual (patient-", "level CV, pooled across patients)", "P(glucose &gt; 180 within 2 h) with", "empirical per-horizon uncertainty"], "#fff", "#0f8b8d")
box(1045, 398, 240, 150, "D  Alert policy", ["One pooled threshold, set on", "calibration windows for", "&lt;= 2 false alerts / patient-day", "Per-excursion lead time"], "#fff", "#0f8b8d")
box(1320, 398, 240, 150, "E  What-if simulator", ["Meal swap / post-meal walk are", "ODE inputs, re-simulated with", "60 posterior samples + portion", "noise: bands, not point guesses"], "#fff", "#0f8b8d")
box(40, 580, 640, 150, "F  Drift monitor and recalibration loop (bidirectional)", ["24-h mean of the 60-min forecast error vs calibration window;", "shift &gt; 20 mg/dL triggers a refit on the last 3 days, prior = current", "parameters. Closed loop: recommend, simulated / observed response,", "compare forecast vs outcome, recalibrate.  Counterfactual check on", "synthetic truth: walk effect direction right, size over-stated."], "#fff", "#c9372c")
box(710, 580, 850, 150, "G  Evaluation harness (patient-level, time-ordered)", ["Baselines: persistence, linear extrapolation, ODE+UKF only, meal-rule.  Metrics: RMSE/MAE at 30/60/120 min, Clarke error grid,", "AUROC / AUPRC, Brier + reliability, alerts per day, sensitivity, lead time, drift ablation.", "Internal: CGMacros, 5-fold by patient.  External: Shanghai with the CGMacros-trained pieces frozen", "(Libre-15-min-trained residual, no steps / HbA1c features); only per-patient ODE parameters are fitted.", "Calibrate on first 3-4 days, test on the rest. Synthetic data is never the headline."], "#fff", "#7a5bd1")

# --- layer 4: serving + UI
band(20, 765, 1560, 195, "4  SERVING AND CLINICIAN UI", "#eef1f6")
box(40, 797, 330, 150, "Stdlib HTTP server (twin/server.py)", ["GET /api/results   precomputed twin + metrics", "POST /api/whatif   live ODE re-simulation", "GET /api/fhir/&lt;id&gt;   FHIR Bundle", "Static: dashboard, three.js (vendored)"], "#fff")
box(400, 797, 1160, 150, "Dashboard (web/index.html, web/human.js)", ["Triage list ranked by 2-h risk   |   Forecast cone with the real outcome revealed in replay   |   Risk, alert feed, mechanistic drivers of the 60-min forecast", "What-if: Indian meal A vs B (+ walk) with uncertainty bands   |   Twin state: SI / SG / k with 95% CI + drift events   |   AGP + time-in-range vs consensus targets", "Validation tab: baselines, external Shanghai table, alert operating curve, reliability, per-patient table   |   'Twin vs classifier' checklist", "3D physical twin: clothed figure walks / eats / sleeps from logged activity; CGM patch and floor ring show glucose zone; X-ray mode shows organs", "driven by model state (stomach = carbs on board, pancreas = insulin action).  WebGL failure falls back to a 2D figure."], "#fff")

# --- governance strip
o.append('<rect x="20" y="975" width="1560" height="50" rx="10" fill="#fff8e6" stroke="#f0dc9a"/>')
o.append('<text x="34" y="996" font-size="12.5" font-weight="700" fill="#7a5a00">Governance</text>')
o.append('<text x="120" y="996" font-size="12.5" fill="#5c4a14">Intended use: lifestyle decision support reviewed by a clinician; no diagnosis, no insulin dosing. CDSCO draft MDSW (Oct 2025): plausibly Class B (hypothesis).</text>')
o.append('<text x="120" y="1014" font-size="12.5" fill="#5c4a14">DPDP: public / synthetic data only here. Production: consent notice, ABDM consent artefact, store twin parameters not raw streams, erase on withdrawal. No Indian CGM data exists in this work; India-specific priors are labelled assumptions.</text>')

# arrows
arrow(175, 217, 175, 275); arrow(465, 217, 465, 275); arrow(745, 217, 745, 275); arrow(1040, 217, 1040, 275); arrow(1270, 217, 1270, 275, "FHIR in/out", dash=True, lx=34)
arrow(190, 337, 190, 398); arrow(525, 337, 525, 398)
arrow(340, 473, 375, 473, "theta"); arrow(675, 473, 710, 473, "state"); arrow(1010, 473, 1045, 473, "P"); arrow(1285, 473, 1320, 473)
arrow(525, 580, 525, 548, "", "#c9372c"); arrow(190, 548, 190, 580, "refit", "#c9372c", lx=-22)
arrow(860, 548, 860, 580, "", "#7a5bd1")
arrow(1100, 548, 1100, 580, "", "#7a5bd1")
arrow(205, 730, 205, 797); arrow(980, 730, 980, 797)
arrow(370, 872, 400, 872)
o.append('</svg>')
svg = "\n".join(o)
open("docs/architecture.svg", "w").write(svg)
open("docs/architecture.html", "w").write(f'<!doctype html><meta charset="utf-8"><style>@page{{size:{W}px {H}px;margin:0}}body{{margin:0}}svg{{display:block}}</style>{svg}')
print("ok")
