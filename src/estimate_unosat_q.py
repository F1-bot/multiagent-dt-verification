"""Estimate a CTMC deterioration generator Q from UNOSAT Mariupol multi-temporal damage assessment.
States: 0 = no visible / possible damage, 1 = moderate, 2 = severe, 3 = destroyed.
Panel-data MLE (Kalbfleisch & Lawless, 1985) with upper-triangular generator (deterioration only)."""
import sys, json, datetime, collections
import numpy as np, shapefile
from scipy.linalg import expm
from scipy.optimize import minimize
sys.stdout.reconfigure(encoding="utf-8")
shp, out = sys.argv[1], sys.argv[2]
share_damaged = float(sys.argv[3]) if len(sys.argv) > 3 else 0.32   # UNOSAT note: damaged ~32% of structures
MAP = {"No Visible Damage": 0, "Possible Damage": 0, "Moderate Damage": 1, "Severe Damage": 2, "Destroyed": 3}
r = shapefile.Reader(shp, encoding="utf-8")
counts = np.zeros((4, 4), int); per = collections.defaultdict(lambda: np.zeros((4, 4), int))
excluded = collections.Counter()
for rec in r.records():
    if rec["d_SiteID"] in ("Road", "Field"): excluded["road/field"] += 1; continue
    a, b = rec["d_Main_Dam"], rec["d_Main_D_1"]
    if a not in MAP or b not in MAP: excluded["crater/other"] += 1; continue
    d0, d1 = rec["SensorDate"], rec["SensorDa_1"]
    if d1 is None or d1 <= d0: excluded["no re-assessment"] += 1; continue
    i, j = MAP[a], MAP[b]
    if j < i: excluded["decrease (treated as label noise)"] += 1; continue
    dt = (d1 - d0).days
    counts[i, j] += 1; per[dt][i, j] += 1
n_damaged_may = int(counts[:, 1:].sum() + 0)  # damaged in May (moderate+)
# Buildings with 'possible damage' in May are also counted as damaged by UNOSAT; we keep them in state 0.
n_records = int(counts.sum())
total = int(round((n_records) / share_damaged))
extra00 = total - n_records
mean_dt = sum(dt * m.sum() for dt, m in per.items()) / n_records
per[int(round(mean_dt))][0, 0] += extra00
print("excluded:", dict(excluded)); print("records used:", n_records, "| implied total structures:", total, "| added 0->0:", extra00)
print("counts (March->May):\n", counts)
IDX = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
def Q_of(theta):
    Q = np.zeros((4, 4))
    for (i, j), t in zip(IDX, theta): Q[i, j] = np.exp(t)
    np.fill_diagonal(Q, -Q.sum(1)); return Q
def nll(theta):
    Q = Q_of(theta); s = 0.0
    for dt, m in per.items():
        P = expm(Q * dt); P = np.clip(P, 1e-300, 1)
        s -= (m * np.log(P)).sum()
    return s
best = None
for seed in range(8):
    rng = np.random.default_rng(seed)
    res = minimize(nll, rng.normal(-6, 1, 6), method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
    res = minimize(nll, res.x, method="BFGS")
    if best is None or res.fun < best.fun: best = res
Q = Q_of(best.x)
P30, P57 = expm(Q * 30), expm(Q * mean_dt)
emp = counts.astype(float).copy(); emp[0, 0] += extra00; emp = emp / emp.sum(1, keepdims=True)
print("mean interval (days): %.1f" % mean_dt)
np.set_printoptions(precision=5, suppress=True)
print("Q (1/day):\n", Q); print("P(57d) model:\n", P57); print("P empirical:\n", emp); print("P(30d):\n", P30)
json.dump({"states": ["none/possible", "moderate", "severe", "destroyed"], "Q_per_day": Q.tolist(), "mean_interval_days": mean_dt,
           "counts": counts.tolist(), "extra_00": extra00, "implied_total": total, "share_damaged": share_damaged,
           "P_empirical": emp.tolist(), "P_model_mean_interval": P57.tolist(), "excluded": dict(excluded), "nll": best.fun},
          open(out, "w", encoding="utf-8"), indent=2)
