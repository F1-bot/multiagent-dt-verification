"""Diagnostics quoted in the text: inspection targeting, the illustrative trace (object 575), population composition.
Usage: python -I diagnostics.py <unosat_shp> <crisismmd_features.csv> <results_final_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
import sim_dt

sys.stdout.reconfigure(encoding="utf-8")
shp, feat, R = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(os.path.join(R, "sim_config.json")))
p = dict(cfg["params"]); Q = np.array(cfg["Q_cal"]); O = np.array(cfg["O"])
sim_dt.POLICIES["B3-linearPool"]["kappa0"] = cfg["kappa0"]
df = sim_dt.load_population(shp)
ev = df[~df.calib].reset_index(drop=True)
items = sim_dt.load_items(feat)
P1 = expm(Q)
out = {}
n_rec = int((df.index < (len(df) - 11760)).sum())
out["population"] = {"total": int(len(df)), "records": n_rec, "imputed_undamaged": int(len(df) - n_rec),
                     "eval_total": int(len(ev)),
                     "eval_records": int((~df.calib & (df.index < n_rec)).sum()),
                     "eval_imputed": int((~df.calib & (df.index >= n_rec)).sum())}
out["test_majority_class_acc"] = float(np.bincount(items["y"]).max() / len(items["y"]))
# inspection targeting over 5 runs
targ = {k: {"share_changed": [], "unique_share": []} for k in ["B0-periodic", "B1-VoI", "B3-linearPool", "P-full"]}
for s in range(5):
    world = sim_dt.make_world(ev, Q, items, p, 1000 + s)
    X = world[0]
    changed = X[:, 0] != X[:, -1]
    for name in targ:
        picked = []
        orig = np.argpartition
        def ap(a, k, *args, **kw):
            r = orig(a, k, *args, **kw)
            picked.append(np.asarray(r[:k]).copy())
            return r
        np.argpartition = ap
        try:
            sim_dt.run_policy(sim_dt.POLICIES[name], world, ev, items, P1, O, p)
        finally:
            np.argpartition = orig
        if name == "B0-periodic":
            perm = np.random.default_rng(12345).permutation(len(ev)); C = p["C"]
            idx = np.concatenate([perm[np.arange((t - 1) * C, t * C) % len(ev)] for t in range(1, p["T"] + 1)])
        else:
            idx = np.concatenate(picked)
        u = np.unique(idx)
        targ[name]["share_changed"].append(float(changed[u].mean()))
        targ[name]["unique_share"].append(float(len(u) / len(idx)))
out["inspection_targeting_mean_5runs"] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in targ.items()}
# illustrative object 575 (seed 1000): inspection days per policy
world = sim_dt.make_world(ev, Q, items, p, 1000)
trace = {}
for name in ["P-full", "P-gamma0", "P-noQuarantine", "B3-linearPool", "B2-naiveBayes"]:
    _, _, tr = sim_dt.run_policy(sim_dt.POLICIES[name], world, ev, items, P1, O, p, trace_ids=[575])
    bel = np.array(tr["trace"][575])[:, 2:].sum(1)
    det = next((t for t in range(26, len(bel) + 1) if bel[t - 1] >= 0.5), None)
    trace[name] = {"inspection_days": [e[0] for e in tr["events"][575] if e[1] == "inspection"], "detection_day": det,
                   "post_weights": [round(e[4], 3) for e in tr["events"][575] if e[1] in ("post", "quarantine")]}
out["object_575_seed1000"] = trace
json.dump(out, open(os.path.join(R, "diagnostics.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
