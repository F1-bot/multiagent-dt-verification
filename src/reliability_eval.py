"""E2: XAI-grounded reliability of multimodal evidence (CrisisMMD damage task).
Reliability models are fitted on the dev split and evaluated on the held-out test split.
Usage: python -I reliability_eval.py <crisismmd_features.csv> <out_dir>
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

sys.stdout.reconfigure(encoding="utf-8")
f = pd.read_csv(sys.argv[1])
out = sys.argv[2]
dev, test = f[f.split == "dev"].reset_index(drop=True), f[f.split == "test"].reset_index(drop=True)
y_dev, y_te = dev.correct.values, test.correct.values
SETS = {
    "S0: max-probability": None,
    "S1: + cross-modal (kappa, agreement)": ["conf", "kappa", "agree"],
    "S2: + XAI (Grad-CAM concentration, deletion faithfulness)": ["conf", "kappa", "agree", "cam_conc", "faith"],
    "S3: + provenance (sentiment intensity, duplicate)": ["conf", "kappa", "agree", "cam_conc", "faith", "sent_intensity", "dup"],
}
rng = np.random.default_rng(0)
B = 2000
boot = [rng.integers(0, len(test), len(test)) for _ in range(B)]


def ece(y, r, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) if hi < 1 else (r >= lo) & (r <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - r[m].mean())
    return e


res, scores, coefs, curves = {}, {}, {}, {}
for name, cols in SETS.items():
    if cols is None:
        r = test[["p_fus0", "p_fus1", "p_fus2"]].values.max(1)
    else:
        sc = StandardScaler().fit(dev[cols].values)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(dev[cols].values), y_dev)
        r = lr.predict_proba(sc.transform(test[cols].values))[:, 1]
        coefs[name] = dict(zip(cols, lr.coef_[0].round(3).tolist()))
    scores[name] = r
    aucs = np.array([roc_auc_score(y_te[i], r[i]) for i in boot])
    fpr, tpr, _ = roc_curve(y_te, r)
    curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    res[name] = {"auroc": roc_auc_score(y_te, r), "auroc_ci": np.percentile(aucs, [2.5, 97.5]).tolist(),
                 "brier": brier_score_loss(y_te, r), "ece": ece(y_te, r)}
    print(f"{name:62s} AUROC={res[name]['auroc']:.3f} [{res[name]['auroc_ci'][0]:.3f},{res[name]['auroc_ci'][1]:.3f}] "
          f"Brier={res[name]['brier']:.3f} ECE={res[name]['ece']:.3f}")
base = scores["S0: max-probability"]
for name in list(SETS)[1:]:
    d = np.array([roc_auc_score(y_te[i], scores[name][i]) - roc_auc_score(y_te[i], base[i]) for i in boot])
    res[name]["delta_auroc_vs_S0"] = float(d.mean())
    res[name]["delta_auroc_ci"] = np.percentile(d, [2.5, 97.5]).tolist()
    res[name]["p_delta_le_0"] = float((d <= 0).mean())
    print(f"   dAUROC vs S0 = {d.mean():+.3f} 95%CI [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]  P(d<=0)={(d <= 0).mean():.3f}")
d23 = np.array([roc_auc_score(y_te[i], scores[list(SETS)[2]][i]) - roc_auc_score(y_te[i], scores[list(SETS)[1]][i]) for i in boot])
res["xai_increment_S2_vs_S1"] = {"mean": float(d23.mean()), "ci": np.percentile(d23, [2.5, 97.5]).tolist(), "p_le_0": float((d23 <= 0).mean())}
print(f"XAI increment (S2 - S1): {d23.mean():+.3f} [{np.percentile(d23, 2.5):+.3f},{np.percentile(d23, 97.5):+.3f}] P<=0 {(d23 <= 0).mean():.3f}")
# feature-level descriptive statistics (correct vs incorrect)
desc = test.groupby("correct")[["conf", "kappa", "agree", "cam_conc", "faith", "sent_intensity", "dup"]].mean().round(4)
print(desc)
# reliability diagram data for S0 and S3
rel = {}
for name in [list(SETS)[0], list(SETS)[3]]:
    r = scores[name]
    edges = np.linspace(0, 1, 11)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) if hi < 1 else (r >= lo) & (r <= hi)
        if m.sum() >= 5:
            pts.append([float(r[m].mean()), float(y_te[m].mean()), int(m.sum())])
    rel[name] = pts
json.dump({"results": res, "coefs": coefs, "roc": curves, "reliability_diagram": rel,
           "feature_means_by_correct": desc.reset_index().to_dict(orient="list"),
           "n_dev": int(len(dev)), "n_test": int(len(test)), "test_accuracy": float(y_te.mean())},
          open(os.path.join(out, "reliability_eval.json"), "w"), indent=1)
