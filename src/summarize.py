"""Summaries of the multi-source replay: table numbers, paired tests, sweeps and data files for Fig. 3.
Usage: python src/summarize.py <replay_results_dir> <figure_data_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
R, D = sys.argv[1], sys.argv[2]
os.makedirs(D, exist_ok=True)
M = ["auroc_mean", "ap_mean", "auroc_end", "ap_end", "precision_end", "recall_end", "f1_end", "final_acc_real",
     "mean_acc", "mean_acc_changed", "mean_brier", "det7", "det7_hi", "det14", "det14_hi", "det21", "det21_hi",
     "delay_median", "delay_median_hi", "delay_mean_hi", "insp_yield_changed", "insp_yield_corrective",
     "false_alarm", "false_alarm_posted", "conflict", "accepted", "quarantined", "phi_bias", "postcov7",
     "det7_given_postcov", "cov7", "det7_given_cov", "share_posted"]
KEY = ["auroc_mean", "auroc_end", "precision_end", "recall_end", "f1_end", "final_acc_real", "mean_acc_changed",
       "mean_brier", "det14", "det14_hi", "det21_hi", "delay_median_hi", "false_alarm", "insp_yield_corrective"]


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))] if len(x) > 1 else [float("nan")] * 2


def paired(a, b, m):
    a, b = a.sort_values("seed")[m].values, b.sort_values("seed")[m].values
    d = a - b
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return [float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), p]


out = {}
d = pd.read_csv(os.path.join(R, "sim_main.csv"))
out["summary"] = {pol: {m: ci(g[m]) for m in M if m in g} for pol, g in d.groupby("policy", sort=False)}
a = d[d.policy == "P-full"]
out["paired_vs_P"] = {pol: {m: paired(a, d[d.policy == pol], m) for m in KEY}
                      for pol in d.policy.unique() if pol != "P-full"}
# ---- sweeps over radar settings, misleading posts, social weight and thresholds
for exp in ["sar_auc", "sar_rho", "sar_period", "mu", "gamma"]:
    s = pd.read_csv(os.path.join(R, f"sim_{exp}.csv"))
    out[exp] = {}
    for (v, pol), g in s.groupby(["value", "policy"], sort=False):
        out[exp].setdefault(str(v), {})[pol] = {m: ci(g[m]) for m in KEY if m in g}
    if exp != "gamma":
        out[exp + "_P_vs"] = {}
        for v, gv in s.groupby("value"):
            pa = gv[gv.policy == "P-full"]
            out[exp + "_P_vs"][str(v)] = {pol: {m: paired(pa, gv[gv.policy == pol], m)
                                               for m in ["auroc_end", "precision_end", "f1_end", "mean_acc_changed"]}
                                          for pol in gv.policy.unique() if pol != "P-full"}
sens = pd.read_csv(os.path.join(R, "sim_sens.csv"))
base = a
dev = {}
for (par, v), g in sens.groupby(["param", "value"]):
    dev[f"{par}={v}"] = {m: float(g[m].mean() - base[m].mean()) for m in ["auroc_end", "precision_end", "f1_end", "det14_hi", "mean_acc_changed"]}
out["sensitivity"] = dev
out["sensitivity_maxabs"] = {m: max(abs(x[m]) for k, x in dev.items() if not k.startswith("eta_sar"))
                             for m in ["auroc_end", "precision_end", "f1_end", "det14_hi", "mean_acc_changed"]}
out["config"] = json.load(open(os.path.join(R, "sim_config.json")))
out["sar_weights"] = json.load(open(os.path.join(R, "sim_sar_weights.json")))
json.dump(out, open(os.path.join(R, "numbers.json"), "w"), indent=1)

# ---- figure data
ts = json.load(open(os.path.join(R, "sim_timeseries.json")))
T = len(ts["P-full"]["auc_t"])
f = pd.DataFrame({"day": np.arange(1, T + 1)})
for pol, tag in [("B0-periodic", "B0"), ("B1-SAR", "B1R"), ("B2-naiveBayes", "B2"), ("B3-linearPool", "B3"),
                 ("B4-confTempered", "B4"), ("P-full", "P"), ("P-postsOnly", "Pposts")]:
    f["auc_" + tag] = ts[pol]["auc_t"]
    f["ap_" + tag] = ts[pol]["ap_t"]
    f["ch_" + tag] = ts[pol]["accch_t"]
f.round(5).to_csv(os.path.join(D, "fig3_timeseries.csv"), index=False, sep=" ")
c = pd.read_csv(os.path.join(R, "sim_cdf.csv"))
cc = pd.DataFrame({"d": c["d"]})
for pol, tag in [("B0-periodic", "B0"), ("B1-SAR", "B1R"), ("B2-naiveBayes", "B2"), ("B3-linearPool", "B3"),
                 ("P-full", "P"), ("P-postsOnly", "Pposts")]:
    cc[tag] = c[pol]
    cc[tag + "hi"] = c[pol + "|hi"]
cc.round(5).to_csv(os.path.join(D, "fig3_cdf.csv"), index=False, sep=" ")
for exp, metric, fname in [("sar_auc", "auroc_end", "fig3c_sar_auc.csv"), ("sar_auc", "f1_end", "fig3c_sar_f1.csv"), ("sar_rho", "precision_end", "fig3d_sar_rho.csv")]:
    s = pd.read_csv(os.path.join(R, f"sim_{exp}.csv"))
    g = s.groupby(["value", "policy"])[metric]
    tab = g.mean().unstack()
    err = (1.96 * g.std(ddof=1) / np.sqrt(g.count())).unstack()
    o = pd.DataFrame({"x": tab.index})
    for pol, tag in [("B1-SAR", "B1R"), ("B2-naiveBayes", "B2"), ("B3-linearPool", "B3"), ("P-full", "P"),
                     ("P-sarUntempered", "Pnaive")]:
        if pol in tab:
            o[tag] = tab[pol].values
            o[tag + "ci"] = err[pol].values
    o.round(5).to_csv(os.path.join(D, fname), index=False, sep=" ")

# ---- print key numbers
S = out["summary"]
cols = ["auroc_mean", "auroc_end", "precision_end", "recall_end", "f1_end", "final_acc_real", "mean_acc_changed",
        "det14", "det14_hi", "det21_hi", "delay_median_hi", "insp_yield_changed", "insp_yield_corrective", "false_alarm"]
print(pd.DataFrame({pol: {m: S[pol][m][0] for m in cols} for pol in S}).T.round(4).to_string())
print("\nmax CI half-widths:", {m: round(max(S[pol][m][1] for pol in S if not np.isnan(S[pol][m][1])), 4) for m in cols})
print("\nP-full minus:")
for pol, v in out["paired_vs_P"].items():
    print(f"  {pol:16s}", {m: (round(x[0], 4), f"{x[2]:.0e}") for m, x in v.items()
                           if m in ("auroc_end", "precision_end", "f1_end", "final_acc_real", "det14_hi", "false_alarm")})
print("\nsensitivity max |dev| (excluding eta_sar):", {k: round(v, 4) for k, v in out["sensitivity_maxabs"].items()})
print("eta_sar devs:", {k: {m: round(x, 4) for m, x in v.items()} for k, v in dev.items() if k.startswith("eta_sar")})
print("\nmu sweep (mean_acc_changed / f1):")
mu = pd.read_csv(os.path.join(R, "sim_mu.csv"))
print(mu.groupby(["value", "policy"])[["mean_acc_changed", "f1_end", "precision_end", "det14_hi"]].mean().round(4).unstack().to_string())
print("\ngamma sweep:")
gm = pd.read_csv(os.path.join(R, "sim_gamma.csv"))
print(gm.groupby("value")[["det14_hi", "det21_hi", "delay_median_hi", "f1_end", "mean_acc_changed", "false_alarm"]].mean().round(4).to_string())
