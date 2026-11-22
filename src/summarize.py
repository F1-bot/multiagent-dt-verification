"""Summaries of the replay: table numbers, paired Wilcoxon tests, sensitivity, and data files for Fig. 3.
Usage: python summarize.py <replay_results_dir> <figure_data_dir>
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
R, D = sys.argv[1], sys.argv[2]
d = pd.read_csv(os.path.join(R, "sim_main.csv"))
metrics = ["mean_acc", "mean_acc_changed", "final_acc_real", "mean_brier", "delay_mean", "det7", "delay_mean_hi",
           "det7_hi", "false_alarm", "false_alarm_posted", "phi_bias"]


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


out = {"summary": {}, "paired_vs_P": {}}
for pol, g in d.groupby("policy"):
    out["summary"][pol] = {m: list(ci(g[m])) for m in metrics}
    out["summary"][pol]["conflict"] = list(ci(g["conflict"]))
    out["summary"][pol]["accepted"] = list(ci(g["accepted"]))
    out["summary"][pol]["share_posted"] = list(ci(g["share_posted"]))
a = d[d.policy == "P-full"].sort_values("seed")
for base in sorted(set(d.policy) - {"P-full"}):
    b = d[d.policy == base].sort_values("seed")
    out["paired_vs_P"][base] = {}
    for m in metrics:
        diff = a[m].values - b[m].values
        try:
            pv = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            pv = 1.0
        out["paired_vs_P"][base][m] = [float(diff.mean()), float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))), float(pv)]
# ---- robustness (same seeds)
r = pd.read_csv(os.path.join(R, "sim_robust.csv"))
rob = {"mu": sorted(r.mu.unique())}
for k, v in {"B2-naiveBayes": "B2", "B3-linearPool": "B3", "B4-confTempered": "B4", "P-full": "P"}.items():
    g = r[r.policy == k].groupby("mu")
    rob[v] = g["mean_acc_changed"].mean().values
    rob[v + "ci"] = (1.96 * g["mean_acc_changed"].std(ddof=1) / np.sqrt(g["mean_acc_changed"].count())).values
    rob[v + "det"] = g["det7"].mean().values
pd.DataFrame(rob).to_csv(os.path.join(D, "fig3b_robust.csv"), index=False, sep=" ")
out["robust"] = {k: (list(map(float, v)) if not isinstance(v, list) else v) for k, v in rob.items()}
# paired P vs B3 at each mu
out["robust_P_vs_B3"] = {}
for mu in rob["mu"]:
    pa = r[(r.policy == "P-full") & (r.mu == mu)].sort_values("seed")["mean_acc_changed"].values
    pb = r[(r.policy == "B3-linearPool") & (r.mu == mu)].sort_values("seed")["mean_acc_changed"].values
    out["robust_P_vs_B3"][str(mu)] = [float((pa - pb).mean()), float(wilcoxon(pa - pb).pvalue)]
# ---- gamma sweep
g = pd.read_csv(os.path.join(R, "sim_gamma.csv"))
agg = g.groupby("gamma").agg(det7_hi=("det7_hi", "mean"), det7=("det7", "mean"), ch=("mean_acc_changed", "mean"),
                             fa=("false_alarm", "mean"), dhi=("delay_mean_hi", "mean"),
                             det7_hi_sd=("det7_hi", "std"), det7_sd=("det7", "std"), n=("det7", "count")).reset_index()
agg["det7_hi_ci"] = 1.96 * agg.det7_hi_sd / np.sqrt(agg.n)
agg["det7_ci"] = 1.96 * agg.det7_sd / np.sqrt(agg.n)
agg.to_csv(os.path.join(D, "fig3c_gamma.csv"), index=False, sep=" ")
out["gamma"] = agg.round(5).to_dict(orient="list")
# ---- CDF over all 30 runs
c = pd.read_csv(os.path.join(R, "sim_cdf.csv"))
c.to_csv(os.path.join(D, "fig3_cdf.csv"), index=False, sep=" ")
out["cdf_at"] = {str(x): {k: float(c.loc[c.d == x, k].iloc[0]) for k in c.columns if k != "d"} for x in (7, 14, 40)}
# ---- time series
ts = json.load(open(os.path.join(R, "sim_timeseries.json")))
T = len(ts["P-full"]["acc_t"])
df = pd.DataFrame({"day": np.arange(1, T + 1)})
for k, v in {"B0-periodic": "B0", "B1-VoI": "B1", "B2-naiveBayes": "B2", "B3-linearPool": "B3",
             "B4-confTempered": "B4", "P-full": "P"}.items():
    df["ch_" + v] = ts[k]["accch_t"]; df["acc_" + v] = ts[k]["acc_t"]; df["phi_" + v] = ts[k]["phi_t"]
df.to_csv(os.path.join(D, "fig3_timeseries.csv"), index=False, sep=" ")
out["day59"] = {k: {"acc_changed": ts[k]["accch_t"][-1], "acc": ts[k]["acc_t"][-1], "phi": ts[k]["phi_t"][-1]} for k in ts}
# ---- sensitivity (one at a time around eps=0.02, w*=0.5, W=7)
s = pd.read_csv(os.path.join(R, "sim_sensitivity.csv"))
base = d[d.policy == "P-full"]
bm = {m: base[m].mean() for m in ["mean_acc_changed", "det7", "det7_hi", "false_alarm"]}
sens = s.groupby(["param", "value"])[["mean_acc_changed", "det7", "det7_hi", "false_alarm"]].mean().reset_index()
for m in bm:
    sens["d_" + m] = sens[m] - bm[m]
out["sensitivity"] = sens.round(5).to_dict(orient="records")
out["sensitivity_maxabs"] = {m: float(sens["d_" + m].abs().max()) for m in bm}
json.dump(out, open(os.path.join(R, "numbers.json"), "w"), indent=1)
# ---- print key numbers
S = out["summary"]
for pol in ["B0-periodic", "B1-VoI", "B2-naiveBayes", "B3-linearPool", "B4-confTempered", "P-full", "P-gamma0",
            "P-noQuarantine", "P-noXAI", "P-asymVoI"]:
    print(f"{pol:16s} " + " ".join(f"{m}={S[pol][m][0]:.4f}" for m in ["mean_acc", "mean_acc_changed", "final_acc_real",
          "mean_brier", "det7", "det7_hi", "delay_mean_hi", "false_alarm", "false_alarm_posted"]) +
          f" conflict={S[pol]['conflict'][0]:.1f} accepted={S[pol]['accepted'][0]:.0f}")
print("share_posted", S["P-full"]["share_posted"])
for base in ["B2-naiveBayes", "B3-linearPool", "B4-confTempered", "P-gamma0", "P-noQuarantine", "P-noXAI", "P-asymVoI"]:
    pv = out["paired_vs_P"][base]
    print("P minus", base, {m: (round(pv[m][0], 5), "%.1e" % pv[m][2]) for m in
                            ["mean_acc_changed", "final_acc_real", "mean_brier", "det7", "det7_hi", "delay_mean_hi",
                             "false_alarm", "false_alarm_posted"]})
print("robust:", pd.DataFrame(rob).round(4).to_string())
print("robust P-B3:", out["robust_P_vs_B3"])
print("gamma:", agg[["gamma", "det7_hi", "det7", "ch", "fa", "dhi"]].round(4).to_string())
print("cdf:", out["cdf_at"])
print("day59:", {k: {m: round(v, 4) for m, v in x.items()} for k, x in out["day59"].items() if k in ("P-full", "B0-periodic", "B3-linearPool")})
print("sens maxabs:", out["sensitivity_maxabs"])
print(sens[["param", "value", "d_mean_acc_changed", "d_det7", "d_det7_hi", "d_false_alarm"]].round(4).to_string())
