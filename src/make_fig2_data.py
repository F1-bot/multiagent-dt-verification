"""Data files for Fig. 2: UNOSAT transition matrix (heatmap cells) and the trace of one illustrative object.

Usage: python src/make_fig2_data.py [--shp <UNOSAT damage .shp>] [--replay results/replay]
       [--calibration results/calibration] [--out figures/data] [--object 575]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.linalg import expm
import sim_dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92)
DEFAULT_SHP = os.path.join(ROOT, "data", "raw", "unosat", "extracted", "UNOSAT_Mariupol_Livoberezhyny_DA_12May2022.shp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", default=DEFAULT_SHP)
    ap.add_argument("--replay", default=os.path.join(ROOT, "results", "replay"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "results", "calibration"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "data"))
    ap.add_argument("--object", type=int, default=575, help="index of the traced object (seed 1000 of the replay)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- (a) empirical 57-day transitions (all assessed objects + imputed undamaged) vs fitted CTMC
    un = json.load(open(os.path.join(a.calibration, "unosat_Q.json")))
    cfg = json.load(open(os.path.join(a.replay, "sim_config.json")))
    cnt = np.array(un["counts"], float)
    cnt[0, 0] += un["extra_00"]
    emp = cnt / cnt.sum(1, keepdims=True)
    mod = expm(np.array(cfg["Q_cal"]) * un["mean_interval_days"])
    rows = [{"x": j, "y": i, "emp": emp[i, j], "mod": mod[i, j], "n": int(cnt[i, j])} for i in range(4) for j in range(4)]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "fig2a_transition.csv"), index=False, sep=" ")
    cs = 0.82
    L = []
    for r in rows:
        i, j, e = r["y"], r["x"], r["emp"]
        if j < i:
            continue
        x0, y0 = j * cs, -(i + 1) * cs
        pct = int(round(100 * np.sqrt(e)))
        L.append(BS + "fill[dtL!%d!seqLo] (%.3f,%.3f) rectangle ++(%.3f,%.3f);" % (pct, x0, y0, cs, cs))
        L.append(BS + "node[text=%s, font=%sfootnotesize, inner sep=0pt] at (%.3f,%.3f) {%.2f};"
                 % ("white" if pct > 55 else "ink", BS, x0 + cs / 2, y0 + cs / 2, e))
    for k in range(5):
        L.append(BS + "draw[white, line width=1.5pt] (%.3f,0) -- (%.3f,%.3f);" % (k * cs, k * cs, -4 * cs))
        L.append(BS + "draw[white, line width=1.5pt] (0,%.3f) -- (%.3f,%.3f);" % (-k * cs, 4 * cs, -k * cs))
    for i in range(4):
        L.append(BS + "node[anchor=east, font=%sfootnotesize, text=ink, inner sep=1.5pt] at (0,%.3f) {%d};" % (BS, -(i + 0.5) * cs, i))
        L.append(BS + "node[anchor=south, font=%sfootnotesize, text=ink, inner sep=1.5pt] at (%.3f,0) {%d};" % (BS, (i + 0.5) * cs, i))
    open(os.path.join(a.out, "fig2a_cells.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---- (b-d) trace of one object: expected state +- 2 sd, posts, inspections
    tr = json.load(open(os.path.join(a.replay, "sim_trace.json")))
    b = str(a.object)
    df = sim_dt.load_population(a.shp)
    ev = df[~df.calib].reset_index(drop=True)
    pi0 = sim_dt.init_beliefs(ev.s0.values[[a.object]], ev.conf0.values[[a.object]])[0]
    k = np.arange(4)
    out = {"day": np.arange(0, 60)}
    for pol, tag in [("P-full", "P"), ("B3-linearPool", "B")]:
        bel = np.vstack([pi0, np.array(tr[pol]["trace"][b])])
        m = bel @ k
        sd = np.sqrt(np.maximum(bel @ (k ** 2) - m ** 2, 0))
        out["m" + tag], out["lo" + tag], out["hi" + tag] = m, np.clip(m - 2 * sd, 0, 3), np.clip(m + 2 * sd, 0, 3)
    out["truth"] = np.array(tr["P-full"]["truth"][b])
    pd.DataFrame(out).round(4).to_csv(os.path.join(a.out, "fig2_trace.csv"), sep=" ", index=False)
    posts = [(e[0], e[2], e[4]) for e in tr["P-full"]["events"][b] if e[1] in ("post", "quarantine")]
    pd.DataFrame(posts, columns=["day", "cls", "w"]).to_csv(os.path.join(a.out, "fig2_posts.csv"), sep=" ", index=False)
    ins = [(e[0], 1) for e in tr["P-full"]["events"][b] if e[1] == "inspection"] + \
          [(e[0], 0) for e in tr["B3-linearPool"]["events"][b] if e[1] == "inspection"]
    pd.DataFrame(ins, columns=["day", "row"]).to_csv(os.path.join(a.out, "fig2_insp.csv"), sep=" ", index=False)
    print("figure 2 data written to", a.out)


if __name__ == "__main__":
    main()
