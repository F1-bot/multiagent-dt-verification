"""E1: trace-driven replay of multi-agent DT verification on real data.

Ground truth: UNOSAT multi-temporal building damage assessment (Mariupol, 14 Mar -> 7/8/12 May 2022).
Real initial and final states; intermediate trajectories sampled from the endpoint-conditioned CTMC.
Citizen evidence: real CrisisMMD (test split) image-text posts with real perception outputs.
Usage: python -I sim_dt.py <unosat_shp> <crisismmd_features.csv> <out_dir> [--quick]
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import shapefile
from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import dtcore as dc

sys.stdout.reconfigure(encoding="utf-8")
K = 4
M = np.array([0, 1, 2, 2])                     # DT state -> CrisisMMD class
G_POST = np.array([0.1, 1.0, 1.5, 2.0])        # relative posting propensity by damage state
CRIT_SITES = {"School / University", "Hospital", "Church", "Market", "Industrial Facility"}
MAPD = {"No Visible Damage": 0, "Possible Damage": 0, "Moderate Damage": 1, "Severe Damage": 2, "Destroyed": 3}
CONF = {"Very High": 0.95, "Medium": 0.8, "Uncertain": 0.6}
IDX_Q = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FEAT_FULL = ["conf", "kappa", "agree", "cam_conc", "faith", "sent_intensity", "dup"]
FEAT_NOXAI = ["conf", "kappa", "agree", "sent_intensity", "dup"]


# ------------------------------------------------------------------ data
def load_population(shp, share_damaged=0.32, split_seed=2026):
    r = shapefile.Reader(shp, encoding="utf-8")
    recs = []
    for rec in r.records():
        if rec["d_SiteID"] in ("Road", "Field"):
            continue
        a, b = rec["d_Main_Dam"], rec["d_Main_D_1"]
        if a not in MAPD or b not in MAPD:
            continue
        d0, d1 = rec["SensorDate"], rec["SensorDa_1"]
        if d1 is None or d1 <= d0:
            continue
        i, j = MAPD[a], MAPD[b]
        if j < i:
            continue
        recs.append((i, j, (d1 - d0).days, CONF.get(rec["d_Confiden"], 0.8), rec["d_SiteID"] in CRIT_SITES))
    n_rec = len(recs)
    total = int(round(n_rec / share_damaged))
    recs += [(0, 0, 57, 0.95, False)] * (total - n_rec)
    df = pd.DataFrame(recs, columns=["s0", "s1", "dt", "conf0", "critical"])
    rng = np.random.default_rng(split_seed)
    df["calib"] = False
    for _, g in df.groupby(["s0", "s1"]):
        idx = rng.permutation(g.index.values)
        df.loc[idx[: len(idx) // 2], "calib"] = True
    return df


def Q_from(theta):
    Q = np.zeros((K, K))
    for (i, j), t in zip(IDX_Q, theta):
        Q[i, j] = np.exp(t)
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


def estimate_Q(df):
    groups = {}
    for (s0, s1, dt), n in df.groupby(["s0", "s1", "dt"]).size().items():
        groups.setdefault(dt, np.zeros((K, K)))[s0, s1] += n

    def nll(theta):
        Q = Q_from(theta)
        s = 0.0
        for dt, m in groups.items():
            s -= (m * np.log(np.clip(expm(Q * dt), 1e-300, 1))).sum()
        return s
    best = None
    for seed in range(6):
        x0 = np.random.default_rng(seed).normal(-6, 1, 6)
        res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
        res = minimize(nll, res.x, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    return Q_from(best.x)


def load_items(csv):
    f = pd.read_csv(csv)
    dev, test, train = f[f.split == "dev"], f[f.split == "test"], f[f.split == "train"]
    rho = np.bincount(train.y.values, minlength=3) / len(train)
    rel = {}
    for name, cols in [("r_full", FEAT_FULL), ("r_noxai", FEAT_NOXAI)]:
        sc = StandardScaler().fit(dev[cols].values)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(dev[cols].values), dev.correct.values)
        rel[name] = lr.predict_proba(sc.transform(test[cols].values))[:, 1]
    P = test[["p_fus0", "p_fus1", "p_fus2"]].values
    items = {"p": P, "conf": test.conf.values, "neg": test.sent_negative.values, "y": test.y.values,
             "r_conf": P.max(1), "r_full": rel["r_full"], "r_noxai": rel["r_noxai"], "rho": rho,
             "correct": test.correct.values,
             "pool": [np.where(test.y.values == c)[0] for c in range(3)]}
    return items


# ------------------------------------------------------------------ world
def inspection_model(acc=0.9):
    O = np.zeros((K, K))
    for k in range(K):
        nb = [j for j in (k - 1, k + 1) if 0 <= j < K]
        O[k, k] = acc
        for j in nb:
            O[k, j] = (1 - acc) / len(nb)
    return O


def sample_truth(dfh, Q, T, rng):
    """Endpoint-conditioned CTMC paths by rejection sampling (Hobolth & Stone, 2009)."""
    N = len(dfh)
    X = np.repeat(dfh.s0.values[:, None], T + 1, axis=1)
    rates = -np.diag(Q)
    jump = np.zeros_like(Q)
    for i in range(K):
        if rates[i] > 0:
            jump[i] = Q[i] / rates[i]
            jump[i, i] = 0.0
    s0, s1, dts = dfh.s0.values, dfh.s1.values, dfh.dt.values
    for n in np.where(s1 > s0)[0]:
        while True:
            t, s, jumps = 0.0, s0[n], []
            while rates[s] > 0:
                t += rng.exponential(1.0 / rates[s])
                if t > dts[n]:
                    break
                s = rng.choice(K, p=jump[s])
                jumps.append((t, s))
            if s == s1[n]:
                break
        for tj, sj in jumps:
            X[n, int(np.ceil(tj)):] = sj
    return X


def make_world(dfh, Q, items, p, seed):
    rng = np.random.default_rng(seed)
    T = p["T"]
    X = sample_truth(dfh, Q, T, rng)
    N = len(dfh)
    sal = rng.lognormal(-0.5, 1.0, N)
    lam = p["lam0"] * G_POST[X[:, 1:]] * sal[:, None]
    cnt = rng.poisson(lam)
    b, d = np.nonzero(cnt)
    rep = cnt[b, d]
    b, d = np.repeat(b, rep), np.repeat(d, rep) + 1
    true_c = M[X[b, d]]
    mis = rng.random(len(b)) < p["mu"]
    c = np.where(mis, (true_c + rng.integers(1, 3, len(b))) % 3, true_c)
    item = np.empty(len(c), int)
    for cl in range(3):
        sel = c == cl
        item[sel] = rng.choice(items["pool"][cl], sel.sum())
    o = np.argsort(d, kind="stable")
    posts = {"b": b[o], "d": d[o], "item": item[o], "mis": mis[o]}
    U = rng.random((N, T + 1))
    return X, sal, posts, U


def init_beliefs(s0, conf0):
    Pi = np.zeros((len(s0), K))
    for n, (s, c) in enumerate(zip(s0, conf0)):
        nb = [j for j in (s - 1, s + 1) if 0 <= j < K]
        Pi[n, s] = c
        for j in nb:
            Pi[n, j] = (1 - c) / len(nb)
    return Pi


# ------------------------------------------------------------------ policy run
def run_policy(cfg, world, dfh, items, P1, O, p, trace_ids=None):
    X, sal, posts, U = world
    N, T1 = X.shape
    T = T1 - 1
    Pi = init_beliefs(dfh.s0.values, dfh.conf0.values)
    omega = np.where(dfh.critical.values, 3.0, 1.0)
    C = p["C"]
    LB = np.zeros(N, int)
    negmass = np.zeros(N)
    flagged = np.zeros(N, bool)
    quarantine = {}
    perm = np.random.default_rng(12345).permutation(N)
    crit0 = X[:, 0] < 2
    first = np.where((X >= 2).any(1) & crit0, np.argmax(X >= 2, axis=1), -1)
    det = np.full(N, -1)
    mapst = np.zeros((N, T + 1), np.int8)
    mapst[:, 0] = Pi.argmax(1)
    acc_t, brier_t, phi_t, accch_t = [], [], [], []
    changed = X[:, 0] != X[:, -1]
    fa_num = fa_den = fa_num_p = fa_den_p = 0
    posted = np.bincount(posts["b"], minlength=N) > 0
    day_ptr = np.searchsorted(posts["d"], np.arange(T + 2))
    rho = items["rho"]
    cnt_M = np.bincount(M)[M]
    stats = {"accepted": 0, "quarantined": 0, "released": 0, "mis_accepted": 0, "mis_total": 0, "conflict": 0}
    trace = {int(b): [] for b in (trace_ids if trace_ids is not None else [])}
    events = {int(b): [] for b in trace}
    decay = np.exp(-1.0 / p["tau_sc"])
    for t in range(1, T + 1):
        Pi = dc.aging(Pi, P1)
        if cfg["posts"]:
            for k in range(day_ptr[t], day_ptr[t + 1]):
                b, it, mis = posts["b"][k], posts["item"][k], posts["mis"][k]
                pr = items["p"][it]
                negmass[b] += items["neg"][it]
                stats["mis_total"] += int(mis)
                if cfg["fusion"] == "linear":
                    Pi[b] = dc.linear_pool(Pi[b], dc.normalize(pr[M] / cnt_M), cfg["kappa0"] * items["conf"][it])
                    stats["accepted"] += 1
                    stats["mis_accepted"] += int(mis)
                    if b in events:
                        events[b].append((t, "post", int(np.argmax(pr)), bool(mis), 1.0))
                    continue
                lik = dc.evidence_likelihood(pr, rho, M)
                if cfg.get("robust"):
                    lik = dc.robust_likelihood(lik, p["mu_hat"])
                w = 1.0 if cfg["fusion"] == "bayes" else \
                    p[cfg.get("eta_key", "eta")] * float(dc.reliability_to_weight(items[cfg["rel"]][it], 3))
                cand = dc.tempered_update(Pi[b], lik, w)
                if not cfg["quarantine"]:
                    Pi[b] = cand
                    stats["accepted"] += 1
                    stats["mis_accepted"] += int(mis)
                    if b in events:
                        events[b].append((t, "post", int(np.argmax(pr)), bool(mis), w))
                    continue
                S = float(dc.bayes_surprise(cand, Pi[b]))
                dec = dc.verification_decision(S, w, p["eps"], p["wstar"])
                c_cls = int(np.argmax(pr))
                qlist = [e for e in quarantine.get(b, []) if t - e[0] <= p["W"]]
                corr = [e for e in qlist if e[3] == c_cls and e[4] != it]
                if dec == 2 and not corr:
                    qlist.append((t, lik, w, c_cls, it, mis))
                    quarantine[b] = qlist
                    flagged[b] = True
                    stats["quarantined"] += 1
                    if b in events:
                        events[b].append((t, "quarantine", c_cls, bool(mis), w))
                    continue
                new = cand
                for e in corr:
                    new = dc.tempered_update(new, e[1], e[2])
                    stats["released"] += 1
                    stats["mis_accepted"] += int(e[5])
                qlist = [q for q in qlist if not any(q is e for e in corr)]   # remove by identity
                quarantine[b] = qlist
                if dec == 1:
                    flagged[b] = True
                if LB[b] > 0 and float(new[:LB[b]].sum()) > 0.5:
                    stats["conflict"] += 1          # CONFLICT condition: evidence contradicts a confident inspection
                Pi[b] = dc.project_monotone(new, LB[b])
                stats["accepted"] += 1
                stats["mis_accepted"] += int(mis)
                if b in events:
                    events[b].append((t, "post", c_cls, bool(mis), w))
        negmass *= decay
        if cfg["sched"] == "periodic":
            idx = perm[np.arange((t - 1) * C, t * C) % N]
        else:
            voi = cfg.get("voi", "acc")
            ig = dc.expected_accuracy_gain(Pi, O) if voi == "acc" else \
                dc.expected_risk_reduction(Pi, O, p["Lasym"]) if voi == "asym" else dc.expected_info_gain(Pi, O)
            sc = dc.social_concern(negmass) if cfg["gamma"] > 0 else np.zeros(N)
            fl = flagged if cfg["quarantine"] else np.zeros(N, bool)
            prio = dc.priority(omega, ig, sc, cfg["gamma"], fl, p["delta"])
            idx = np.argpartition(-prio, C)[:C]
        cdf = np.cumsum(O[X[idx, t]], axis=1)
        y = np.minimum((U[idx, t][:, None] > cdf).sum(1), K - 1)
        Pi[idx] = dc.normalize(Pi[idx] * O[:, y].T)
        for b in idx:
            if b in events:
                events[b].append((t, "inspection", int(y[list(idx).index(b)]), False, 1.0))
        if cfg["quarantine"]:
            flagged[idx] = False
            for b in idx:
                quarantine.pop(b, None)
            ok = Pi[idx].max(1) >= 0.95
            LB[idx[ok]] = np.maximum(LB[idx[ok]], Pi[idx[ok]].argmax(1))
        ms = Pi.argmax(1)
        mapst[:, t] = ms
        acc_t.append(float((ms == X[:, t]).mean()))
        accch_t.append(float((ms[changed] == X[changed, t]).mean()))
        brier_t.append(float(((Pi - np.eye(K)[X[:, t]]) ** 2).sum(1).mean()))
        phi_t.append(dc.fidelity_index(Pi))
        cb = Pi[:, 2:].sum(1) >= 0.5
        nc = X[:, t] < 2
        fa_num += int((cb & nc).sum())
        fa_den += int(nc.sum())
        fa_num_p += int((cb & nc & posted).sum())
        fa_den_p += int((nc & posted).sum())
        nd = (det < 0) & (first > 0) & (t >= first) & cb
        det[nd] = t
        for b in trace:
            trace[b].append(Pi[b].tolist())
    # ---- metrics
    has = first > 0
    delay = np.where(det[has] > 0, det[has] - first[has], T + 1 - first[has])
    detected7 = (det[has] > 0) & (det[has] - first[has] <= 7)
    hi = sal >= np.quantile(sal, 0.9)
    dts = np.minimum(dfh.dt.values, T)
    final_real = float((mapst[np.arange(N), dts] == dfh.s1.values).mean())
    res = {"mean_acc": float(np.mean(acc_t)), "final_acc_real": final_real, "mean_brier": float(np.mean(brier_t)),
           "mean_acc_changed": float(np.mean(accch_t)), "final_acc_changed": float(accch_t[-1]),
           "phi_gap": float(np.mean(np.abs(np.array(phi_t) - np.array(acc_t)))),
           "phi_bias": float(np.mean(np.array(phi_t) - np.array(acc_t))),
           "delay_median": float(np.median(delay)), "delay_mean": float(np.mean(delay)),
           "det7": float(detected7.mean()), "n_crit": int(has.sum()),
           "delay_mean_hi": float(np.mean(np.where(det[has & hi] > 0, det[has & hi] - first[has & hi], T + 1 - first[has & hi]))) if (has & hi).any() else np.nan,
           "det7_hi": float(((det[has & hi] > 0) & (det[has & hi] - first[has & hi] <= 7)).mean()) if (has & hi).any() else np.nan,
           "false_alarm": fa_num / max(fa_den, 1), "false_alarm_posted": fa_num_p / max(fa_den_p, 1),
           "share_posted": float(posted.mean()), **stats}
    ts = {"acc_t": acc_t, "phi_t": phi_t, "brier_t": brier_t, "accch_t": accch_t}
    return res, ts, {"trace": trace, "events": events, "truth": {int(b): X[b].tolist() for b in trace},
                     "det": det, "first": first, "hi": hi}


POLICIES = {
    "B0-periodic": dict(posts=False, sched="periodic", gamma=0.0, quarantine=False, fusion=None),
    "B1-VoI": dict(posts=False, sched="voi", gamma=0.0, quarantine=False, fusion=None),
    "B2-naiveBayes": dict(posts=True, sched="voi", gamma=0.0, quarantine=False, fusion="bayes"),
    "B3-linearPool": dict(posts=True, sched="voi", gamma=0.0, quarantine=False, fusion="linear", kappa0=0.1),
    "B4-confTempered": dict(posts=True, sched="voi", gamma=0.0, quarantine=False, fusion="tempered", rel="r_conf",
                            eta_key="eta_b4"),
    "P-full": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_full", robust=True),
    "P-noXAI": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_noxai", robust=True),
    "P-noQuarantine": dict(posts=True, sched="voi", gamma=1.0, quarantine=False, fusion="tempered", rel="r_full",
                           robust=True),
    "P-gamma0": dict(posts=True, sched="voi", gamma=0.0, quarantine=True, fusion="tempered", rel="r_full", robust=True),
    "P-asymVoI": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_full", robust=True,
                      voi="asym"),
}


def tune(cal, Q, items, P1, O, p, n_seeds, t0):
    """All hyper-parameters are selected on the calibration half only (objective: mean DT accuracy)."""
    worlds = [make_world(cal, Q, items, p, 10_000 + s) for s in range(n_seeds)]
    score = lambda cfg, pp: float(np.mean([run_policy(cfg, w, cal, items, P1, O, pp)[0]["mean_acc"] for w in worlds]))
    log = {}
    best = max((score(dict(POLICIES["B3-linearPool"], kappa0=k), p), k) for k in [0.02, 0.05, 0.1, 0.25, 0.5])
    POLICIES["B3-linearPool"]["kappa0"] = best[1]
    log["B3"] = best
    best = max((score(POLICIES["B4-confTempered"], dict(p, eta_b4=e)), e) for e in [0.25, 0.5, 0.75, 1.0])
    p["eta_b4"] = best[1]
    log["B4"] = best
    best = max((score(POLICIES["P-full"], dict(p, mu_hat=m, eta=e)), m, e)
               for m in [0.0] for e in [0.5, 0.75, 1.0])
    p["mu_hat"], p["eta"] = best[1], best[2]
    log["P_stage1"] = best
    best = max((score(POLICIES["P-full"], dict(p, eps=e, wstar=w)), e, w) for e in [0.02, 0.05, 0.15, 0.3] for w in [0.3, 0.5, 0.7])
    p["eps"], p["wstar"] = best[1], best[2]
    log["P_stage2"] = best
    print(f"tuned: {log} ({time.time()-t0:.0f}s)", flush=True)
    return log


def summarize(rows):
    d = pd.DataFrame(rows)
    cols = ["mean_acc", "mean_acc_changed", "final_acc_real", "mean_brier", "phi_bias", "delay_mean", "det7",
            "delay_mean_hi", "det7_hi", "false_alarm", "quarantined", "mis_accepted", "mis_total"]
    print(d.groupby("policy")[cols].mean().round(4).to_string(), flush=True)


def main():
    shp, feat_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
    quick, dev = "--quick" in sys.argv, "--dev" in sys.argv
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = load_population(shp)
    cal, ev = df[df.calib].reset_index(drop=True), df[~df.calib].reset_index(drop=True)
    Q = estimate_Q(cal)
    P1 = expm(Q)
    O = inspection_model(0.9)
    items = load_items(feat_csv)
    p = dict(T=59, lam0=0.025, mu=0.2, C=int(round(0.005 * len(ev))), tau_sc=10.0, W=7, delta=1.0,
             eps=0.05, wstar=0.5, mu_hat=0.2, eta=1.0, eta_b4=1.0)
    Lm = np.ones((K, K)) - np.eye(K)
    Lm[2:, :2] = 5.0                     # asymmetric loss: missing a severe/destroyed state costs five times more
    p["Lasym"] = Lm
    print(f"N_cal={len(cal)} N_eval={len(ev)} C={p['C']} Q=\n{np.round(Q, 5)}", flush=True)
    tlog = tune(cal, Q, items, P1, O, p, 2 if quick else 3, t0)
    pj = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    json.dump({"params": pj, "Q_cal": Q.tolist(), "O": O.tolist(), "kappa0": POLICIES["B3-linearPool"]["kappa0"],
               "tuning": {k: list(v) for k, v in tlog.items()}, "N_eval": len(ev), "N_cal": len(cal),
               "rho": items["rho"].tolist()}, open(os.path.join(out, "sim_config.json"), "w"), indent=2)
    if dev:   # development check on the calibration half only (fresh seeds); the evaluation half is untouched
        rows = []
        for s in range(6):
            world = make_world(cal, Q, items, p, 20_000 + s)
            for name, cfg in POLICIES.items():
                res = run_policy(cfg, world, cal, items, P1, O, p)[0]
                res.update(policy=name, seed=s)
                rows.append(res)
        summarize(rows)
        pd.DataFrame(rows).to_csv(os.path.join(out, "sim_dev_calibration_half.csv"), index=False)
        print(f"DEV DONE {time.time()-t0:.0f}s")
        return
    # ---- main scenario (evaluation half, fresh seeds)
    n_main = 0 if "--sweeps-only" in sys.argv else (5 if quick else 30)
    rows, tsacc = [], {}
    trace_dump = None
    cdf_acc = {}
    for s in range(n_main):
        world = make_world(ev, Q, items, p, 1000 + s)
        tr_ids = None
        if s == 0:
            X, sal, posts, U = world
            nposts = np.bincount(posts["b"], minlength=len(ev))
            nmis = np.bincount(posts["b"], weights=posts["mis"].astype(float), minlength=len(ev))
            cand = np.where((X[:, 0] < 2) & (X[:, -1] >= 2) & (nposts >= 3) & (nmis >= 1))[0]
            tr_ids = cand[:40]
        for name, cfg in POLICIES.items():
            res, ts, tr = run_policy(cfg, world, ev, items, P1, O, p, trace_ids=tr_ids)
            res.update(policy=name, seed=s)
            rows.append(res)
            tsacc.setdefault(name, []).append(ts)
            if s == 0 and name in ("P-full", "B2-naiveBayes", "B3-linearPool"):
                trace_dump = trace_dump or {}
                trace_dump[name] = {k: v for k, v in tr.items() if k in ("trace", "events", "truth")}
            if name in ("P-full", "B3-linearPool", "B2-naiveBayes", "B0-periodic"):
                det, first, hi = tr["det"], tr["first"], tr["hi"]
                has = first > 0
                dl = np.where(det[has] > 0, det[has] - first[has], 10 ** 6)
                acc_ = cdf_acc.setdefault(name, {"all": [], "hi": []})
                acc_["all"].append(dl)
                acc_["hi"].append(dl[hi[has]])
        print(f"  main seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    if n_main > 0:
        grid = np.arange(0, 41)
        cdf = {"d": grid}
        for name, tag in [("B0-periodic", "B0"), ("B2-naiveBayes", "B2"), ("B3-linearPool", "B3"), ("P-full", "P")]:
            a, h = np.concatenate(cdf_acc[name]["all"]), np.concatenate(cdf_acc[name]["hi"])
            cdf[tag] = [(a <= x).mean() for x in grid]
            cdf[tag + "hi"] = [(h <= x).mean() for x in grid]
        pd.DataFrame(cdf).to_csv(os.path.join(out, "sim_cdf.csv"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(out, "sim_main.csv"), index=False)
        summarize(rows)
        agg = {k: {m: np.mean([d[m] for d in v], axis=0).tolist() for m in ("acc_t", "phi_t", "brier_t", "accch_t")}
               for k, v in tsacc.items()}
        json.dump(agg, open(os.path.join(out, "sim_timeseries.json"), "w"))
        json.dump(trace_dump, open(os.path.join(out, "sim_trace.json"), "w"))
    # ---- robustness to misleading evidence (same 30 seeds as the main scenario: common random numbers)
    n_sw = 3 if quick else 30
    rows = []
    for mu in [0.0, 0.1, 0.2, 0.3, 0.4]:
        pm = dict(p, mu=mu)
        for s in range(n_sw):
            world = make_world(ev, Q, items, pm, 1000 + s)
            for name in ["B2-naiveBayes", "B3-linearPool", "B4-confTempered", "P-full"]:
                res = run_policy(POLICIES[name], world, ev, items, P1, O, pm)[0]
                res.update(policy=name, seed=s, mu=mu)
                rows.append(res)
        print(f"  robustness mu={mu} done ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "sim_robust.csv"), index=False)
    # ---- social weighting sweep and one-at-a-time sensitivity of the verifier thresholds (same 30 seeds)
    rows, srows = [], []
    sens = [("eps", v) for v in [0.005, 0.01, 0.05, 0.1, 0.2]] + [("wstar", v) for v in [0.3, 0.4, 0.6, 0.7]] + \
           [("W", v) for v in [3, 14, 28]]
    for s in range(n_sw):
        world = make_world(ev, Q, items, p, 1000 + s)
        for g in [0.0, 0.5, 1.0, 2.0, 4.0]:
            res = run_policy(dict(POLICIES["P-full"], gamma=g), world, ev, items, P1, O, p)[0]
            res.update(gamma=g, seed=s)
            rows.append(res)
        for par, val in sens:
            res = run_policy(POLICIES["P-full"], world, ev, items, P1, O, dict(p, **{par: val}))[0]
            res.update(param=par, value=val, seed=s)
            srows.append(res)
        print(f"  gamma/sensitivity seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "sim_gamma.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(out, "sim_sensitivity.csv"), index=False)
    print(f"ALL DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
