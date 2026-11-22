"""E1: trace-driven replay of multi-agent DT verification on real data.

Ground truth: UNOSAT multi-temporal building damage assessment (Mariupol, 14 Mar -> 7/8/12 May 2022).
Real initial and final states; intermediate trajectories sampled from the endpoint-conditioned CTMC.
Citizen evidence: real CrisisMMD (test split) image-text posts with real perception outputs.
Satellite evidence: Sentinel-1 change scores simulated with the per-acquisition accuracy reported for Ukraine
(AUC 0.81, Dietrich et al. 2025), a 12-day revisit (Sentinel-1A only in 2022) and a persistent object-specific error.
Usage: python src/sim_dt.py <unosat_shp> <crisismmd_features.csv> <out_dir> [--quick]
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
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import dtcore as dc

sys.stdout.reconfigure(encoding="utf-8")
K = 4
M = np.array([0, 1, 2, 2])                     # DT state -> CrisisMMD class
L_CRIT = np.array([[float((k >= 2) != (a >= 2)) for a in range(4)] for k in range(4)])   # loss of the critical/non-critical call
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
    sar_on = bool(cfg.get("use_sar")) and p.get("sar_period", 0) > 0
    if sar_on:
        sar_rng = np.random.default_rng(int(U[0, 0] * 1e9) + 7)    # identical draws for every policy on a world
        d_sar = float(np.sqrt(2.0) * norm.ppf(p["sar_auc"]))       # per-acquisition AUC -> separation d'
        a_sar = np.where(np.arange(K) >= 2, d_sar, 0.0)            # mean change score by damage state
        sar_bias = sar_rng.standard_normal(N)                       # persistent object-specific error
    pool = X[:, 0] < 2                                              # objects that can become critical
    auc_t, ap_t = [], []
    insp_total = insp_changed = insp_corrective = 0
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
        if sar_on and t % p["sar_period"] == 0:
            # change score of every object: s = a[x] + sqrt(rho) * persistent error + sqrt(1 - rho) * new error
            s = a_sar[X[:, t]] + np.sqrt(p["sar_rho"]) * sar_bias + np.sqrt(1.0 - p["sar_rho"]) * sar_rng.standard_normal(N)
            lr = np.exp(np.clip(d_sar * s - d_sar ** 2 / 2, -30.0, 30.0))     # calibrated per-acquisition likelihood ratio
            lik = np.where(np.arange(K)[None, :] >= 2, lr[:, None], 1.0)
            ps = lr / (1.0 + lr)
            if cfg["fusion"] == "linear":                                   # opinion pooling, as for the posts
                q = np.stack([(1 - ps) / 2, (1 - ps) / 2, ps / 2, ps / 2], axis=1)
                a = (p["kappa_sar"] * np.abs(2 * ps - 1))[:, None]
                Pi = dc.normalize((1.0 - a) * Pi + a * q)
                w = a[:, 0]
            else:
                if cfg["fusion"] == "bayes" or cfg.get("sar_untempered"):
                    w = np.ones(N)
                elif cfg.get("rel") == "r_conf":                            # confidence tempering (B4)
                    w = p["eta_b4_sar"] * np.abs(2 * ps - 1)
                else:                                                       # source-level reliability weight
                    w = np.full(N, p["eta_sar"])
                cand = dc.normalize(Pi * lik ** w[:, None])
                if cfg["quarantine"]:                                       # verifier: surprise flag + invariant
                    flagged |= dc.bayes_surprise(cand, Pi) > p["eps"]
                    below = np.arange(K)[None, :] < LB[:, None]
                    stats["conflict"] += int(((cand * below).sum(1) > 0.5).sum())
                    proj = cand * ~below
                    ok = proj.sum(1) > 1e-12
                    cand[ok] = dc.normalize(proj[ok])
                Pi = cand
            for b in trace:
                events[b].append((t, "sar", float(s[b]), False, float(w[b])))
        if cfg["sched"] == "periodic":
            idx = perm[np.arange((t - 1) * C, t * C) % N]
        else:
            voi = cfg.get("voi", "acc")
            ig = dc.expected_accuracy_gain(Pi, O) if voi == "acc" else \
                dc.expected_risk_reduction(Pi, O, p["Lasym"]) if voi == "asym" else \
                dc.expected_risk_reduction(Pi, O, L_CRIT) if voi == "crit" else dc.expected_info_gain(Pi, O)
            sc = dc.social_concern(negmass) if cfg["gamma"] > 0 else np.zeros(N)
            fl = flagged if cfg["quarantine"] else np.zeros(N, bool)
            prio = dc.priority(omega, ig, sc, cfg["gamma"], fl, p["delta"])
            if cfg["sched"] == "oracle":       # upper bound: the planner knows which objects are critical but undetected
                missed = (X[:, t] >= 2) & (Pi[:, 2:].sum(1) < 0.5)
                prio = prio + 10.0 * missed + 1.0 * (Pi.argmax(1) != X[:, t])
            idx = np.argpartition(-prio, C)[:C]
        insp_total += len(idx)
        insp_changed += int((X[idx, t] != X[idx, 0]).sum())
        insp_corrective += int((Pi[idx].argmax(1) != X[idx, t]).sum())
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
        yt = X[pool, t] >= 2
        if 0 < yt.sum() < yt.size:
            rt = Pi[pool, 2:].sum(1)
            auc_t.append(float(roc_auc_score(yt, rt)))
            ap_t.append(float(average_precision_score(yt, rt)))
        else:
            auc_t.append(np.nan)
            ap_t.append(np.nan)
        for b in trace:
            trace[b].append(Pi[b].tolist())
    # ---- metrics
    has = first > 0
    delay = np.where(det[has] > 0, det[has] - first[has], T + 1 - first[has])
    full7 = has & (first <= T - 7)                       # transitions with a complete seven-day window
    detected7 = (det[full7] > 0) & (det[full7] - first[full7] <= 7)
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
           "det7_hi": float(((det[full7 & hi] > 0) & (det[full7 & hi] - first[full7 & hi] <= 7)).mean()) if (full7 & hi).any() else np.nan,
           "false_alarm": fa_num / max(fa_den, 1), "false_alarm_posted": fa_num_p / max(fa_den_p, 1),
           "share_posted": float(posted.mean()), **stats}
    risk = Pi[:, 2:].sum(1)
    yT = X[:, T] >= 2
    flagT = pool & (risk >= 0.5)
    tp = int((flagT & yT).sum())
    prec = tp / max(int(flagT.sum()), 1)
    rec = tp / max(int((pool & yT).sum()), 1)
    res.update({"auroc_mean": float(np.nanmean(auc_t)), "ap_mean": float(np.nanmean(ap_t)),
                "auroc_end": float(roc_auc_score(yT[pool], risk[pool])),
                "ap_end": float(average_precision_score(yT[pool], risk[pool])),
                "precision_end": float(prec), "recall_end": float(rec),
                "f1_end": float(2 * prec * rec / max(prec + rec, 1e-12)),
                "recall_ever": float((det[has] > 0).mean()),
                "delay_median_hi": float(np.median(np.where(det[has & hi] > 0, det[has & hi] - first[has & hi], T + 1 - first[has & hi]))) if (has & hi).any() else np.nan,
                "insp_yield_changed": insp_changed / max(insp_total, 1),
                "insp_yield_corrective": insp_corrective / max(insp_total, 1)})
    # coverage of critical transitions by the evidence the policy uses, and detection given coverage
    cov_p = np.zeros(N, bool)
    if cfg["posts"]:
        pb, pd_ = posts["b"], posts["d"]
        okp = (first[pb] > 0) & (pd_ >= first[pb]) & (pd_ <= first[pb] + 7)
        cov_p[pb[okp]] = True
    cov_s = np.zeros(N, bool)
    if sar_on:
        nxt = np.ceil(np.maximum(first, 1) / p["sar_period"]) * p["sar_period"]
        cov_s = (first > 0) & (nxt <= np.minimum(first + 7, T))
    cov = (cov_p | cov_s)[full7]
    cp = cov_p[full7]
    d7 = (det[full7] > 0) & (det[full7] - first[full7] <= 7)
    for dd in (14, 21):
        fw = has & (first <= T - dd)
        okd = (det > 0) & (det - first <= dd)
        res[f"det{dd}"] = float(okd[fw].mean()) if fw.any() else np.nan
        res[f"det{dd}_hi"] = float(okd[fw & hi].mean()) if (fw & hi).any() else np.nan
    res.update({"cov7": float(cov.mean()), "det7_given_cov": float(d7[cov].mean()) if cov.any() else np.nan,
                "postcov7": float(cp.mean()), "det7_given_postcov": float(d7[cp].mean()) if cp.any() else np.nan})
    ts = {"acc_t": acc_t, "phi_t": phi_t, "brier_t": brier_t, "accch_t": accch_t, "auc_t": auc_t, "ap_t": ap_t}
    return res, ts, {"trace": trace, "events": events, "truth": {int(b): X[b].tolist() for b in trace},
                     "det": det, "first": first, "hi": hi, "Pi_final": Pi}


POLICIES = {
    "B0-periodic": dict(posts=False, sched="periodic", gamma=0.0, quarantine=False, fusion=None),
    "B1-VoI": dict(posts=False, sched="voi", gamma=0.0, quarantine=False, fusion=None),
    "B1-SAR": dict(posts=False, sched="voi", gamma=0.0, quarantine=False, fusion="bayes", use_sar=True),
    "B2-naiveBayes": dict(posts=True, sched="voi", gamma=0.0, quarantine=False, fusion="bayes", use_sar=True),
    "B3-linearPool": dict(posts=True, sched="voi", gamma=0.0, quarantine=False, fusion="linear", kappa0=0.1,
                          use_sar=True),
    "B4-confTempered": dict(posts=True, sched="voi", gamma=0.0, quarantine=False, fusion="tempered", rel="r_conf",
                            eta_key="eta_b4", use_sar=True),
    "P-full": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_full", robust=True,
                   use_sar=True),
    "P-postsOnly": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_full",
                        robust=True, use_sar=False),
    "P-sarUntempered": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_full",
                            robust=True, use_sar=True, sar_untempered=True),
    "P-noXAI": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_noxai", robust=True,
                    use_sar=True),
    "P-noQuarantine": dict(posts=True, sched="voi", gamma=1.0, quarantine=False, fusion="tempered", rel="r_full",
                           robust=True, use_sar=True),
    "P-gamma0": dict(posts=True, sched="voi", gamma=0.0, quarantine=True, fusion="tempered", rel="r_full", robust=True,
                     use_sar=True),
    "P-asymVoI": dict(posts=True, sched="voi", gamma=1.0, quarantine=True, fusion="tempered", rel="r_full", robust=True,
                      voi="asym", use_sar=True),
}
MAIN = ["B0-periodic", "B1-VoI", "B1-SAR", "B2-naiveBayes", "B3-linearPool", "B4-confTempered", "P-full",
        "P-postsOnly", "P-sarUntempered", "P-gamma0", "P-noQuarantine", "P-noXAI", "P-asymVoI"]
KEEP = {"B0-periodic", "B1-SAR", "B2-naiveBayes", "B3-linearPool", "B4-confTempered", "P-full", "P-postsOnly"}

# ------------------------------------------------------------------ parallel execution
_G = {}


def _init_worker(shp, feat_csv, Q, O):
    df = load_population(shp)
    _G["cal"] = df[df.calib].reset_index(drop=True)
    _G["ev"] = df[~df.calib].reset_index(drop=True)
    _G["items"] = load_items(feat_csv)
    _G["Q"] = np.asarray(Q)
    _G["P1"] = expm(_G["Q"])
    _G["O"] = np.asarray(O)


def _run_task(task):
    """One world (half, seed, world parameters) and a list of policy runs on it."""
    dfh = _G[task["half"]]
    world = make_world(dfh, _G["Q"], _G["items"], task["world_p"], task["seed"])
    tr_ids = None
    if task.get("trace"):
        X, sal, posts, U = world
        nposts = np.bincount(posts["b"], minlength=len(dfh))
        nmis = np.bincount(posts["b"], weights=posts["mis"].astype(float), minlength=len(dfh))
        tr_ids = np.where((X[:, 0] < 2) & (X[:, -1] >= 2) & (nposts >= 3) & (nmis >= 1))[0][:40]
    out = []
    for tags, cfg, pp, keep in task["runs"]:
        res, ts, tr = run_policy(cfg, world, dfh, _G["items"], _G["P1"], _G["O"], pp,
                                 trace_ids=tr_ids if keep else None)
        res.update(tags)
        res["seed"] = task["seed"]
        extra = None
        if keep:
            has = tr["first"] > 0
            T = world[0].shape[1] - 1
            event = tr["det"][has] > 0
            time_ = np.where(event, tr["det"][has] - tr["first"][has], T - tr["first"][has])
            extra = {"ts": ts, "km_t": time_.astype(int).tolist(), "km_e": event.tolist(),
                     "km_hi": tr["hi"][has].tolist()}
            if tr_ids is not None:
                extra["trace"] = {"trace": {str(k): v for k, v in tr["trace"].items()},
                                  "events": {str(k): v for k, v in tr["events"].items()},
                                  "truth": {str(k): v for k, v in tr["truth"].items()}}
        out.append((res, extra))
    return out


def tune(ex, p, n_seeds, t0):
    """All hyper-parameters are selected on the calibration half only (objective: Brier score, strictly proper)."""
    seeds = [10_000 + s for s in range(n_seeds)]

    def best_of(configs):
        return _best_of(ex, p, seeds, configs)

    log = {}
    (k0, ks), sc = best_of([((k0, ks), dict(POLICIES["B3-linearPool"], kappa0=k0), dict(p, kappa_sar=ks))
                            for k0 in [0.05, 0.1, 0.25] for ks in [0.05, 0.1, 0.25, 0.5, 1.0]])
    POLICIES["B3-linearPool"]["kappa0"], p["kappa_sar"] = k0, ks
    log["B3"] = (sc, k0, ks)
    (e4, e4s), sc = best_of([((e, es), POLICIES["B4-confTempered"], dict(p, eta_b4=e, eta_b4_sar=es))
                             for e in [0.5, 0.75, 1.0] for es in [0.25, 0.5, 0.75, 1.0]])
    p["eta_b4"], p["eta_b4_sar"] = e4, e4s
    log["B4"] = (sc, e4, e4s)
    (e, es), sc = best_of([((e, es), POLICIES["P-full"], dict(p, mu_hat=0.0, eta=e, eta_sar=es))
                           for e in [0.5, 0.75, 1.0] for es in [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]])
    p["mu_hat"], p["eta"], p["eta_sar"] = 0.0, e, es
    log["P_stage1"] = (sc, e, es)
    (eps, ws), sc = best_of([((eps, ws), POLICIES["P-full"], dict(p, eps=eps, wstar=ws))
                             for eps in [0.02, 0.05, 0.15, 0.3] for ws in [0.3, 0.5, 0.7]])
    p["eps"], p["wstar"] = eps, ws
    log["P_stage2"] = (sc, eps, ws)
    print(f"tuned: {log} ({time.time()-t0:.0f}s)", flush=True)
    return log


def _best_of(ex, p, seeds, configs):
    """Evaluate (key, cfg, params) configurations on calibration worlds; return the key with the lowest Brier score."""
    tasks = [dict(half="cal", seed=s, world_p=p, runs=[({"key": i}, c, pp, False)])
             for s in seeds for i, (_, c, pp) in enumerate(configs)]
    sc = {}
    for r in ex.map(_run_task, tasks):
        for res, _ in r:
            sc.setdefault(res["key"], []).append(res["mean_brier"])
    scores = [float(np.mean(sc[i])) for i in range(len(configs))]
    j = int(np.argmin(scores))
    return configs[j][0], scores[j]


def tune_sar_weights(ex, p, seeds):
    """Re-select the radar weight of every tunable policy for one sensor setting (calibration half, Brier score)."""
    out = {}
    out["kappa_sar"], _ = _best_of(ex, p, seeds, [(k, POLICIES["B3-linearPool"], dict(p, kappa_sar=k))
                                                 for k in [0.05, 0.1, 0.25, 0.5, 1.0]])
    out["eta_b4_sar"], _ = _best_of(ex, p, seeds, [(e, POLICIES["B4-confTempered"], dict(p, eta_b4_sar=e))
                                                  for e in [0.25, 0.5, 0.75, 1.0]])
    out["eta_sar"], _ = _best_of(ex, p, seeds, [(e, POLICIES["P-full"], dict(p, eta_sar=e))
                                               for e in [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]])
    return out


def summarize(rows):
    d = pd.DataFrame(rows)
    cols = ["auroc_mean", "ap_mean", "auroc_end", "precision_end", "recall_end", "f1_end", "final_acc_real",
            "mean_acc_changed", "delay_median", "det7", "det7_hi", "insp_yield_corrective", "false_alarm"]
    print(d.groupby("policy", sort=False)[cols].mean().round(4).to_string(), flush=True)


def build_tasks(p, n, sw):
    """sw[(param, value)] -> radar weights re-selected on the calibration half for that sensor setting."""
    P = POLICIES
    tasks = []
    for s in range(n):
        seed = 1000 + s
        tasks.append(dict(half="ev", seed=seed, world_p=p, trace=(s == 0),
                          runs=[({"exp": "main", "policy": m}, P[m], p, m in KEEP) for m in MAIN]))
        tasks.append(dict(half="ev", seed=seed, world_p=p, runs=[
            ({"exp": "sar_auc", "policy": m, "value": v}, P[m], dict(p, sar_auc=v, **sw[("sar_auc", v)]), False)
            for v in [0.70, 0.75, 0.81, 0.90, 0.95] for m in ["B1-SAR", "B2-naiveBayes", "B3-linearPool", "P-full"]]))
        tasks.append(dict(half="ev", seed=seed, world_p=p, runs=[
            ({"exp": "sar_rho", "policy": m, "value": v}, P[m], dict(p, sar_rho=v, **sw[("sar_rho", v)]), False)
            for v in [0.0, 0.25, 0.5, 0.75] for m in ["B2-naiveBayes", "B3-linearPool", "P-full", "P-sarUntempered"]]))
        tasks.append(dict(half="ev", seed=seed, world_p=p, runs=[
            ({"exp": "sar_period", "policy": m, "value": v}, P[m], dict(p, sar_period=v, **sw[("sar_period", v)]), False)
            for v in [6, 12, 24] for m in ["B1-SAR", "B3-linearPool", "P-full"]]))
        sens = [("eps", v) for v in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]] + \
               [("wstar", v) for v in [0.3, 0.4, 0.5, 0.6, 0.7]] + [("W", v) for v in [3, 7, 14, 28]] + \
               [("eta_sar", v) for v in [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]]
        tasks.append(dict(half="ev", seed=seed, world_p=p, runs=
            [({"exp": "gamma", "policy": "P-full", "value": g}, dict(P["P-full"], gamma=g), p, False)
             for g in [0.0, 0.5, 1.0, 2.0, 4.0]] +
            [({"exp": "sens", "policy": "P-full", "param": k, "value": v}, P["P-full"], dict(p, **{k: v}), False)
             for k, v in sens]))
        for mu in [0.0, 0.1, 0.2, 0.3, 0.4]:
            pm = dict(p, mu=mu)
            tasks.append(dict(half="ev", seed=seed, world_p=pm, runs=[
                ({"exp": "mu", "policy": m, "value": mu}, P[m], pm, False)
                for m in ["B2-naiveBayes", "B3-linearPool", "B4-confTempered", "P-full"]]))
    return tasks


def main():
    from concurrent.futures import ProcessPoolExecutor
    shp, feat_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
    quick = "--quick" in sys.argv
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = load_population(shp)
    cal, ev = df[df.calib].reset_index(drop=True), df[~df.calib].reset_index(drop=True)
    Q = estimate_Q(cal)
    O = inspection_model(0.9)
    p = dict(T=59, lam0=0.025, mu=0.2, C=int(round(0.005 * len(ev))), tau_sc=10.0, W=7, delta=1.0,
             eps=0.05, wstar=0.5, mu_hat=0.2, eta=1.0, eta_b4=1.0,
             sar_period=12, sar_auc=0.81, sar_rho=0.5, eta_sar=0.5, kappa_sar=0.25, eta_b4_sar=1.0)
    Lm = np.ones((K, K)) - np.eye(K)
    Lm[2:, :2] = 5.0                     # asymmetric loss: missing a severe/destroyed state costs five times more
    p["Lasym"] = Lm
    print(f"N_cal={len(cal)} N_eval={len(ev)} C={p['C']} Q=\n{np.round(Q, 5)}", flush=True)
    workers = max(1, min((os.cpu_count() or 2) - 1, 27))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(shp, feat_csv, Q.tolist(), O.tolist())) as ex:
        tlog = tune(ex, p, 2 if quick else 3, t0)
        pj = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
        json.dump({"params": pj, "Q_cal": Q.tolist(), "O": O.tolist(), "kappa0": POLICIES["B3-linearPool"]["kappa0"],
                   "tuning": {k: list(v) for k, v in tlog.items()}, "N_eval": len(ev), "N_cal": len(cal)},
                  open(os.path.join(out, "sim_config.json"), "w"), indent=2)
        seeds = [10_000 + s for s in range(2 if quick else 3)]
        sw = {}
        base_w = {k: p[k] for k in ("kappa_sar", "eta_b4_sar", "eta_sar")}
        for par, vals in [("sar_auc", [0.70, 0.75, 0.81, 0.90, 0.95]), ("sar_rho", [0.0, 0.25, 0.5, 0.75]),
                          ("sar_period", [6, 12, 24])]:
            for v in vals:
                if v == p[par]:
                    sw[(par, v)] = dict(base_w)
                else:
                    sw[(par, v)] = tune_sar_weights(ex, dict(p, **{par: v}), seeds)
        print("radar weights per setting:", {f"{k[0]}={k[1]}": v for k, v in sw.items()}, flush=True)
        json.dump({f"{k[0]}={k[1]}": v for k, v in sw.items()}, open(os.path.join(out, "sim_sar_weights.json"), "w"), indent=1)
        tasks = build_tasks(p, 5 if quick else 30, sw)
        rows, extras = [], {}
        for k, r in enumerate(ex.map(_run_task, tasks)):
            for res, extra in r:
                rows.append(res)
                if extra is not None:
                    extras.setdefault(res["policy"], []).append(extra)
            if k % 25 == 0:
                print(f"  task {k + 1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    for e, g in d.groupby("exp"):
        g.dropna(axis=1, how="all").to_csv(os.path.join(out, f"sim_{e}.csv"), index=False)
    summarize(d[d.exp == "main"].to_dict("records"))
    agg, grid = {}, np.arange(0, 41)
    cdf = {"d": grid}

    def km(t, e):
        """Kaplan-Meier estimate of the cumulative detection probability by delay d (censoring at the horizon)."""
        t, e = np.asarray(t), np.asarray(e, bool)
        S, F = 1.0, []
        for d in grid:
            n = (t >= d).sum()
            if n > 0:
                S *= 1.0 - ((t == d) & e).sum() / n
            F.append(1.0 - S)
        return F

    for m, xs in extras.items():
        agg[m] = {k: np.nanmean([x["ts"][k] for x in xs], axis=0).tolist() for k in xs[0]["ts"]}
        t = np.concatenate([x["km_t"] for x in xs])
        e = np.concatenate([x["km_e"] for x in xs])
        h = np.concatenate([x["km_hi"] for x in xs]).astype(bool)
        cdf[m] = km(t, e)
        cdf[m + "|hi"] = km(t[h], e[h])
    json.dump(agg, open(os.path.join(out, "sim_timeseries.json"), "w"))
    pd.DataFrame(cdf).to_csv(os.path.join(out, "sim_cdf.csv"), index=False)
    trace = {m: xs[0]["trace"] for m, xs in extras.items() if "trace" in xs[0]}
    json.dump(trace, open(os.path.join(out, "sim_trace.json"), "w"))
    print(f"ALL DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
