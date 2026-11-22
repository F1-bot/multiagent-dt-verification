"""E3: Fault-detection power of DT-oracle tests for generated verification modules (mutation analysis).

Subject: verification modules in dtcore.py (the code a tool-smith agent would synthesise).
Suites:  EB  - example-based unit tests (3 typical input/output examples per function, as LLM-written tests);
         DP  - DT-oracle property tests derived from DT invariants (simplex, identities, monotonicity, bounds);
         DR  - DT-oracle replay tests: historical verified decisions logged from a reference DT run.
Usage: python mutation_study.py <unosat_shp> <crisismmd_features.csv> <out_dir>
"""
import ast
import copy
import json
import os
import sys
import types
import time
import warnings
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dtcore as ref
import sim_dt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dtcore.py")
SRC = open(SRC_PATH, encoding="utf-8").read()
TARGETS = ["evidence_likelihood", "reliability_to_weight", "tempered_update", "linear_pool", "bayes_surprise",
           "verification_decision", "project_monotone", "monotone_violation", "expected_accuracy_gain",
           "fidelity_index", "social_concern", "priority"]
K = 4
O_REF = sim_dt.inspection_model(0.9)
RNG = np.random.default_rng(7)


# Mutants judged semantically equivalent by manual inspection (index in the deterministic mutant list).
EQUIVALENT = {
    32: 'axis -1 vs 0 identical for the 1-D belief vectors used by the DT',
    47: 'lb<=0 vs lb<=-1: lb=0 path is a no-op slice, same output',
    48: 'lb<=0 vs lb<0: lb=0 path is a no-op slice, same output',
    51: 'sum vs mean against EPS: differs only for total mass in (1e-12, 4e-12]',
    52: '<= vs < EPS: differs only at exact equality',
    55: 'out[lb]=2.0 is renormalised to the same one-hot vector',
    57: 'lb>0 vs lb>-1: lb=0 slices an empty prefix, sum=0',
    58: 'lb>0 vs lb>=0: lb=0 slices an empty prefix, sum=0',
    60: 'sum vs mean against tol=1e-6: threshold rescaled by lb, no behavioural difference at DT scales',
    61: '> vs >= tol: differs only at exact equality',
    76: 'max(axis=-1) vs max(axis=1) identical after atleast_2d',
}


def rand_simplex(n, k=K, rng=RNG):
    return rng.dirichlet(np.ones(k) * 0.7, size=n)


# ---------------------------------------------------------------- mutant generation
class Mutator(ast.NodeTransformer):
    BIN = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div], ast.Div: [ast.Mult], ast.Pow: [ast.Mult]}
    CMP = {ast.Lt: [ast.LtE, ast.Gt], ast.LtE: [ast.Lt, ast.GtE], ast.Gt: [ast.GtE, ast.Lt], ast.GtE: [ast.Gt, ast.LtE],
           ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq]}
    CALL = {"maximum": "minimum", "minimum": "maximum", "argmax": "argmin", "max": "min", "min": "max", "sum": "mean"}

    def __init__(self, target_fn, point=None, variant=None):
        self.fn, self.point, self.variant = target_fn, point, variant
        self.count, self.inside, self.desc = 0, False, None
        self.points = []

    def visit_FunctionDef(self, node):
        if node.name != self.fn:
            return node
        self.inside = True
        node.body = [self.visit(n) for n in node.body]
        self.inside = False
        return node

    def _mut(self, node, options, kind):
        if not self.inside:
            return node
        here = self.count
        self.count += 1          # every candidate node consumes exactly one point index
        for v, opt in enumerate(options):
            if self.point is None:
                self.points.append((here, v, kind))
            elif self.point == here and self.variant == v:
                self.desc = kind
                return opt()
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        reps = self.BIN.get(type(node.op), [])
        return self._mut(node, [lambda r=r: ast.BinOp(left=node.left, op=r(), right=node.right) for r in reps], "AOR")

    def visit_Compare(self, node):
        self.generic_visit(node)
        reps = self.CMP.get(type(node.ops[0]), [])
        return self._mut(node, [lambda r=r: ast.Compare(left=node.left, ops=[r()] + node.ops[1:],
                                                         comparators=node.comparators) for r in reps], "ROR")

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return node
        v = node.value
        alts = [v + 1, v - 1] if isinstance(v, int) else ([1.0] if v == 0.0 else [0.0, v * 2.0])
        return self._mut(node, [lambda a=a: ast.Constant(value=a) for a in alts], "CRP")

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.USub, ast.Not)):
            return self._mut(node, [lambda: node.operand], "UOI")
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else None)
        if name in self.CALL:
            def rep():
                n2 = copy.deepcopy(node)
                if isinstance(n2.func, ast.Attribute):
                    n2.func.attr = self.CALL[name]
                else:
                    n2.func.id = self.CALL[name]
                return n2
            return self._mut(node, [rep], "FCR")
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        return self._mut(node, [lambda: ast.IfExp(test=node.test, body=node.orelse, orelse=node.body)], "BSW")


def make_mutants():
    tree = ast.parse(SRC)
    mutants = []
    for fn in TARGETS:
        m = Mutator(fn)
        m.visit(copy.deepcopy(tree))
        for point, variant, kind in m.points:
            mm = Mutator(fn, point, variant)
            new = ast.fix_missing_locations(mm.visit(copy.deepcopy(tree)))
            src = ast.unparse(new)
            mutants.append({"fn": fn, "kind": kind, "point": point, "variant": variant, "src": src})
    return mutants


def load_module(src, name="dtcore_mut"):
    mod = types.ModuleType(name)
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------- example-based tests (EB)
def build_eb():
    """3 typical examples per function, expected values snapshotted from the reference implementation."""
    ex = {}
    pri = np.array([0.6, 0.25, 0.1, 0.05]); lik = np.array([0.5, 1.2, 2.0, 2.0])
    ex["evidence_likelihood"] = [((np.array([0.2, 0.3, 0.5]), np.array([0.135, 0.238, 0.627]), np.array([0, 1, 2, 2])),),
                                 ((np.array([0.7, 0.2, 0.1]), np.array([0.3, 0.3, 0.4]), np.array([0, 1, 2, 2])),),
                                 ((np.array([0.1, 0.1, 0.8]), np.array([0.2, 0.3, 0.5]), np.array([0, 1, 2, 2])),)]
    ex["reliability_to_weight"] = [((0.8, 3),), ((0.6, 3),), ((0.9, 4),)]
    ex["tempered_update"] = [((pri, lik, 0.5),), ((pri, lik, 0.8),), ((np.array([0.25] * 4), lik, 0.3),)]
    ex["linear_pool"] = [((pri, np.array([0.1, 0.2, 0.3, 0.4]), 0.3),), ((pri, np.array([0.0, 0.0, 0.5, 0.5]), 0.5),),
                         ((np.array([0.25] * 4), pri, 0.7),)]
    ex["bayes_surprise"] = [((np.array([0.4, 0.3, 0.2, 0.1]), pri),), ((pri, np.array([0.25] * 4)),),
                            ((np.array([0.1, 0.2, 0.3, 0.4]), pri),)]
    ex["verification_decision"] = [((0.05, 0.6, 0.15, 0.5),), ((0.4, 0.7, 0.15, 0.5),), ((0.4, 0.2, 0.15, 0.5),)]
    ex["project_monotone"] = [((pri, 1),), ((pri, 2),), ((np.array([0.1, 0.2, 0.3, 0.4]), 1),)]
    ex["monotone_violation"] = [((pri, 1),), ((np.array([0.0, 0.2, 0.4, 0.4]), 1),), ((pri, 2),)]
    ex["expected_accuracy_gain"] = [((np.array([pri, [0.25] * 4]), O_REF),), ((np.array([[0.55, 0.45, 0, 0]]), O_REF),),
                                    ((np.array([[0.1, 0.2, 0.3, 0.4]]), O_REF),)]
    ex["fidelity_index"] = [((np.array([pri, [0.1, 0.2, 0.3, 0.4]]),),), ((np.array([pri, [0.1, 0.2, 0.3, 0.4]]), np.array([1.0, 3.0])),),
                            ((np.array([[0.7, 0.1, 0.1, 0.1]]),),)]
    ex["social_concern"] = [((0.5,),), ((2.0,),), ((np.array([0.1, 1.0]),),)]
    ex["priority"] = [((np.array([1.0, 3.0]), np.array([0.2, 0.1]), np.array([0.5, 0.0]), 1.0, np.array([False, True]), 1.0),),
                      ((np.array([1.0]), np.array([0.3]), np.array([0.2]), 2.0, np.array([False]), 1.0),),
                      ((np.array([1.0, 1.0]), np.array([0.05, 0.4]), np.array([0.9, 0.1]), 0.5, np.array([True, False]), 1.0),)]
    tests = []
    for fn, cases in ex.items():
        for (args,) in cases:
            exp = getattr(ref, fn)(*args)
            tests.append((fn, args, exp))
    return tests


def run_eb(mod, tests, fns):
    for fn, args, exp in tests:
        if fn not in fns:
            continue
        got = getattr(mod, fn)(*args)
        if not np.allclose(np.asarray(got, float), np.asarray(exp, float), atol=1e-9, equal_nan=False):
            return False
    return True


# ---------------------------------------------------------------- DT-oracle property tests (DP)
def dp_tests(m):
    """Each test returns True if the property holds. Properties are derived from DT semantics, not from the reference code."""
    r = np.random.default_rng(11)
    n = 200
    pis, lks = rand_simplex(n, rng=r), r.gamma(1.0, 1.0, size=(n, K)) + 1e-3
    ws = r.random(n)
    t = {}
    simplex = lambda v: np.all(np.isfinite(v)) and np.all(v >= -1e-12) and np.allclose(np.sum(v, -1), 1.0)
    # tempered update: simplex; w=0 identity; w=1 Bayes; KL monotone in w; scale invariance; clipping
    def tu():
        for p, l, w in zip(pis, lks, ws):
            out = m.tempered_update(p, l, w)
            if not simplex(out): return False
            if not np.allclose(m.tempered_update(p, l, 0.0), p / p.sum()): return False
            b = p * l; b = b / b.sum()
            if not np.allclose(m.tempered_update(p, l, 1.0), b): return False
            if not np.allclose(m.tempered_update(p, 3.7 * l, w), out): return False
            if not np.allclose(m.tempered_update(p, l, 1.5), b) or not np.allclose(m.tempered_update(p, l, -0.5), p / p.sum()): return False
            k1 = np.sum(m.tempered_update(p, l, 0.3) * np.log(m.tempered_update(p, l, 0.3) / p))
            k2 = np.sum(m.tempered_update(p, l, 0.9) * np.log(m.tempered_update(p, l, 0.9) / p))
            if k2 < k1 - 1e-12: return False
        return True
    t["tempered_update"] = tu
    def lp():
        for p, q, a in zip(pis, rand_simplex(n, rng=r), ws):
            out = m.linear_pool(p, q, a)
            if not simplex(out) or not np.allclose(m.linear_pool(p, q, 0.0), p) or not np.allclose(m.linear_pool(p, q, 1.0), q): return False
            if np.any(out < np.minimum(p, q) - 1e-12) or np.any(out > np.maximum(p, q) + 1e-12): return False
        return True
    t["linear_pool"] = lp
    def el():
        for _ in range(n):
            pr = r.dirichlet(np.ones(3)); po = r.dirichlet(np.ones(3)); mp = np.array([0, 1, 2, 2])
            L = m.evidence_likelihood(po, pr, mp)
            if np.any(L <= 0) or L[2] != L[3]: return False
            L0 = m.evidence_likelihood(pr, pr, mp)          # uninformative evidence
            if not np.allclose(L0, L0[0]): return False
            if not np.allclose(L * pr[mp] / po[mp], 1.0): return False
        return True
    t["evidence_likelihood"] = el
    def rw():
        for kk in (2, 3, 4):
            if not np.isclose(m.reliability_to_weight(1.0 / kk, kk), 0.0) or not np.isclose(m.reliability_to_weight(1.0, kk), 1.0): return False
            g = np.linspace(-0.2, 1.2, 57)
            w = np.array([m.reliability_to_weight(x, kk) for x in g])
            if np.any(np.diff(w) < -1e-12) or np.any(w < 0) or np.any(w > 1): return False
        return True
    t["reliability_to_weight"] = rw
    def bs():
        for p, q in zip(pis, rand_simplex(n, rng=r)):
            if m.bayes_surprise(p, q) < -1e-12 or not np.isclose(m.bayes_surprise(p, p), 0.0): return False
        return True
    t["bayes_surprise"] = bs
    def vd():
        cases = [(0.1, 0.9, 0.15, 0.5, 0), (0.15, 0.9, 0.15, 0.5, 0), (0.2, 0.5, 0.15, 0.5, 1), (0.2, 0.49, 0.15, 0.5, 2),
                 (5.0, 0.0, 0.15, 0.5, 2), (0.16, 1.0, 0.15, 0.5, 1)]
        return all(m.verification_decision(s, w, e, ws_) == d for s, w, e, ws_, d in cases)
    t["verification_decision"] = vd
    def pm():
        for p in pis:
            for lb in range(K):
                out = m.project_monotone(p, lb)
                if not simplex(out) or np.sum(out[:lb]) > 1e-12: return False
                if lb > 0 and np.sum(p[lb:]) > 1e-9 and not np.allclose(out[lb:], p[lb:] / p[lb:].sum()): return False
                if lb == 0 and not np.allclose(out, p / p.sum()): return False
        if not np.allclose(m.project_monotone(np.array([1.0, 0, 0, 0]), 2), [0, 0, 1, 0]): return False
        return True
    t["project_monotone"] = pm
    def mv():
        for p in pis:
            for lb in range(K):
                if m.monotone_violation(m.project_monotone(p, lb), lb): return False
                if lb > 0 and np.sum(p[:lb]) > 1e-3 and not m.monotone_violation(p, lb): return False
            if m.monotone_violation(p, 0): return False
        return True
    t["monotone_violation"] = mv
    def voi():
        v = m.expected_accuracy_gain(pis, O_REF)
        mx = pis.max(1)
        if np.any(v < -1e-9) or np.any(v > 1 - mx + 1e-9): return False           # bounded by the residual error
        if not np.allclose(m.expected_accuracy_gain(np.eye(K), O_REF), 0.0, atol=1e-9): return False   # known state
        if not np.allclose(m.expected_accuracy_gain(pis, np.eye(K)), 1 - mx, atol=1e-9): return False  # perfect sensor
        if not np.allclose(m.expected_accuracy_gain(pis, np.full((K, K), 1.0 / K)), 0.0, atol=1e-9): return False  # useless sensor
        return True
    t["expected_accuracy_gain"] = voi
    def fi():
        f = m.fidelity_index(pis)
        if not (1.0 / K - 1e-12 <= f <= 1 + 1e-12): return False
        if not np.isclose(m.fidelity_index(np.eye(K)), 1.0) or not np.isclose(m.fidelity_index(np.full((3, K), 0.25)), 0.25): return False
        if not np.isclose(m.fidelity_index(pis, np.ones(n)), f): return False
        om = r.random(n) + 0.1
        if not np.isclose(m.fidelity_index(pis, om), np.sum(om * pis.max(1)) / om.sum()): return False
        return True
    t["fidelity_index"] = fi
    def sc():
        x = np.linspace(0, 10, 101)
        v = m.social_concern(x)
        return np.isclose(m.social_concern(0.0), 0.0) and np.all(np.diff(v) >= -1e-12) and np.all(v < 1) and np.all(v >= 0) \
            and np.isclose(m.social_concern(-3.0), 0.0)
    t["social_concern"] = sc
    def pr():
        om, igv, scv = r.random(n) + 0.5, r.random(n), r.random(n)
        f0 = np.zeros(n, bool)
        if not np.allclose(m.priority(om, igv, scv, 0.0, f0, 1.0), om * igv): return False
        if np.any(m.priority(om, igv, scv, 2.0, f0, 1.0) < m.priority(om, igv, scv, 1.0, f0, 1.0) - 1e-12): return False
        if np.any(m.priority(om, igv, scv, 1.0, ~f0, 1.0) < m.priority(om, igv, scv, 1.0, f0, 1.0) - 1e-12): return False
        if np.any(m.priority(om, igv + 0.1, scv, 1.0, f0, 1.0) < m.priority(om, igv, scv, 1.0, f0, 1.0) - 1e-12): return False
        return True
    t["priority"] = pr
    return t


# ---------------------------------------------------------------- replay oracle (DR)
def record_replay(shp, feat_csv, n_days=12, max_calls=300):
    """Log real inputs/outputs of the verification modules during a reference DT run (historical decisions)."""
    log = defaultdict(list)
    orig = {fn: getattr(sim_dt.dc, fn) for fn in TARGETS}

    def wrap(fn):
        f0 = orig[fn]
        def g(*a):
            out = f0(*a)
            if len(log[fn]) < max_calls:
                log[fn].append((copy.deepcopy(a), copy.deepcopy(out)))
            return out
        return g
    for fn in TARGETS:
        setattr(sim_dt.dc, fn, wrap(fn))
    try:
        df = sim_dt.load_population(shp)
        ev = df[~df.calib].reset_index(drop=True).iloc[:3000].reset_index(drop=True)
        Q = sim_dt.estimate_Q(df[df.calib])
        items = sim_dt.load_items(feat_csv)
        p = dict(T=n_days, lam0=0.05, mu=0.2, C=15, tau_sc=10.0, W=7, delta=1.0, eps=0.02, wstar=0.5, mu_hat=0.0, eta=1.0, eta_b4=1.0)
        world = sim_dt.make_world(ev, Q, items, p, 0)
        from scipy.linalg import expm
        sim_dt.run_policy(sim_dt.POLICIES["P-full"], world, ev, items, expm(Q), O_REF, p)
        sim_dt.run_policy(sim_dt.POLICIES["B3-linearPool"], world, ev, items, expm(Q), O_REF, p)
    finally:
        for fn in TARGETS:
            setattr(sim_dt.dc, fn, orig[fn])
    # add logged calls for functions not exercised by the simulation run
    return log


def run_dr(mod, log, fns):
    for fn in fns:
        for args, out in log.get(fn, []):
            got = getattr(mod, fn)(*copy.deepcopy(args))
            if not np.allclose(np.asarray(got, float), np.asarray(out, float), atol=1e-9):
                return False
    return True


def main():
    shp, feat_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    mutants = make_mutants()
    import difflib
    ref_lines = ast.unparse(ast.parse(SRC)).splitlines()
    for mu in mutants:
        ch = [l for l in difflib.ndiff(ref_lines, mu["src"].splitlines()) if l.startswith(("+ ", "- "))]
        assert len(ch) == 2, ("mutant must change exactly one line", mu["fn"], ch)
    print("mutants:", len(mutants), "(all single-point)", flush=True)
    eb = build_eb()
    log = record_replay(shp, feat_csv)
    print("replay log sizes:", {k: len(v) for k, v in log.items()}, flush=True)
    # sanity: reference passes all suites
    refmod = load_module(SRC, "dtcore_ref")
    assert run_eb(refmod, eb, TARGETS), "reference fails EB"
    assert all(f() for f in dp_tests(refmod).values()), "reference fails DP"
    assert run_dr(refmod, log, TARGETS), "reference fails DR"
    rows = []
    for k, mu in enumerate(mutants):
        try:
            mod = load_module(mu["src"])
        except Exception:
            continue
        fns = [mu["fn"]]
        def safe(fun):
            try:
                with np.errstate(all="ignore"):
                    return bool(fun())
            except Exception:
                return False
        eb_pass = safe(lambda: run_eb(mod, eb, fns))
        dp_pass = safe(lambda: dp_tests(mod)[mu["fn"]]())
        dr_pass = safe(lambda: run_dr(mod, log, fns)) if log.get(mu["fn"]) else True
        rows.append({"idx": k, "fn": mu["fn"], "kind": mu["kind"], "killed_EB": not eb_pass, "killed_DP": not dp_pass,
                     "killed_DR": not dr_pass, "has_replay": bool(log.get(mu["fn"]))})
    import pandas as pd
    d = pd.DataFrame(rows)
    d["killed_EB_DP"] = d.killed_EB | d.killed_DP
    d["killed_DP_DR"] = d.killed_DP | d.killed_DR
    d["killed_ALL"] = d.killed_EB | d.killed_DP | d.killed_DR
    d.to_csv(os.path.join(out, "mutation_rows.csv"), index=False)
    summ = {"n_mutants": int(len(d)), "time_s": time.time() - t0}
    for c in ["killed_EB", "killed_DP", "killed_DR", "killed_EB_DP", "killed_DP_DR", "killed_ALL"]:
        summ[c] = float(d[c].mean())
    summ["by_fn"] = d.groupby("fn")[["killed_EB", "killed_EB_DP", "killed_ALL"]].mean().round(3).to_dict(orient="index")
    summ["n_by_fn"] = d.groupby("fn").size().to_dict()
    summ["by_kind"] = d.groupby("kind")[["killed_EB", "killed_EB_DP", "killed_ALL"]].mean().round(3).to_dict(orient="index")
    summ["n_by_kind"] = d.groupby("kind").size().to_dict()
    summ["survived_all"] = int((~d.killed_ALL).sum())
    d["equivalent"] = d.idx.isin(EQUIVALENT)
    ne = d[~d.equivalent]
    summ["n_equivalent"] = int(d.equivalent.sum())
    summ["score_non_equivalent"] = {c: float(ne[c].mean()) for c in ["killed_EB", "killed_DP", "killed_DR", "killed_EB_DP", "killed_ALL"]}
    summ["by_fn_non_equivalent"] = ne.groupby("fn")[["killed_EB", "killed_DP", "killed_EB_DP", "killed_ALL"]].sum().astype(int).to_dict(orient="index")
    summ["equivalent_manual"] = {str(k): v for k, v in EQUIVALENT.items()}
    json.dump(summ, open(os.path.join(out, "mutation_study.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict)}, indent=1))
    print(d.groupby("fn")[["killed_EB", "killed_EB_DP", "killed_ALL"]].mean().round(3))
    print(d.groupby("kind")[["killed_EB", "killed_EB_DP", "killed_ALL"]].mean().round(3))


if __name__ == "__main__":
    main()
