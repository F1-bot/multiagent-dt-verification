"""Core verification modules of the multi-agent digital-twin (DT) verification method.

All functions are pure and operate on belief vectors over ordinal damage states K = {0,1,2,3}
(0 none/possible, 1 moderate, 2 severe, 3 destroyed). They are the "verification modules" that a
tool-smith agent would generate; they are also the subjects of the DT-oracle mutation study (E3).
"""
import numpy as np

EPS = 1e-12


def normalize(p):
    """Normalise belief vector(s) along the last axis."""
    p = np.asarray(p, dtype=float)
    s = p.sum(axis=-1, keepdims=True)
    return p / np.maximum(s, EPS)


def aging(Pi, P1):
    """Prediction step of the DT state: propagate beliefs through the one-step transition matrix."""
    return Pi @ P1


def evidence_likelihood(p_obs, prior_obs, state_to_obs):
    """Virtual-evidence likelihood over DT states from a calibrated classifier posterior.
    p_obs: posterior over observation classes; prior_obs: class prior of the classifier's training data;
    state_to_obs: index map from DT states to observation classes."""
    ratio = np.asarray(p_obs, float) / np.maximum(np.asarray(prior_obs, float), EPS)
    return ratio[np.asarray(state_to_obs)]


def robust_likelihood(lik, mu):
    """epsilon-contamination likelihood: with probability mu the evidence is misattributed (uninformative,
    likelihood 1 in virtual-evidence units), otherwise it is genuine. Bounds the influence of one item."""
    mu = float(np.clip(mu, 0.0, 1.0))
    return (1.0 - mu) * np.asarray(lik, float) + mu


def reliability_to_weight(r, n_classes):
    """Map a probability of correct evidence r to a tempering exponent (excess over chance)."""
    chance = 1.0 / n_classes
    return np.clip((np.asarray(r, float) - chance) / (1.0 - chance), 0.0, 1.0)


def tempered_update(prior, lik, w):
    """Reliability-tempered (power-likelihood) Bayesian update: posterior ∝ prior · lik^w, w ∈ [0, 1]."""
    w = np.clip(w, 0.0, 1.0)
    post = np.asarray(prior, float) * np.power(np.maximum(np.asarray(lik, float), EPS), w)
    return normalize(post)


def linear_pool(prior, q, alpha):
    """Confidence-weighted linear opinion pool (baseline aggregation)."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return normalize((1.0 - alpha) * np.asarray(prior, float) + alpha * np.asarray(q, float))


def entropy(p):
    p = np.asarray(p, float)
    return -np.sum(p * np.log(np.maximum(p, EPS)), axis=-1)


def bayes_surprise(post, prior):
    """Bayesian surprise KL(post || prior) in nats."""
    post, prior = np.asarray(post, float), np.asarray(prior, float)
    return np.sum(post * (np.log(np.maximum(post, EPS)) - np.log(np.maximum(prior, EPS))), axis=-1)


def verification_decision(surprise, w, eps_s, w_star):
    """Verifier-agent rule: 0 = CONFIRM, 1 = CHANGE (accept + request verification), 2 = QUARANTINE."""
    if surprise <= eps_s:
        return 0
    return 1 if w >= w_star else 2


def project_monotone(pi, lb):
    """Invariant 'no improvement without a repair record': remove belief mass below the lower bound lb."""
    out = np.array(pi, dtype=float)
    lb = int(lb)
    if lb <= 0:
        return normalize(out)
    out[:lb] = 0.0
    if out.sum() <= EPS:
        out[lb] = 1.0
    return normalize(out)


def monotone_violation(pi, lb, tol=1e-6):
    """True if belief mass below the lower bound exceeds tol (invariant check)."""
    lb = int(lb)
    return lb > 0 and float(np.sum(np.asarray(pi, float)[:lb])) > tol


def expected_info_gain(Pi, O):
    """Expected entropy reduction of each belief in Pi (N, K) from one observation with model O (K, Y)."""
    Pi = np.atleast_2d(np.asarray(Pi, float))
    joint = Pi[:, :, None] * O[None, :, :]
    py = joint.sum(axis=1)
    post = joint / np.maximum(py[:, None, :], EPS)
    h_post = -(post * np.log(np.maximum(post, EPS))).sum(axis=1)
    return entropy(Pi) - (py * h_post).sum(axis=1)


def expected_accuracy_gain(Pi, O):
    """Decision-theoretic value of information (0-1 loss): expected increase of the MAP-state accuracy
    after one observation with model O (K, Y). Non-negative by convexity of max."""
    Pi = np.atleast_2d(np.asarray(Pi, float))
    joint = Pi[:, :, None] * O[None, :, :]
    return joint.max(axis=1).sum(axis=1) - Pi.max(axis=1)


def expected_risk_reduction(Pi, O, L):
    """Decision-theoretic VoI for a general loss matrix L[k, a] (true state k, declared state a):
    prior Bayes risk minus expected posterior Bayes risk. For the 0-1 loss it equals expected_accuracy_gain."""
    Pi = np.atleast_2d(np.asarray(Pi, float))
    r0 = (Pi @ L).min(axis=1)
    joint = Pi[:, :, None] * O[None, :, :]
    r1 = np.einsum("nky,ka->nya", joint, L).min(axis=2).sum(axis=1)
    return r0 - r1


def fidelity_index(Pi, omega=None):
    """DT fidelity (currency) index: criticality-weighted expected accuracy of the MAP state."""
    m = np.atleast_2d(np.asarray(Pi, float)).max(axis=-1)
    if omega is None:
        return float(m.mean())
    omega = np.asarray(omega, float)
    return float((omega * m).sum() / np.maximum(omega.sum(), EPS))


def social_concern(neg_mass):
    """Saturating social-concern score from the decayed mass of negative-sentiment evidence."""
    return 1.0 - np.exp(-np.maximum(np.asarray(neg_mass, float), 0.0))


def priority(omega, ig, sc, gamma, flagged, delta):
    """Socially weighted value-of-information priority used by the planner agent."""
    return np.asarray(omega, float) * (1.0 + gamma * np.asarray(sc, float)) * np.asarray(ig, float) * \
        (1.0 + delta * np.asarray(flagged, float))
