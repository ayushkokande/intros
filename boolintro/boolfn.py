"""Fourier view of boolean rules, and attribute-level statistics of a reported rule.

With inputs and output in {-1, +1}, every rule f over k attributes is a sum of
products of attributes:  f(x) = sum_S  c_S * prod_{i in S} x_i.

    degree 1   c_{i}: how much attribute i alone pushes the choice ("association",
               the analogue of the preference vector P in Atkinson et al.).
               Equals P(rule true | x_i = yes) - P(rule true | x_i = no).
    degree 2+  interactions (AND/OR structure, if-then switches, parity).

The squared coefficients sum to 1, so "degree mass" says what share of a rule is
explainable by single-attribute associations.  L1 is all degree 1; parity of three
is all degree 3.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np

from . import metrics, rules


def _pm(rule: rules.Rule, k: int) -> np.ndarray:
    return np.array([1.0 if rules.evaluate(rule, o) else -1.0 for o in range(2**k)])


def _chi(k: int) -> np.ndarray:
    """[2**k, k] attribute values as +-1."""
    return ((np.arange(2**k)[:, None] >> np.arange(k)) & 1) * 2.0 - 1.0


def fourier(rule: rules.Rule, k: int = 5, max_degree: int | None = None) -> dict[tuple, float]:
    f, x = _pm(rule, k), _chi(k)
    out = {}
    for d in range(0, (max_degree if max_degree is not None else k) + 1):
        for S in combinations(range(k), d):
            out[S] = float(np.mean(f * np.prod(x[:, list(S)], axis=1))) if S else float(f.mean())
    return out


def valence(rule: rules.Rule, k: int = 5) -> np.ndarray:
    """Degree-1 coefficients (the 'preference vector')."""
    f, x = _pm(rule, k), _chi(k)
    return (f[:, None] * x).mean(0)


def degree_vector(rule: rules.Rule, degree: int, k: int = 5) -> np.ndarray:
    """All coefficients of one degree, in combinations() order."""
    f, x = _pm(rule, k), _chi(k)
    return np.array([np.mean(f * np.prod(x[:, list(S)], axis=1)) for S in combinations(range(k), degree)])


def degree_mass(rule: rules.Rule, k: int = 5) -> np.ndarray:
    """Share of the rule's (non-constant) variance at each degree 1..k."""
    c = fourier(rule, k)
    m = np.zeros(k + 1)
    for S, v in c.items():
        m[len(S)] += v * v
    tot = m[1:].sum()
    return m[1:] / tot if tot > 0 else m[1:]


def influence(rule: rules.Rule, k: int = 5) -> np.ndarray:
    """P(flipping attribute i changes the rule's output), over all objects."""
    t = np.array([rules.evaluate(rule, o) for o in range(2**k)])
    return np.array([np.mean(t != t[np.arange(2**k) ^ (1 << i)]) for i in range(k)])


def report_stats(report: rules.Rule, target: rules.Rule, pairs: Sequence[tuple[int, int]],
                 choices: Sequence[int], k: int = 5) -> dict:
    """Attribute-level picture of one parsed report.

    choices: the object the model picked on each pair (faithfulness reference).
    relevant attribute: one the target rule depends on (influence > 0).
    sign_ok: on relevant attributes that both rules give a non-zero degree-1 weight,
             do the signs agree?  (Undefined when the target has no degree-1 weight.)
    """
    sc = [metrics.pair_score(report, a, b, c) for (a, b), c in zip(pairs, choices)]
    dec = [s for s in sc if s != 0.5]
    rel = influence(target, k) > 0
    men = influence(report, k) > 0
    vt, vr = valence(target, k), valence(report, k)
    both = rel & men & (np.abs(vt) > 1e-9) & (np.abs(vr) > 1e-9)
    return {
        "faith": float(np.mean(sc)),
        "tie": float(np.mean([s == 0.5 for s in sc])),
        "decisive_acc": float(np.mean(dec)) if dec else math.nan,
        "n_attr": int(men.sum()),
        "rel_mentioned": float(men[rel].mean()) if rel.any() else math.nan,
        "irr_mentioned": float(men[~rel].mean()) if (~rel).any() else math.nan,
        "sign_ok": float((np.sign(vr[both]) == np.sign(vt[both])).mean()) if both.any() else math.nan,
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def association_ceiling(target: rules.Rule, pairs, choices, k: int = 5) -> float:
    """Best faithfulness reachable by a single literal (the most an association-only
    report naming one attribute can score)."""
    best = 0.0
    for i in range(k):
        for lit in (("var", i), ("not", ("var", i))):
            best = max(best, metrics.rule_score(lit, pairs, choices))
    return best
