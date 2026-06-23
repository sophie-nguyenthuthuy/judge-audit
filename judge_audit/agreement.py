"""Reliability / agreement metrics for LLM-as-judge scores.

All metrics are pure NumPy. Krippendorff's alpha is the workhorse: it handles
ordinal Likert scales, missing values, and any number of raters — the right
tool for judge↔human and judge↔judge agreement on a 1..K scale.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "krippendorff_alpha",
    "spearman",
    "self_consistency",
]


def _coincidence(reliability_data: np.ndarray):
    """Build the coincidence matrix over distinct values.

    reliability_data: (n_raters, n_units) array, np.nan for missing.
    Returns (values, coincidence, marginals, n_pairable).
    """
    data = np.asarray(reliability_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("reliability_data must be 2D (raters x units)")

    values = np.unique(data[~np.isnan(data)])
    idx = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V), dtype=float)

    for unit in data.T:  # iterate units (columns)
        vals = unit[~np.isnan(unit)]
        m_u = len(vals)
        if m_u < 2:
            continue  # single-coder units carry no pairable information
        # each ordered pair within the unit contributes 1/(m_u - 1)
        for a in range(m_u):
            for b in range(m_u):
                if a == b:
                    continue
                o[idx[vals[a]], idx[vals[b]]] += 1.0 / (m_u - 1)

    marginals = o.sum(axis=1)
    n = marginals.sum()
    return values, o, marginals, n


def _delta2(values: np.ndarray, marginals: np.ndarray, level: str) -> np.ndarray:
    """Squared difference function δ²(c,k) for the chosen measurement level."""
    V = len(values)
    D = np.zeros((V, V), dtype=float)
    if level == "nominal":
        for c in range(V):
            for k in range(V):
                D[c, k] = 0.0 if c == k else 1.0
    elif level == "interval":
        for c in range(V):
            for k in range(V):
                D[c, k] = (values[c] - values[k]) ** 2
    elif level == "ordinal":
        # δ_ordinal uses cumulative marginals between the two ranks
        for c in range(V):
            for k in range(V):
                lo, hi = (c, k) if c <= k else (k, c)
                s = marginals[lo:hi + 1].sum() - (marginals[c] + marginals[k]) / 2.0
                D[c, k] = s ** 2
    else:
        raise ValueError(f"unknown level: {level!r}")
    return D


def krippendorff_alpha(reliability_data, level: str = "interval") -> float:
    """Krippendorff's alpha reliability coefficient.

    reliability_data: array-like (n_raters, n_units); np.nan marks missing.
    level: 'nominal' | 'ordinal' | 'interval'.

    Returns alpha in (-inf, 1]. 1 == perfect agreement, 0 == chance,
    < 0 == systematic disagreement. Returns nan if no pairable units exist.
    """
    values, o, marginals, n = _coincidence(reliability_data)
    if n < 2 or len(values) < 2:
        return float("nan")
    D = _delta2(values, marginals, level)

    D_o = (o * D).sum()
    D_e = (np.outer(marginals, marginals) * D).sum() / (n - 1)
    if D_e == 0:
        return 1.0  # everyone agrees; no expected disagreement
    return float(1.0 - D_o / D_e)


def spearman(x, y) -> float:
    """Spearman rank correlation (ties handled by average ranking)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def self_consistency(repeats: np.ndarray, level: str = "ordinal") -> dict:
    """Test-retest reliability: same judge, same items, scored R times.

    repeats: (n_items, n_repeats) array of scores.
    Returns alpha (treating repeats as raters), mean per-item std, and the
    modal-flip rate (fraction of items whose modal score is not unanimous).
    This is the CEILING on everything downstream — a judge that disagrees with
    itself cannot reliably rank models.
    """
    R = np.asarray(repeats, dtype=float)
    if R.ndim != 2 or R.shape[1] < 2:
        raise ValueError("repeats must be (n_items, n_repeats>=2)")
    alpha = krippendorff_alpha(R.T, level=level)  # repeats as raters
    item_std = np.nanstd(R, axis=1)
    # fraction of items not scored identically across all repeats
    flip = np.mean([len(np.unique(row[~np.isnan(row)])) > 1 for row in R])
    return {
        "alpha": alpha,
        "mean_item_std": float(np.nanmean(item_std)),
        "flip_rate": float(flip),
    }
