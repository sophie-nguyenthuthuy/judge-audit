"""Calibration: map raw judge scores onto the human/ground-truth scale, and
quantify how far off the raw scores were.

A judge can be reliable (consistent) yet miscalibrated (consistently 0.6 points
high, or compressing the top of the scale). Isotonic regression fits a
monotone correction from a held-out calibration split; ECE-style calibration
error quantifies before/after. Bootstrap CIs put error bars on leaderboard gaps
so you stop reading noise as signal.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "isotonic_fit",
    "IsotonicCalibrator",
    "calibration_error",
    "reliability_curve",
    "bootstrap_ci",
    "bootstrap_diff",
]


def isotonic_fit(x, y, weights=None):
    """Pool-Adjacent-Violators monotone (non-decreasing) regression.

    Returns (x_thresholds, y_levels) sorted by x, suitable for step interp.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[order], y[order], w[order]

    # PAVA on the y values
    level = list(ys)
    wt = list(ws)
    xv = list(xs)
    i = 0
    blocks = [[xv[i], level[i], wt[i]] for i in range(len(xv))]
    merged = []
    for b in blocks:
        merged.append(b)
        while len(merged) >= 2 and merged[-2][1] > merged[-1][1]:
            x2, y2, w2 = merged.pop()
            x1, y1, w1 = merged.pop()
            nw = w1 + w2
            merged.append([x2, (y1 * w1 + y2 * w2) / nw, nw])
    out_x = np.array([b[0] for b in merged])
    out_y = np.array([b[1] for b in merged])
    return out_x, out_y


class IsotonicCalibrator:
    """Fit on a calibration split, apply to held-out judge scores."""

    def __init__(self):
        self.x_ = None
        self.y_ = None

    def fit(self, judge_scores, target_scores, weights=None):
        self.x_, self.y_ = isotonic_fit(judge_scores, target_scores, weights)
        return self

    def transform(self, judge_scores):
        if self.x_ is None:
            raise RuntimeError("call fit() first")
        return np.interp(np.asarray(judge_scores, dtype=float), self.x_, self.y_)

    def fit_transform(self, judge_scores, target_scores, weights=None):
        return self.fit(judge_scores, target_scores, weights).transform(judge_scores)


def _normalize(a, lo, hi):
    return (np.asarray(a, dtype=float) - lo) / (hi - lo + 1e-12)


def calibration_error(pred, target, n_bins=10, scale=None) -> dict:
    """Binned calibration error (ECE-style) for a scalar judge.

    pred, target on the same scale. We bin items by predicted score and compare
    each bin's mean predicted vs mean target. Returns ECE (mean |gap|, weighted
    by bin count) and MCE (max |gap|), plus per-bin detail. `scale` = (lo, hi)
    to normalize to [0,1]; defaults to the data range of target.
    """
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    lo, hi = scale if scale is not None else (float(t.min()), float(t.max()))
    pn = np.clip(_normalize(p, lo, hi), 0, 1)
    tn = np.clip(_normalize(t, lo, hi), 0, 1)

    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins = []
    N = len(pn)
    for b in range(n_bins):
        m = (pn >= edges[b]) & (pn < edges[b + 1] if b < n_bins - 1 else pn <= edges[b + 1])
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({"lo": edges[b], "hi": edges[b + 1], "n": 0,
                         "mean_pred": np.nan, "mean_target": np.nan, "gap": np.nan})
            continue
        gap = abs(pn[m].mean() - tn[m].mean())
        ece += (cnt / N) * gap
        mce = max(mce, gap)
        bins.append({"lo": edges[b], "hi": edges[b + 1], "n": cnt,
                     "mean_pred": float(pn[m].mean()),
                     "mean_target": float(tn[m].mean()), "gap": float(gap)})
    return {"ece": float(ece), "mce": float(mce), "bins": bins}


def reliability_curve(pred, target, n_bins=10, scale=None):
    """Return (bin_pred, bin_target) points for a reliability diagram."""
    ce = calibration_error(pred, target, n_bins=n_bins, scale=scale)
    xs = [b["mean_pred"] for b in ce["bins"] if b["n"] > 0]
    ys = [b["mean_target"] for b in ce["bins"] if b["n"] > 0]
    return np.array(xs), np.array(ys)


def _rng(seed):
    return np.random.default_rng(seed)


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0) -> dict:
    """Percentile bootstrap CI for a statistic of a single sample."""
    v = np.asarray(values, dtype=float)
    rng = _rng(seed)
    n = len(v)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = stat(v[rng.integers(0, n, n)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return {"point": float(stat(v)), "lo": float(lo), "hi": float(hi)}


def bootstrap_diff(a, b, paired=False, n_boot=2000, alpha=0.05, seed=0) -> dict:
    """Bootstrap CI for mean(a) - mean(b). If the CI straddles 0, the gap is
    not distinguishable from noise — the headline use of this whole module.

    paired=True resamples item indices jointly (a, b scored on same items)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = _rng(seed)
    boot = np.empty(n_boot)
    if paired:
        if len(a) != len(b):
            raise ValueError("paired requires equal length")
        n = len(a)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boot[i] = a[idx].mean() - b[idx].mean()
    else:
        na, nb = len(a), len(b)
        for i in range(n_boot):
            boot[i] = a[rng.integers(0, na, na)].mean() - b[rng.integers(0, nb, nb)].mean()
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    point = float(a.mean() - b.mean())
    return {"diff": point, "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}
