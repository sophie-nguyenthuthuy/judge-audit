"""Temporal-drift monitoring for a judge held on a fixed anchor set.

The provider silently ships a new model version; your prompt template changes;
sampling temperature gets nudged. Any of these moves the judge's scale without
touching the things you're trying to measure. Pin a *golden anchor set* (items
with stable human-consensus labels), re-score it every batch, and run a control
chart on the anchor mean. Same idea as a model-drift gate — applied to the eval.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ewma_chart", "cusum"]


def ewma_chart(series, mu0=None, sigma0=None, lam=0.3, L=3.0) -> dict:
    """EWMA control chart over a per-batch statistic (e.g. anchor-set mean).

    series: sequence of the monitored statistic, one value per batch in order.
    mu0, sigma0: in-control mean/std. If None, estimated from the first half of
                 the series (assumed in-control baseline).
    lam: EWMA smoothing (0<lam<=1). L: control-limit width in sigmas.

    Returns the EWMA path, time-varying control limits, and the index of the
    FIRST out-of-control batch (or None).
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty series")
    if mu0 is None or sigma0 is None:
        base = x[: max(n // 2, 2)]
        mu0 = float(base.mean()) if mu0 is None else mu0
        sigma0 = float(base.std(ddof=1)) if sigma0 is None else sigma0
    sigma0 = max(sigma0, 1e-9)

    z = np.empty(n)
    prev = mu0
    for t in range(n):
        prev = lam * x[t] + (1 - lam) * prev
        z[t] = prev

    t_arr = np.arange(1, n + 1)
    width = L * sigma0 * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t_arr)))
    ucl = mu0 + width
    lcl = mu0 - width

    ooc = np.where((z > ucl) | (z < lcl))[0]
    first = int(ooc[0]) if len(ooc) else None
    return {
        "ewma": z,
        "ucl": ucl,
        "lcl": lcl,
        "mu0": mu0,
        "sigma0": sigma0,
        "first_alarm": first,
    }


def cusum(series, mu0=None, sigma0=None, k=0.5, h=5.0) -> dict:
    """Two-sided CUSUM. Catches small persistent shifts EWMA may smooth over.

    k: slack in sigmas (reference value, ~half the shift you want to detect).
    h: decision interval in sigmas. Alarm when |cumsum| > h.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if mu0 is None or sigma0 is None:
        base = x[: max(n // 2, 2)]
        mu0 = float(base.mean()) if mu0 is None else mu0
        sigma0 = float(base.std(ddof=1)) if sigma0 is None else sigma0
    sigma0 = max(sigma0, 1e-9)

    zscore = (x - mu0) / sigma0
    sh = np.zeros(n)
    sl = np.zeros(n)
    for t in range(n):
        prev_h = sh[t - 1] if t else 0.0
        prev_l = sl[t - 1] if t else 0.0
        sh[t] = max(0.0, prev_h + zscore[t] - k)
        sl[t] = min(0.0, prev_l + zscore[t] + k)
    ooc = np.where((sh > h) | (sl < -h))[0]
    first = int(ooc[0]) if len(ooc) else None
    return {"sh": sh, "sl": sl, "h": h, "first_alarm": first}
