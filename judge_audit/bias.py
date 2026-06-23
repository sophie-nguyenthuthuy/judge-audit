"""Systematic-bias probes for LLM judges.

These are *controlled-perturbation* measurements: you feed the judge matched
inputs that differ only in the nuisance variable (option order, length, ...)
and measure how much the score moves. Observational correlation is not enough.
"""
from __future__ import annotations

import numpy as np

__all__ = ["position_bias", "length_bias"]


def position_bias(winner_ab, winner_ba) -> dict:
    """Pairwise position (order) bias from order-swapped judgements.

    winner_ab[i]: which side won when shown as (A, B) — 0 for first, 1 for second.
    winner_ba[i]: which side won when the SAME pair is shown swapped as (B, A).

    A position-neutral judge picks the same *content* regardless of slot, so the
    slot index should flip between the two runs. We report:
      consistency      — fraction whose content-winner agreed across orders
      flip_rate        — 1 - consistency (order changed the verdict)
      first_pos_rate   — fraction of all judgements that picked the FIRST slot
                         (0.5 == neutral; >0.5 == first-position favouritism)
    """
    ab = np.asarray(winner_ab, dtype=int)
    ba = np.asarray(winner_ba, dtype=int)
    if ab.shape != ba.shape:
        raise ValueError("winner_ab and winner_ba must align")

    # content winner in run AB: slot value directly (0=content-A, 1=content-B)
    # content winner in run BA: slot is swapped, so 0=content-B, 1=content-A
    content_ab = ab
    content_ba = 1 - ba
    consistency = float(np.mean(content_ab == content_ba))

    first_picks = np.concatenate([ab == 0, ba == 0])
    first_pos_rate = float(np.mean(first_picks))
    return {
        "consistency": consistency,
        "flip_rate": 1.0 - consistency,
        "first_pos_rate": first_pos_rate,
    }


def length_bias(scores, lengths, control=None, transform="log") -> dict:
    """Verbosity bias: does score move with response length, holding quality?

    scores:  judge scores.
    lengths: response length (tokens/chars) per item.
    control: optional ground-truth/quality covariate to partial out. If given,
             we regress score on [1, quality, z(length)] and report the length
             coefficient (the bias that survives after controlling for quality).
             If omitted we report the raw standardized slope.
    transform: 'log' (default) standardizes log-length — token counts are
             heavy-tailed, so log is the right scale and matches how verbosity
             effects actually accrue; 'raw' standardizes length directly.

    Returns the length coefficient (per +1 SD of [log-]length) and its
    standard error from the OLS normal equations.
    """
    y = np.asarray(scores, dtype=float)
    L = np.asarray(lengths, dtype=float)
    if transform == "log":
        L = np.log(np.clip(L, 1e-9, None))
    elif transform != "raw":
        raise ValueError("transform must be 'log' or 'raw'")
    zL = (L - L.mean()) / (L.std() + 1e-12)

    if control is None:
        X = np.column_stack([np.ones_like(zL), zL])
        coef_idx = 1
    else:
        c = np.asarray(control, dtype=float)
        X = np.column_stack([np.ones_like(zL), c, zL])
        coef_idx = 2

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {
        "length_coef_per_sd": float(beta[coef_idx]),
        "length_coef_se": float(se[coef_idx]),
        "t_stat": float(beta[coef_idx] / (se[coef_idx] + 1e-12)),
    }
