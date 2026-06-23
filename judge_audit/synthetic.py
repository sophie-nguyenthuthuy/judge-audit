"""Synthetic judge with KNOWN injected pathologies.

The point of the benchmark: inject biases with ground-truth magnitudes, then
show the audit recovers them. Same philosophy as model-collapse-testbed —
plant the signal, prove the instrument finds it.

Injected pathologies
  - self-inconsistency      : Gaussian scoring noise (sigma_judge)
  - nonlinear miscalibration: judge compresses the top of the scale (gamma)
  - verbosity bias          : score rises with response length (verbosity_coef)
  - leniency DRIFT          : scale shifts up after a changepoint batch
  - position bias (pairwise): judge favours the first slot (first_pos_pref)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ["JudgeSpec", "make_benchmark"]


@dataclass
class JudgeSpec:
    n_batches: int = 12
    items_per_batch: int = 120
    anchor_size: int = 40
    repeats: int = 5                 # re-scores per item (self-consistency)
    scale: tuple = (1.0, 5.0)
    quality_gain: float = 1.0        # judge sensitivity to true quality
    gamma: float = 1.6              # >1 compresses the top end (miscalibration)
    verbosity_coef: float = 0.45     # score per +1 SD of length
    sigma_judge: float = 0.40        # self-inconsistency noise (score units)
    drift_changepoint: int = 7       # batch at which leniency starts shifting
    drift_magnitude: float = 0.9     # total upward shift after changepoint
    first_pos_pref: float = 0.68     # P(pick first slot) when items are tied
    n_pairwise: int = 300
    seed: int = 7

    truth: dict = field(default_factory=dict)


def _squash(q01, gamma):
    """Monotone compression of the top of the scale (concave for gamma>1)."""
    return q01 ** (1.0 / gamma)


def make_benchmark(spec: JudgeSpec | None = None) -> dict:
    """Generate a full benchmark dataset for a single judge.

    Returns a dict of arrays; `spec.truth` is filled with the injected values
    so the audit's recovered estimates can be checked against ground truth.
    """
    spec = spec or JudgeSpec()
    rng = np.random.default_rng(spec.seed)
    lo, hi = spec.scale
    span = hi - lo

    rows_q = []        # latent true quality (the human gold), per item
    rows_score = []    # judge mean score (1 draw), per item
    rows_len = []      # response length, per item
    rows_batch = []    # batch index, per item
    repeats = []       # (n_items, repeats) raw judge scores
    anchor_means = []  # per-batch mean over the FIXED anchor set

    # fixed anchor set: same latent quality + length every batch
    aq = rng.uniform(lo, hi, spec.anchor_size)
    alen = rng.lognormal(5.0, 0.5, spec.anchor_size)

    def judge(q, length, batch, n_draws):
        q01 = (q - lo) / span
        base01 = _squash(q01, spec.gamma) * spec.quality_gain
        zlen = (np.log(length) - 5.0) / 0.5
        leniency = 0.0
        if batch >= spec.drift_changepoint:
            # ramp the shift in over a couple of batches after the changepoint
            ramp = min(1.0, (batch - spec.drift_changepoint + 1) / 2.0)
            leniency = spec.drift_magnitude * ramp
        mean = lo + base01 * span + spec.verbosity_coef * zlen + leniency
        draws = mean[:, None] + rng.normal(0, spec.sigma_judge, (len(mean), n_draws))
        draws = np.clip(np.round(draws), lo, hi)  # discrete Likert 1..K
        return draws

    for b in range(spec.n_batches):
        q = rng.uniform(lo, hi, spec.items_per_batch)
        length = rng.lognormal(5.0, 0.5, spec.items_per_batch)
        draws = judge(q, length, b, spec.repeats)
        rows_q.append(q)
        rows_len.append(length)
        rows_batch.append(np.full(spec.items_per_batch, b))
        rows_score.append(draws[:, 0])  # first draw = the "production" score
        repeats.append(draws)

        adraws = judge(aq, alen, b, 1)[:, 0]
        anchor_means.append(float(adraws.mean()))

    # pairwise position-bias subset: tied-quality pairs, judge leans to slot 0
    tie = rng.uniform(lo + 0.4 * span, lo + 0.6 * span, spec.n_pairwise)
    # winner when shown (A,B): 0 with prob first_pos_pref
    winner_ab = (rng.uniform(size=spec.n_pairwise) >= spec.first_pos_pref).astype(int)
    winner_ba = (rng.uniform(size=spec.n_pairwise) >= spec.first_pos_pref).astype(int)

    spec.truth = {
        "verbosity_coef": spec.verbosity_coef,
        "sigma_judge": spec.sigma_judge,
        "drift_changepoint": spec.drift_changepoint,
        "first_pos_pref": spec.first_pos_pref,
        "gamma": spec.gamma,
    }

    return {
        "spec": spec,
        "quality": np.concatenate(rows_q),
        "score": np.concatenate(rows_score),
        "length": np.concatenate(rows_len),
        "batch": np.concatenate(rows_batch),
        "repeats": np.concatenate(repeats, axis=0),
        "anchor_means": np.array(anchor_means),
        "winner_ab": winner_ab,
        "winner_ba": winner_ba,
        "scale": spec.scale,
    }


def two_model_scores(spec: JudgeSpec | None = None, true_gap=0.05, n=300, seed=99):
    """A null-ish leaderboard: two models whose TRUE quality differs by a hair.
    Used to show bootstrap CIs refusing to call a winner on noise."""
    spec = spec or JudgeSpec()
    rng = np.random.default_rng(seed)
    lo, hi = spec.scale
    qa = rng.uniform(lo, hi, n)
    qb = np.clip(qa + true_gap, lo, hi)
    la = rng.lognormal(5.0, 0.5, n)
    lb = rng.lognormal(5.0, 0.5, n)

    def score(q, length):
        q01 = (q - lo) / (hi - lo)
        base = lo + _squash(q01, spec.gamma) * (hi - lo)
        zlen = (np.log(length) - 5.0) / 0.5
        m = base + spec.verbosity_coef * zlen
        return np.clip(np.round(m + rng.normal(0, spec.sigma_judge, n)), lo, hi)

    return {"a": score(qa, la), "b": score(qb, lb), "true_gap": true_gap}
