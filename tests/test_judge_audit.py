import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_audit import (
    krippendorff_alpha, spearman, self_consistency,
    position_bias, length_bias, ewma_chart, cusum,
    IsotonicCalibrator, calibration_error, bootstrap_diff,
    JudgeSpec, make_benchmark, two_model_scores, audit,
)


# ---- agreement -------------------------------------------------------------

def test_krippendorff_known_value():
    # Krippendorff's canonical worked example. Values cross-checked against the
    # reference `krippendorff` PyPI package (exact match to 4 d.p.):
    #   nominal 0.7375, ordinal 0.8495, interval 0.8553
    nan = np.nan
    data = np.array([
        [nan, nan, nan, nan, nan, 3, 4, 1, 2, 1, 1, 3, 3, nan, 3],
        [1, nan, 2, 1, 3, 3, 4, 3, nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, 2, 1, 3, 4, 4, nan, 2, 1, 1, 3, 3, nan, 4],
        [1, nan, 2, 1, 3, 4, 4, 3, nan, nan, nan, nan, nan, nan, nan],
    ])
    assert abs(krippendorff_alpha(data, level="nominal") - 0.7375) < 1e-3
    assert abs(krippendorff_alpha(data, level="ordinal") - 0.8495) < 1e-3
    assert abs(krippendorff_alpha(data, level="interval") - 0.8553) < 1e-3


def test_krippendorff_perfect_and_chance():
    perfect = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=float)
    assert krippendorff_alpha(perfect, level="interval") > 0.99


def test_spearman_monotone():
    x = np.arange(20)
    y = x ** 2  # monotone increasing
    assert spearman(x, y) > 0.99
    assert spearman(x, -y) < -0.99


def test_self_consistency_noise_floor():
    rng = np.random.default_rng(0)
    truth = rng.uniform(1, 5, 200)
    noisy = np.clip(np.round(truth[:, None] + rng.normal(0, 0.4, (200, 6))), 1, 5)
    out = self_consistency(noisy, level="ordinal")
    assert 0.5 < out["alpha"] <= 1.0
    assert 0.2 < out["mean_item_std"] < 0.7


# ---- bias ------------------------------------------------------------------

def test_position_bias_detected():
    rng = np.random.default_rng(1)
    n = 2000
    pref = 0.7  # P(pick first slot) on ties
    ab = (rng.uniform(size=n) >= pref).astype(int)
    ba = (rng.uniform(size=n) >= pref).astype(int)
    out = position_bias(ab, ba)
    assert abs(out["first_pos_rate"] - pref) < 0.03


def test_position_bias_neutral():
    rng = np.random.default_rng(2)
    n = 2000
    # content winner fixed; slot flips correctly -> consistent, neutral
    content = rng.integers(0, 2, n)
    ab = content
    ba = 1 - content
    out = position_bias(ab, ba)
    assert out["consistency"] > 0.99
    assert abs(out["first_pos_rate"] - 0.5) < 0.05


def test_length_bias_recovered():
    rng = np.random.default_rng(3)
    n = 4000
    q = rng.uniform(1, 5, n)
    length = rng.lognormal(5, 0.5, n)
    zlen = (np.log(length) - np.log(length).mean()) / np.log(length).std()
    coef = 0.5
    score = q + coef * zlen + rng.normal(0, 0.3, n)
    out = length_bias(score, length, control=q)
    assert abs(out["length_coef_per_sd"] - coef) < 0.05
    assert out["t_stat"] > 5


# ---- drift -----------------------------------------------------------------

def test_ewma_catches_step():
    series = np.concatenate([np.full(8, 3.0), np.full(8, 4.0)])
    series = series + np.random.default_rng(4).normal(0, 0.05, 16)
    out = ewma_chart(series, lam=0.3, L=3.0)
    assert out["first_alarm"] is not None and 8 <= out["first_alarm"] <= 11


def test_ewma_stable_no_alarm():
    series = 3.0 + np.random.default_rng(5).normal(0, 0.05, 20)
    out = ewma_chart(series, lam=0.3, L=3.0)
    assert out["first_alarm"] is None


def test_cusum_catches_drift():
    series = np.concatenate([np.full(10, 3.0), np.linspace(3.1, 4.0, 10)])
    out = cusum(series, k=0.5, h=4.0)
    assert out["first_alarm"] is not None


# ---- calibration -----------------------------------------------------------

def test_isotonic_monotone_and_reduces_error():
    rng = np.random.default_rng(6)
    n = 3000
    target = rng.uniform(1, 5, n)
    raw = np.clip((target ** 1.4) / (5 ** 1.4) * 4 + 1 + rng.normal(0, 0.2, n), 1, 5)
    cut = n // 2
    cal = IsotonicCalibrator().fit(raw[:cut], target[:cut])
    corrected = cal.transform(raw[cut:])
    # monotone output
    xs = np.argsort(raw[cut:])
    assert np.all(np.diff(corrected[xs]) >= -1e-9)
    before = calibration_error(raw[cut:], target[cut:], scale=(1, 5))["ece"]
    after = calibration_error(corrected, target[cut:], scale=(1, 5))["ece"]
    assert after < before


def test_bootstrap_diff_null_not_significant():
    rng = np.random.default_rng(7)
    a = rng.normal(3, 1, 300)
    b = rng.normal(3, 1, 300)
    out = bootstrap_diff(a, b, seed=0)
    assert out["significant"] is False


def test_bootstrap_diff_real_gap_significant():
    rng = np.random.default_rng(8)
    a = rng.normal(3.5, 1, 600)
    b = rng.normal(3.0, 1, 600)
    out = bootstrap_diff(a, b, seed=0)
    assert out["significant"] is True


# ---- end-to-end inject -> recover -----------------------------------------

def test_benchmark_recovers_injected_biases():
    spec = JudgeSpec()
    bench = make_benchmark(spec)
    a = audit(bench)
    t = a["truth"]

    # verbosity coef recovered within tolerance
    rec = a["bias"]["verbosity"]["length_coef_per_sd"]
    assert abs(rec - t["verbosity_coef"]) < 0.12

    # position preference recovered
    assert abs(a["bias"]["position"]["first_pos_rate"] - t["first_pos_pref"]) < 0.05

    # drift alarm fires at/after the injected changepoint, not before
    alarm = a["drift"]["ewma"]["first_alarm"]
    assert alarm is not None and alarm >= t["drift_changepoint"] - 1

    # isotonic correction reduces calibration error
    assert a["calibration"]["after"]["ece"] < a["calibration"]["before"]["ece"]


def test_leaderboard_on_noise():
    tm = two_model_scores(true_gap=0.03)
    out = bootstrap_diff(tm["a"], tm["b"], seed=0)
    # a 0.03-pt true gap should not be declared significant
    assert out["significant"] is False
