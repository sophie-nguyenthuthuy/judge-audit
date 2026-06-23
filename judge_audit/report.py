"""Assemble a calibration/reliability report (Markdown + optional PNG plots).

One call -> a report you can drop into the data-reliability series: the three
audit axes, the before/after calibration, and bootstrap'd leaderboard gaps.

`audit()` is tolerant: it runs only the axes whose data is present, so it works
on a full synthetic benchmark AND on a partial real eval log (see io.assemble).
"""
from __future__ import annotations

import os
import numpy as np

from . import agreement, bias, calibrate, drift

__all__ = ["audit", "render_markdown", "save_plots"]


def _has(bench, key):
    v = bench.get(key)
    return v is not None and (not hasattr(v, "__len__") or len(v) > 0)


def audit(bench: dict, n_bins: int = 10) -> dict:
    """Run the audit on a benchmark/log dict. Only axes with the required keys
    are computed.

    Recognised keys: score (required), quality (human gold), length, repeats
    (n_items x R), anchor_means (per-batch), winner_ab/winner_ba (pairwise),
    scale (lo,hi). `spec.truth` is surfaced if present (synthetic path).
    """
    if not _has(bench, "score"):
        raise ValueError("bench must contain 'score'")
    s = np.asarray(bench["score"], dtype=float)
    scale = tuple(bench.get("scale", (float(np.nanmin(s)), float(np.nanmax(s)))))
    has_human = _has(bench, "quality")
    q = np.asarray(bench["quality"], dtype=float) if has_human else None

    a: dict = {"scale": scale, "n_items": int(len(s))}

    # --- axis 1: reliability ---
    rel = {}
    if _has(bench, "repeats") and np.asarray(bench["repeats"]).shape[1] >= 2:
        rel["self_consistency"] = agreement.self_consistency(
            np.asarray(bench["repeats"], dtype=float), level="ordinal")
    if has_human:
        rel["spearman_vs_human"] = agreement.spearman(s, q)
        rel["alpha_vs_human"] = agreement.krippendorff_alpha(
            np.vstack([s, q]), level="interval")
    if rel:
        a["reliability"] = rel

    # --- axis 2: systematic bias ---
    bz = {}
    if _has(bench, "winner_ab") and _has(bench, "winner_ba"):
        bz["position"] = bias.position_bias(bench["winner_ab"], bench["winner_ba"])
    if _has(bench, "length"):
        L = np.asarray(bench["length"], dtype=float)
        bz["verbosity"] = bias.length_bias(s, L, control=q if has_human else None)
    if bz:
        a["bias"] = bz

    # --- axis 3: drift on the anchor set ---
    if _has(bench, "anchor_means"):
        am = np.asarray(bench["anchor_means"], dtype=float)
        a["drift"] = {
            "anchor_means": am,
            "ewma": drift.ewma_chart(am, lam=0.3, L=3.0),
            "cusum": drift.cusum(am, k=0.5, h=4.0),
        }

    # --- calibration: needs human gold; fit on first half, test on second ---
    if has_human:
        mask = ~(np.isnan(s) | np.isnan(q))
        ss, qq = s[mask], q[mask]
        n = len(ss)
        idx = np.arange(n)
        np.random.default_rng(0).shuffle(idx)
        cut = n // 2
        cal_idx, test_idx = idx[:cut], idx[cut:]
        cal = calibrate.IsotonicCalibrator().fit(ss[cal_idx], qq[cal_idx])
        s_test, q_test = ss[test_idx], qq[test_idx]
        s_corr = cal.transform(s_test)
        a["calibration"] = {
            "before": calibrate.calibration_error(s_test, q_test, n_bins, scale),
            "after": calibrate.calibration_error(s_corr, q_test, n_bins, scale),
            "curve_before": calibrate.reliability_curve(s_test, q_test, n_bins, scale),
            "curve_after": calibrate.reliability_curve(s_corr, q_test, n_bins, scale),
            "calibrator": cal,
        }

    spec = bench.get("spec")
    if spec is not None and getattr(spec, "truth", None):
        a["truth"] = spec.truth
    return a


def _spark(values, lo=None, hi=None):
    blocks = "▁▂▃▄▅▆▇█"
    v = np.asarray(values, dtype=float)
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    if hi - lo < 1e-12:
        return blocks[0] * len(v)
    idx = np.clip(((v - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int),
                  0, len(blocks) - 1)
    return "".join(blocks[i] for i in idx)


def render_markdown(a: dict, plot_paths: dict | None = None) -> str:
    plot_paths = plot_paths or {}
    r = a.get("reliability", {})
    b = a.get("bias", {})
    d = a.get("drift")
    c = a.get("calibration")
    t = a.get("truth", {})

    L = ["# Judge audit report\n",
         "Treats the LLM judge as a measurement instrument and reports its "
         "reliability, systematic bias, temporal drift, and calibration.\n",
         f"_Audited {a.get('n_items', '?')} judgements; axes shown are those the "
         "data supports._\n",
         "## TL;DR\n"]

    if "self_consistency" in r:
        sc = r["self_consistency"]
        L.append(f"- **Self-consistency** (ceiling on everything): Krippendorff α "
                 f"= `{sc['alpha']:.3f}`, flip-rate `{sc['flip_rate']:.1%}`, mean "
                 f"item σ `{sc['mean_item_std']:.2f}` pts.")
    if "spearman_vs_human" in r:
        L.append(f"- **Validity vs human**: Spearman ρ = "
                 f"`{r['spearman_vs_human']:.3f}`, interval α = "
                 f"`{r['alpha_vs_human']:.3f}`.")
    if "position" in b:
        p = b["position"]
        L.append(f"- **Position bias**: first-slot rate `{p['first_pos_rate']:.1%}` "
                 f"(0.5 = neutral); order flips the verdict `{p['flip_rate']:.1%}` "
                 f"of the time.")
    if "verbosity" in b:
        v = b["verbosity"]
        L.append(f"- **Verbosity bias**: `{v['length_coef_per_sd']:+.3f}` pts per "
                 f"+1 SD length (t = {v['t_stat']:.1f}), quality partialled out.")
    if d is not None:
        fa = d["ewma"]["first_alarm"]
        cp = t.get("drift_changepoint")
        if fa is not None:
            L.append(f"- **Drift**: anchor-set EWMA alarms at **batch {fa}**"
                     + (f" (injected changepoint = {cp})." if cp is not None else "."))
        else:
            L.append("- **Drift**: no anchor-set alarm.")
    if c is not None:
        L.append(f"- **Calibration**: ECE `{c['before']['ece']:.3f}` → "
                 f"`{c['after']['ece']:.3f}` after isotonic correction "
                 f"(−{100*(1-c['after']['ece']/max(c['before']['ece'],1e-9)):.0f}%).")
    L.append("")

    if t:
        sc = r.get("self_consistency", {})
        fa = d["ewma"]["first_alarm"] if d is not None else "—"
        L.append("## Inject → recover check\n")
        L.append("| pathology | injected | recovered |")
        L.append("|---|---|---|")
        if "verbosity" in b:
            L.append(f"| verbosity coef (pts/SD) | {t.get('verbosity_coef','—')} | "
                     f"{b['verbosity']['length_coef_per_sd']:.3f} |")
        if sc:
            L.append(f"| self-consistency σ (pts) | {t.get('sigma_judge','—')} | "
                     f"{sc['mean_item_std']:.3f} |")
        if "position" in b:
            L.append(f"| first-slot preference | {t.get('first_pos_pref','—')} | "
                     f"{b['position']['first_pos_rate']:.3f} |")
        if d is not None:
            L.append(f"| drift changepoint (batch) | {t.get('drift_changepoint','—')} "
                     f"| {fa} |")
        L.append("")

    if d is not None:
        am = d["anchor_means"]
        L.append("## Axis 3 — drift on the anchor set\n")
        L.append(f"Anchor mean per batch: `{_spark(am)}`  "
                 f"(min {am.min():.2f} → max {am.max():.2f})")
        L.append(f"EWMA first alarm: **batch {d['ewma']['first_alarm']}**, "
                 f"CUSUM first alarm: **batch {d['cusum']['first_alarm']}**.\n")
        if "drift" in plot_paths:
            L.append(f"![drift]({plot_paths['drift']})\n")

    if c is not None:
        L.append("## Calibration — before vs after\n")
        L.append("Reliability diagram points (predicted → actual, normalized 0..1):\n")
        xb, yb = c["curve_before"]
        xa, ya = c["curve_after"]
        L.append("| bin | raw pred | raw actual | corrected pred | corrected actual |")
        L.append("|---|---|---|---|---|")
        for i in range(max(len(xb), len(xa))):
            rb = f"{xb[i]:.2f} | {yb[i]:.2f}" if i < len(xb) else "— | —"
            ra = f"{xa[i]:.2f} | {ya[i]:.2f}" if i < len(xa) else "— | —"
            L.append(f"| {i} | {rb} | {ra} |")
        L.append("")
        if "calibration" in plot_paths:
            L.append(f"![calibration]({plot_paths['calibration']})\n")

    L.append("## What to do about it (corrections)\n")
    L.append("1. **Position bias** → always score both orders; only count a "
             "verdict when the two orders agree, else mark *tie/uncertain*.")
    L.append("2. **Miscalibration** → ship the fitted isotonic map; report "
             "corrected scores + ECE, not raw judge scores.")
    L.append("3. **Drift** → keep the anchor-set EWMA/CUSUM gate in CI; block "
             "eval runs (or re-baseline) when it alarms.")
    L.append("4. **Reliability floor** → if self-consistency α is low, raise "
             "repeats / lower temperature before trusting any ranking.")
    L.append("5. **Error bars** → every leaderboard gap gets a bootstrap CI; "
             "if it straddles 0, report *no significant difference*.")
    L.append("")
    return "\n".join(L)


def save_plots(a: dict, outdir: str) -> dict:
    """Save drift + calibration PNGs if matplotlib is available AND the
    corresponding axes were computed. Returns {name: relpath}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    os.makedirs(outdir, exist_ok=True)
    paths = {}

    d = a.get("drift")
    if d is not None:
        am = d["anchor_means"]; ew = d["ewma"]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(am))
        ax.plot(x, am, "o-", color="#888", label="anchor mean", lw=1)
        ax.plot(x, ew["ewma"], "-", color="#1f77b4", label="EWMA", lw=2)
        ax.plot(x, ew["ucl"], "--", color="#d62728", lw=1, label="control limits")
        ax.plot(x, ew["lcl"], "--", color="#d62728", lw=1)
        if ew["first_alarm"] is not None:
            ax.axvline(ew["first_alarm"], color="#d62728", alpha=0.3)
        ax.set_title("Anchor-set drift (EWMA control chart)")
        ax.set_xlabel("batch"); ax.set_ylabel("judge score")
        ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(outdir, "drift.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        paths["drift"] = os.path.basename(p)

    c = a.get("calibration")
    if c is not None:
        xb, yb = c["curve_before"]; xa, ya = c["curve_after"]
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot([0, 1], [0, 1], ":", color="#aaa", label="perfect")
        ax.plot(xb, yb, "o-", color="#d62728", label=f"raw (ECE {c['before']['ece']:.3f})")
        ax.plot(xa, ya, "s-", color="#2ca02c", label=f"isotonic (ECE {c['after']['ece']:.3f})")
        ax.set_title("Reliability diagram")
        ax.set_xlabel("predicted (norm)"); ax.set_ylabel("actual (norm)")
        ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(outdir, "calibration.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        paths["calibration"] = os.path.basename(p)
    return paths
