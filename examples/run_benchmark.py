"""Run the full inject->recover benchmark and write a calibration report.

    python examples/run_benchmark.py [outdir]

Produces <outdir>/report.md (+ drift.png, calibration.png if matplotlib).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from judge_audit import (
    JudgeSpec, make_benchmark, two_model_scores,
    audit, render_markdown, save_plots, bootstrap_diff,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(outdir, exist_ok=True)

    spec = JudgeSpec()
    bench = make_benchmark(spec)
    a = audit(bench)

    plots = save_plots(a, outdir)
    md = render_markdown(a, plot_paths=plots)

    # leaderboard-on-noise demo: two near-identical models
    tm = two_model_scores(spec, true_gap=0.05)
    diff = bootstrap_diff(tm["a"], tm["b"], paired=False, seed=1)
    md += (
        "\n## Leaderboard sanity — don't read noise as signal\n\n"
        f"Two models, true quality gap = {tm['true_gap']} pts. "
        f"Judge says A−B = `{diff['diff']:+.3f}` "
        f"(95% CI [{diff['lo']:+.3f}, {diff['hi']:+.3f}]). "
        f"**Significant: {diff['significant']}** — the CI "
        f"{'excludes' if diff['significant'] else 'straddles'} 0, so the "
        f"leaderboard {'can' if diff['significant'] else 'cannot'} call a winner.\n"
    )

    path = os.path.join(outdir, "report.md")
    with open(path, "w") as f:
        f.write(md)

    print(f"wrote {path}")
    if plots:
        print("plots:", ", ".join(os.path.join(outdir, p) for p in plots.values()))
    print("\n--- console summary ---")
    r = a["reliability"]; b = a["bias"]; d = a["drift"]; c = a["calibration"]
    print(f"self-consistency alpha : {r['self_consistency']['alpha']:.3f}")
    print(f"spearman vs human      : {r['spearman_vs_human']:.3f}")
    print(f"position first-slot     : {b['position']['first_pos_rate']:.3f} "
          f"(injected {a['truth'].get('first_pos_pref')})")
    print(f"verbosity coef/SD       : {b['verbosity']['length_coef_per_sd']:+.3f} "
          f"(injected {a['truth'].get('verbosity_coef')})")
    print(f"drift alarm batch       : {d['ewma']['first_alarm']} "
          f"(injected changepoint {a['truth'].get('drift_changepoint')})")
    print(f"ECE before -> after     : {c['before']['ece']:.3f} -> {c['after']['ece']:.3f}")


if __name__ == "__main__":
    main()
