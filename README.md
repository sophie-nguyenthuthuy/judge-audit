# judge-audit

**Your eval is a measurement instrument. Audit it like one.**

LLM-as-judge has quietly become the backbone of model evaluation — but the
judge itself is rarely audited. `judge-audit` treats the judge as an instrument
and asks the questions you'd ask of any sensor before trusting its readout:

1. **Reliability** — does it give the same answer twice? (self-consistency, judge↔judge)
2. **Validity** — does it agree with humans? (Spearman, Krippendorff's α)
3. **Systematic bias** — position bias, verbosity bias, measured by *controlled perturbation*, not correlation.
4. **Drift** — does the scale move over time as the provider silently ships new model versions? (anchor-set EWMA / CUSUM control charts)
5. **Calibration** — map raw judge scores onto the human scale (isotonic), with honest bootstrap error bars on every leaderboard gap.

Pure NumPy, zero required deps beyond `numpy` (`matplotlib` optional for plots).

## Install

```bash
pip install -e .          # core
pip install -e ".[plot]"  # + matplotlib for PNG plots
pip install -e ".[dev]"   # + pytest
```

## 60-second demo

```bash
python examples/run_benchmark.py
# -> examples/out/report.md  (+ drift.png, calibration.png)
```

Or open the interactive walkthrough: [`examples/demo.ipynb`](examples/demo.ipynb).

The demo runs an **inject → recover** benchmark: a synthetic judge is built
with *known* pathologies, then the audit recovers them — the same "plant the
signal, prove the instrument finds it" design as model-collapse-testbed.

| pathology | injected | recovered |
|---|---|---|
| verbosity coef (pts / SD log-len) | 0.45 | 0.37¹ |
| self-consistency σ (pts) | 0.40 | 0.32 |
| first-slot preference | 0.68 | 0.71 |
| drift changepoint (batch) | 7 | 8² |
| calibration (ECE) | — | 0.177 → **0.042** after isotonic |

¹ integer-Likert rounding attenuates measured coefficients — an honest, expected effect.
² EWMA lags the changepoint by ~1 batch by design (that's the smoothing).

![drift](examples/out/drift.png)
![calibration](examples/out/calibration.png)

## Use on your own data

```python
import numpy as np
from judge_audit import (
    self_consistency, krippendorff_alpha, spearman,
    position_bias, length_bias, ewma_chart,
    IsotonicCalibrator, calibration_error, bootstrap_diff,
)

# 1. reliability ceiling: same judge, same items, scored R times
self_consistency(repeats)                      # repeats: (n_items, R)

# 2. validity vs human labels
krippendorff_alpha(np.vstack([judge, human]), level="interval")
spearman(judge, human)

# 3. systematic bias (controlled-perturbation inputs)
position_bias(winner_ab, winner_ba)            # order-swapped pairwise verdicts
length_bias(scores, lengths, control=human)    # verbosity, quality partialled out

# 4. drift on a fixed anchor set, one stat per batch
ewma_chart(anchor_mean_per_batch, lam=0.3, L=3.0)

# 5. calibrate + error bars
cal = IsotonicCalibrator().fit(judge_cal, human_cal)
corrected = cal.transform(judge_test)
calibration_error(corrected, human_test, scale=(1, 5))
bootstrap_diff(model_a_scores, model_b_scores) # CI straddles 0 -> no winner
```

Or run everything and render a report:

```python
from judge_audit import make_benchmark, audit, render_markdown, save_plots
a = audit(make_benchmark())          # or pass your own dict (see synthetic.py keys)
open("report.md", "w").write(render_markdown(a, save_plots(a, "out")))
```

## On your own eval logs (CSV / JSONL)

`io.assemble` turns a long-format log into the same dict; `audit()` then runs
**only the axes your columns support** (no human labels ⇒ no calibration; no
anchor set ⇒ no drift; etc.).

```python
from judge_audit import read_jsonl, assemble, pairwise, audit, render_markdown

recs  = read_jsonl("eval_log.jsonl")              # rows: item_id, score, human?, length?, batch?, repeat?
bench = assemble(recs, scale=(1, 5), anchor_ids=my_anchor_ids)
bench.update(pairwise(read_jsonl("pairwise_log.jsonl")))   # optional: pair_id, order, winner
print(render_markdown(audit(bench)))
```

Columns named differently? Pass a mapping: `assemble(recs, fields={"score": "rating", "human": "gold"})`.
A runnable end-to-end example: `python examples/make_sample_log.py && python examples/audit_log.py`.

## The five corrections

| problem | correction |
|---|---|
| position bias | score both orders; only count when they agree, else *tie/uncertain* |
| miscalibration | ship the isotonic map; report corrected scores + ECE, not raw |
| drift | keep the anchor-set control chart in CI; block/re-baseline on alarm |
| low reliability | raise repeats / lower temperature before trusting any ranking |
| noisy leaderboards | bootstrap CI on every gap; straddles 0 ⇒ "no significant difference" |

## Layout

```
judge_audit/
  agreement.py   Krippendorff α (nominal/ordinal/interval), Spearman, self-consistency
  bias.py        position bias (order-swap), verbosity bias (partialled OLS)
  drift.py       EWMA + CUSUM control charts
  calibrate.py   isotonic (PAVA), ECE / reliability curve, bootstrap CIs
  synthetic.py   judge with injected pathologies + ground truth
  io.py          CSV/JSONL adapters for real eval logs -> audit-ready dict
  report.py      run the full audit (tolerant of missing axes), render MD + plots
examples/
  run_benchmark.py   synthetic inject->recover report
  make_sample_log.py + audit_log.py   real-log adapter path
  demo.ipynb         interactive walkthrough
tests/           Krippendorff cross-checked against the reference package
```

See [`ARTICLE.md`](ARTICLE.md) (Vietnamese) / [`ARTICLE.en.md`](ARTICLE.en.md)
(English) for the write-up.

## License

MIT
