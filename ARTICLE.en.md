# Your eval needs an eval: auditing LLM-as-judge reliability

*Part 4 — Data Reliability series. This time we don't measure the model; we measure the ruler.*

## 1. The problem: an eval number is a measurement, but nobody calibrates the ruler

LLM-as-judge has become the backbone of model evaluation: cheap, fast, scalable
to tens of thousands of samples. But a `7.2/10` from a judge **is a
measurement** — and like any measurement it is meaningless without two pieces
of evidence:

- **Reliability** — measure it twice, do you get the same answer?
- **Validity** — does it measure what you think it measures?

We subject *models* to tests, CI, and error bars, yet the *ruler* that scores
those models is trusted blindly. This piece turns the eval pipeline into an
object of audit — squarely in the spirit of the whole series: *if you can't
quantify the reliability of your data (or your instrument), every conclusion
downstream is built on sand.*

Shipping with the article: the `judge-audit` package (pure NumPy) and an
**inject → recover** benchmark — build a synthetic judge with *known* diseases,
then prove the tool *recovers* exactly those diseases. Same philosophy as
model-collapse-testbed: plant the signal, prove the instrument finds it.

## 2. Three axes

### Axis 1 — Reliability & Validity

**Self-consistency is the ceiling on everything.** Score the same item R times
at `temperature > 0`. If the judge can't agree with *itself*, every leaderboard
downstream is noise. Measure with Krippendorff's α (treating the R re-scores as
R raters) plus a flip-rate.

**Validity** = agreement with humans. Spearman ρ for ranking; Krippendorff's α
(interval) for how tightly the scales coincide. Krippendorff's α is the right
choice because it handles ordinal Likert scales, missing values, and any number
of raters — exactly the shape of eval data.

> Implementation note: the package's Krippendorff α matches the reference
> `krippendorff` package to four decimal places on the canonical example
> (nominal 0.7375 / ordinal 0.8495 / interval 0.8553). This is precisely where
> hand-rolled audit code tends to be silently wrong.

### Axis 2 — Systematic bias (measured by *controlled perturbation*)

Passive correlation is not enough — you must actively perturb the nuisance
variable:

- **Position bias** (pairwise): score the same pair in both orders, `(A,B)` and
  `(B,A)`. A neutral judge picks the same *content* regardless of slot. Report
  *flip-rate* (order changed the verdict) and *first-slot rate* (0.5 = neutral;
  > 0.5 = first-position favouritism).
- **Verbosity bias**: regress score on `z(log-length)` *after partialling out
  true quality*. A non-zero coefficient means the judge rewards length, not
  quality. Use log because token counts are heavy-tailed.

### Axis 3 — Temporal drift (the bridge to the Lyapunov / lyapmon work)

The provider silently changes model version; the prompt template gets edited;
temperature is nudged. Any of these *drifts* the judge's scale without touching
what it measures. To catch it:

1. Pin a **golden anchor set** — a few dozen items with stable human-consensus labels.
2. Re-score the anchor set **every batch**.
3. Run a **control chart (EWMA + CUSUM)** on the anchor mean. Crossing the
   control limit raises an alarm.

This is the familiar drift gate — just placed on the *eval* instead of the *model*.

## 3. Inject → recover: evidence the tool works

The synthetic judge is planted with 5 diseases of known magnitude. The audit
recovers:

| injected disease | injected | recovered |
|---|---|---|
| verbosity coef (pts / SD log-len) | 0.45 | 0.37¹ |
| self-consistency σ (pts) | 0.40 | 0.32 |
| first-slot preference | 0.68 | 0.71 |
| drift changepoint (batch) | 7 | 8² |
| calibration (ECE) | — | 0.177 → **0.042** after isotonic |

¹ Rounding to an integer Likert scale *attenuates* the measured coefficient — a
real, expected effect, and a lesson in itself: discretization hides bias.
² EWMA lags the changepoint by ~1 batch — the price of smoothing.

![drift](examples/out/drift.png)

The control chart catches the anchor mean breaching its limit at batch 8 (drift
starts at 7).

## 4. Calibration: pull judge scores onto the human scale

A judge can be *reliable* yet *miscalibrated* — consistently 0.6 points high, or
compressing the top of the scale. Fit **isotonic regression** (monotone) from
judge-score → human-score on a calibration split, then apply it to held-out data.

![calibration](examples/out/calibration.png)

ECE drops `0.177 → 0.042` (−76%). The red curve (raw) bows below the diagonal —
the judge compresses high scores; the green curve (isotonic) snaps onto the
diagonal.

## 5. Don't read noise as signal

Two models with a true quality gap of 0.05 points. The judge reports
`A − B = +0.030`, **95% CI `[-0.16, +0.21]`**. The CI straddles 0 ⇒ **no winner
can be called.** Every gap on a leaderboard needs a bootstrap CI; if it straddles
zero, report *"no significant difference"* rather than promoting it to a conclusion.

## 6. Five corrections to take to production

| problem | correction |
|---|---|
| position bias | score both orders; only count a verdict when they agree, else *tie/uncertain* |
| miscalibration | ship the isotonic map; report corrected scores + ECE, not raw |
| drift | keep the anchor-set control chart in CI; on alarm, block the run / re-baseline |
| low reliability | raise repeats / lower temperature before trusting any ranking |
| noisy leaderboards | bootstrap CI on every gap; straddles 0 ⇒ "no significant difference" |

## 7. Closing

The same toolkit as the rest of the series — drift monitors, anchor gates,
inject-then-recover, error bars — applied to a new object: the ruler, not the
thing measured. The one-line message: **your eval needs an eval.**

> Reproducible code + benchmark: `judge-audit` (pure NumPy). Run
> `python examples/run_benchmark.py` to regenerate the report and both figures,
> or open `examples/demo.ipynb` for the interactive walkthrough.
