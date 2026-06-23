# Judge audit report

Treats the LLM judge as a measurement instrument and reports its reliability, systematic bias, temporal drift, and calibration.

## TL;DR

- **Self-consistency** (ceiling on everything): Krippendorff α = `0.847`, flip-rate `68.3%`, mean item σ `0.32` pts.
- **Validity vs human**: Spearman ρ = `0.795`, interval α = `0.650`.
- **Position bias**: first-slot rate `70.5%` (0.5 = neutral); order flips the verdict `58.3%` of the time.
- **Verbosity bias**: `+0.366` pts per +1 SD length (t = 22.6), after controlling for true quality.
- **Drift**: anchor-set EWMA alarms at **batch 8** (injected changepoint = 7).
- **Calibration**: ECE `0.177` → `0.042` after isotonic correction (−76%).

## Inject → recover check

| pathology | injected | recovered |
|---|---|---|
| verbosity coef (pts/SD) | 0.45 | 0.366 |
| self-consistency σ (pts) | 0.4 | 0.318 |
| first-slot preference | 0.68 | 0.705 |
| drift changepoint (batch) | 7 | 8 |

## Axis 3 — drift on the anchor set

Anchor mean per batch: `▃▂▃▃▂▁▁▅█▇▇▇`  (min 3.20 → max 4.10)
EWMA first alarm: **batch 8**, CUSUM first alarm: **batch 8**.

![drift](drift.png)

## Calibration — before vs after

Reliability diagram points (predicted → actual, normalized 0..1):

| bin | raw pred | raw actual | corrected pred | corrected actual |
|---|---|---|---|---|
| 0 | 0.00 | 0.08 | 0.09 | 0.08 |
| 1 | 0.25 | 0.19 | 0.16 | 0.19 |
| 2 | 0.50 | 0.33 | 0.39 | 0.33 |
| 3 | 0.75 | 0.56 | 0.60 | 0.56 |
| 4 | 1.00 | 0.76 | 0.80 | 0.76 |

![calibration](calibration.png)

## What to do about it (corrections)

1. **Position bias** → always score both orders; only count a verdict when the two orders agree, else mark *tie/uncertain*.
2. **Miscalibration** → ship the fitted isotonic map; report corrected scores + ECE, not raw judge scores.
3. **Drift** → keep the anchor-set EWMA/CUSUM gate in CI; block eval runs (or re-baseline) when it alarms.
4. **Reliability floor** → if self-consistency α is low, raise repeats / lower temperature before trusting any ranking.
5. **Error bars** → every leaderboard gap gets a bootstrap CI; if it straddles 0, report *no significant difference*.

## Leaderboard sanity — don't read noise as signal

Two models, true quality gap = 0.05 pts. Judge says A−B = `+0.030` (95% CI [-0.160, +0.207]). **Significant: False** — the CI straddles 0, so the leaderboard cannot call a winner.
