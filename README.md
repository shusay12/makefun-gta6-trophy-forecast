# GTA VI Trophy Rarity Forecaster

Entry for make.fun's **gta6-trophy-rarity-forecast-01**: forecast, for every
PlayStation 5 trophy of Grand Theft Auto VI, the percent of Exophase's
tracked players who will have earned it at the case's target (2026-11-26
00:00 UTC for the final), as five quantiles (5%, 25%, 50%, 75%, 95%),
scored by pinball loss on the logit scale.

## The program

`main.py` (Python 3.13, standard library only) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`):

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It is deterministic (same case in, same bytes out) and runs in a few
milliseconds per case, far inside the 8-second limit.

## The model

Launch-week earn rates are strongly non-stationary: the tracked-player
count explodes first (diluting every rate), then progression takes over
and rates climb along saturating curves. A single fit over the whole
history is dragged by the early dip; the recent local trend is the
signal. Per trophy:

1. **Count-smoothed logits.** Each read's rate becomes
   `100 * (earners + 0.5) / (players_tracked + 1)` before the logit, so
   trophies held by a handful of players do not whipsaw the fit.
2. **Recent-window line.** Ordinary least squares of logit(rate) against
   `ln(1 + hours since release)` over the **last 12 hours** of reads
   (whole history when the window holds fewer than 3 reads).
3. **Pooled slope shrinkage.** Each slope is shrunk toward the
   cross-trophy median slope with weight `Sxx / (Sxx + 0.5)`, and the
   line is re-anchored at the window's mean point — noisy trophies borrow
   the field's trend, well-pinned ones keep theirs.
4. **Median + honest spread.** The median is the shrunk line at the
   target, capped at 3.5 logits from the last smoothed logit. The spread
   combines the random-walk scale `0.5 * sqrt(horizon days)` with the
   fit's prediction variance (residual variance × leverage at the
   target), so far extrapolations widen.

## How it was chosen

Candidates (persistence, the house's 24-hour curve fit, all-history and
quadratic log-time fits, a saturating-exponential fit, and windowed fits
of 6–48 hours with and without shrinkage) were compared by backtest on
simulated launch dynamics — a cohort model seeded with the real
launch-night counts — across **five checkpoint/horizon folds**
(36→96h, 36→120h, 48→120h, 48→168h, 59.5→168h) and two page-refresh
regimes, then re-confirmed on fresh seeds. The shipped configuration
sits on a wide plateau (windows 8–12h and shrinkage 0.5–1.0 are within
noise of each other) and beat the house curve baseline by ~45% and
persistence by ~60% in every fold and regime. Full fold-by-fold numbers
are in BUILD-REPORT.md.
