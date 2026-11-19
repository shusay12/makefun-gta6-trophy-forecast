"""gta6-trophy-rarity-forecast-01 entry: a local-trend forecaster for GTA VI
trophy earn rates on Exophase (Python 3.13, standard library only).

For every trophy in a case it writes five quantiles (5%, 25%, 50%, 75%, 95%)
of the trophy's earn rate at the case's target, scored by pinball loss on the
logit scale.

Model (chosen by multi-fold backtests on simulated launch dynamics, across
several checkpoint/horizon splits and page-refresh regimes; the recent-window
fit with pooled shrinkage sat on a wide, flat plateau of parameter space and
beat the persistence and 24-hour-curve house baselines in every fold):

  - Work in logit space on count-smoothed rates: a read's earners are
    rate * players_tracked / 100, and the smoothed rate is
    100 * (earners + 0.5) / (players_tracked + 1), so a trophy earned by a
    handful of players does not whipsaw the fit.
  - Fit, per trophy, a straight line of logit(rate) against ln(1 + hours
    since release) over the LAST 12 HOURS of reads before the checkpoint.
    Launch-week rates are strongly non-stationary (a dilution dip while the
    tracked-player count explodes, then progression-driven growth), so the
    recent local trend is the signal; older reads mostly mislead.
  - Shrink each trophy's slope toward the cross-trophy median slope, with
    shrinkage weight Sxx / (Sxx + 0.5): a trophy whose own recent reads pin
    its slope down keeps it, a noisy one borrows the field's trend. The
    line is re-anchored at the window's mean point with the shrunk slope.
  - The median is the shrunk line read at the target. The spread combines
    the house random-walk scale, 0.5 * sqrt(horizon in days) logits, with
    the fit's own prediction variance (residual variance times leverage at
    the target), so an extrapolated trophy honestly widens.
  - A trophy with too few usable reads in the window is fitted on its whole
    history instead; one that cannot be fitted at all falls back to
    persistence: its last smoothed logit, with the random-walk spread.
  - Safety guards: the median is capped at 3.5 logits from the trophy's
    last smoothed logit, and the emitted quantiles are clipped to the
    valid range by construction (inverse logit) and kept non-decreasing.

Deterministic: the output is a pure function of the case file. Runs in a
few milliseconds per case, far inside the 8-second limit.

The grader runs `python3 -P -s /bot/main.py` with only the case's file in
/input; locally, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

WINDOW_HOURS = 12.0   # reads used per trophy fit: the last 12 hours
SHRINK_K = 0.5        # slope shrinkage strength toward the pooled slope
RW_SIGMA = 0.5        # random-walk spread, logit units per sqrt(day)
MEDIAN_CAP = 3.5      # a median never lands further than this many logits
                      # from the trophy's last smoothed logit: the guard
                      # keeps a wild window slope from extrapolating a
                      # near-zero rate to an absurd one over a long horizon


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_time(case: dict) -> datetime:
    rd = (case.get("game") or {}).get("release_date") or "2026-11-19"
    return datetime.fromisoformat(rd).replace(tzinfo=timezone.utc)


def smoothed_logit(rate: float, players: int) -> float:
    """Logit of the count-smoothed earn rate for one read."""
    if players and players > 0:
        earners = rate * players / 100.0
        return logit(100.0 * (earners + 0.5) / (players + 1.0))
    return logit(rate)


def fit_line(xs: list[float], ys: list[float]):
    """Ordinary least squares y ~ a + b*x.

    Returns (a, b, resid_var, sxx, xbar, ybar) or None when degenerate.
    """
    n = len(xs)
    if n < 3:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    resid_var = sse / max(1, n - 2)
    return a, b, resid_var, sxx, xbar, ybar


def norm_z(tau: float) -> float:
    """The standard normal's quantile at tau: the exact constants for the
    contest's five levels, Acklam's approximation for any other level a
    case might ask for."""
    for level, z in zip(LEVELS, Z):
        if abs(tau - level) < 1e-9:
            return z
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if tau < plow:
        q = math.sqrt(-2 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if tau > phigh:
        q = math.sqrt(-2 * math.log(1 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def emit(median_logit: float, sigma: float, levels: list) -> list:
    sigma = max(0.0, min(4.0, sigma))
    qs = [round4(inv_logit(median_logit + norm_z(tau) * sigma)) for tau in levels]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict) -> dict:
    if not case["history"]:
        raise ValueError("case has no history")
    t0 = release_time(case)

    def hours(ts: str) -> float:
        return (parse_ts(ts) - t0).total_seconds() / 3600.0

    h_check = hours(case["at"])
    x_target = math.log(1.0 + max(hours(case["target"]), 0.0))
    d_days = case["horizon_minutes"] / 1440.0
    rw_var = (RW_SIGMA * math.sqrt(d_days)) ** 2

    levels = case.get("levels") or LEVELS

    # Per-trophy series: every usable read as (x, smoothed logit), the last
    # 12 hours of it, and the last smoothed logit.
    per_trophy = []
    for i in range(len(case["trophies"])):
        all_pts = []
        win_pts = []
        last_y = None
        for read in case["history"]:
            r = read["rates"][i]
            if r is None:
                continue
            h = hours(read["at"])
            y = smoothed_logit(r, read.get("players_tracked") or 0)
            last_y = y
            pt = (math.log(1.0 + max(h, 0.0)), y)
            all_pts.append(pt)
            if h >= h_check - WINDOW_HOURS:
                win_pts.append(pt)
        per_trophy.append((all_pts, win_pts, last_y))

    # First pass: fit every trophy (the window first; a trophy whose window
    # holds too few reads is fitted on its whole history instead), and
    # collect the slopes for the pooled median.
    fits = []
    slopes = []
    for all_pts, win_pts, _ in per_trophy:
        pts = win_pts if len(win_pts) >= 3 else all_pts
        fit = fit_line([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 3 else None
        fits.append((fit, pts))
        if fit is not None:
            slopes.append(fit[1])
    pooled = sorted(slopes)[len(slopes) // 2] if slopes else 0.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        _, _, last_y = per_trophy[i]
        fit, pts = fits[i]
        if fit is None or last_y is None:
            # Persistence fallback: last smoothed logit (or the raw last rate).
            if last_y is None:
                r = case["history"][-1]["rates"][i]
                last_y = logit(r if r is not None else 0.0)
            rows.append({"id": trophy["id"], "q": emit(last_y, math.sqrt(rw_var), levels)})
            continue
        _, b, resid_var, sxx, xbar, ybar = fit
        lam = sxx / (sxx + SHRINK_K)
        b_shrunk = lam * b + (1.0 - lam) * pooled
        median = ybar + b_shrunk * (x_target - xbar)
        median = last_y + max(-MEDIAN_CAP, min(MEDIAN_CAP, median - last_y))
        # Prediction variance at the target: resid_var * (1/n + (x*-xbar)^2/Sxx),
        # the leverage of the two-parameter fit.
        n = len(pts)
        leverage = 1.0 / n + (x_target - xbar) ** 2 / sxx
        sigma = math.sqrt(rw_var + resid_var * leverage)
        rows.append({"id": trophy["id"], "q": emit(median, sigma, levels)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
