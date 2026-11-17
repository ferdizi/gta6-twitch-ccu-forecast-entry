# gta6-twitch-ccu-forecast-entry

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ferdizi/gta6-twitch-ccu-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ferdizi/gta6-twitch-ccu-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour forecast of GTA VI's launch-day Twitch viewers (2026-11-19 UTC, hours 2–24), for [make.fun gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Model

`main.py` (Python 3.13, stdlib only, deterministic, no network) is an equal-weight ensemble of three comparables-based models, all trained at run time on the case's own comparables (past AAA launch-day curves):

1. **Per-hour ridge** — log-ratio ln(y_h / y_1) on [ln(p1/100k), ln(eve6mean/p1)], US-unlock (DST) shift-aligned, spread from training residuals (empirical quantiles shrunk 25% toward normal).
2. **One-factor** — evening amplitude A (mean log-ratio, hours 18–22) and dip D (hours 2–4) each regressed on launch features; the curve is A times a pooled normalized shape (median over peers); spread from leave-one-out residuals.
3. **Hour-group pooled ridge** — shared feature slopes within each hour group (2–5, 6–9, 10–13, 14–17, 18–24), per-hour intercepts, normal spread.

Peers are "clean" rolling-midnight launches only (peak/hour-1 < 20, no early-access contamination, hour-1 ≥ 10k viewers) — 19 peers per practice case. The key edge over the house persistence baseline: launch scale predicts the curve shape (corr(ln hour-1, ln peak/hour-1) = −0.76 — huge launches stay flat, small ones ramp), and the evening level is essentially one-factor (corr ≈ 0.98 with the peak).

## Local grade

Practice cases (ground truth recovered from the SullyGnome API documented in the kit's NOTICE.txt; the house baseline scores 0.3871 on the same answers, which matches the RULES figure exactly):

| case | loss |
|---|---|
| p00 Marvel's Wolverine | 0.201 |
| p01 Red Dead Redemption 2 | 0.138 |
| p02 God of War Ragnarök | 0.131 |
| p03 Marvel's Spider-Man 2 | 0.181 |
| **overall** | **0.163** |

58% better than the house persistence baseline. Interval coverage 0.96–1.00. Leave-one-out over the 19 clean peers: 0.251. Two runs produce byte-identical output; ~0.1 s per case (limit 8 s).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=qgr0hfw9&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
