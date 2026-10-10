# Success criteria (written before any modelling)

## 1. Prediction target
Predict `DeltaToQuali`: a race lap's time minus the same driver's best qualifying lap at the same event (seconds).
- Removes track length and car pace, leaving fuel, tyre age, compound and race conditions.
- Leakage-safe: qualifying happens before the race.
- Risks handled in Stage 3: wet qualifying; drivers without a qualifying time (fallback, e.g. team-mate's time).

## 2. Laps used for training and evaluation
Clean, dry, racing laps only. Excluded: lap 1; in-laps and out-laps; safety car, VSC and red-flag laps (plus the restart lap); laps marked inaccurate or with no time; intermediate and wet tyre laps. Yellow-flag laps: decided in Stage 3 using data.
Excluded laps are still used by the strategy simulator (safety-car pit discounts and frequency).

## 3. Data split (time-ordered, never random)
- Train: 2022–2023 seasons
- Validation: 2024 season (model comparison and tuning)
- Test: 2025 season (used once, at the end)
2026 (new regulations) is out of scope, optionally a later "does it transfer?" check.

## 4. Baselines (fitted on training data only)
- B0 constant: median `DeltaToQuali` of all training laps.
- B1 historical track + compound (main baseline): median `DeltaToQuali` per circuit and compound in training seasons; falls back to compound-only median for unseen circuits.

## 5. Metrics and targets
Lap-time model:
- Primary: MAE (seconds per lap). Secondary: RMSE.
- Strategy-relevant: stint total-time error (sum of predicted laps vs actual).
- Success: beat B1's validation MAE by at least 15%, and still beat B1 on the 2025 test set.

Strategy simulator:
- Back-test: on test races without safety cars, simulate each driver's actual strategy and compare predicted vs actual total race time (seconds and %).
- Plausibility: does the recommended number of stops match the majority of the top 10 finishers?
- Limitation: the counterfactual (what a different strategy would have done) cannot be observed, so we test plausibility, not proof.

## 6. Scope and rules
- Dry races only. Simulator enforces the rule that at least two different dry compounds must be used.
- Not modelled: traffic, overtaking, dirty air, team orders, tyre-set allocation.
- Fuel load is not published; lap number is used as a proxy.
