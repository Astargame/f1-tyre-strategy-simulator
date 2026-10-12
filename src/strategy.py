"""Pit-stop strategy simulator built on the lap-time model.

A plan is a list of stints, e.g. [("MEDIUM", 20), ("HARD", 37)] for a 57-lap race.
The simulator predicts ONE grid of lap times per race (compound x tyre age x lap number),
then adds up each plan's laps from that grid, so thousands of plans can be compared quickly.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd

DRY = ["SOFT", "MEDIUM", "HARD"]


class StrategySimulator:
    """Rank pit-stop plans for one race by predicted total race time.

    model:          the saved lap-time pipeline (predicts DeltaToQuali)
    params:         models/strategy_params.json (pit losses, calibration, safety-car chances)
    input_columns:  the model's input columns (from the model's metadata file)
    calibrate:      add the measured wear calibration to every predicted lap
    max_calibration_age: tyres older than this use this age's correction, because data on
                    very old tyres only comes from stints where the tyres were lasting well
                    (selection bias), which doesn't apply to an average stint
    """

    def __init__(self, model, params, input_columns, calibrate=True, max_calibration_age=30):
        self.model = model
        self.params = params
        self.input_columns = list(input_columns)
        self.calibrate = calibrate
        self.max_calibration_age = max_calibration_age

    # ---------- ingredients ----------
    def pit_loss(self, circuit, condition="green"):
        """Time lost by one pit stop (s). condition: 'green', 'sc' or 'vsc'."""
        green = self.params["pit_loss_green_by_circuit"].get(circuit, self.params["pit_loss_green_default"])
        factor = {"green": 1.0,
                  "sc": self.params["pit_loss_factor_sc"],
                  "vsc": self.params["pit_loss_factor_vsc"]}[condition]
        return green * factor

    def calibration(self, compound, ages):
        """Correction (s) to add to predicted laps on this compound at these tyre ages."""
        ages = np.asarray(ages)
        if not self.calibrate:
            return np.zeros(len(ages))
        bands = np.array(self.params["wear_calibration_bands"])
        labels = self.params["wear_calibration_labels"]
        table = self.params["wear_calibration"].get(compound, {})
        capped = np.minimum(ages, self.max_calibration_age)
        idx = np.clip(np.searchsorted(bands, capped, side="left") - 1, 0, len(labels) - 1)
        return np.array([table.get(labels[i], 0.0) for i in idx])

    # ---------- the lap-time grid ----------
    def lap_grid(self, race):
        """Predicted lap time vs qualifying for every compound x tyre age x lap.

        race: dict with Circuit, RaceLaps, TrackTemp, AirTemp.
        Returns an array of shape (3 compounds, max tyre age, race laps).
        """
        n_laps = int(race["RaceLaps"])
        max_age = n_laps + 10
        ages = np.arange(1, max_age + 1)
        lap_numbers = np.arange(1, n_laps + 1)
        a, l = np.meshgrid(ages, lap_numbers, indexing="ij")

        parts = [pd.DataFrame({"Compound": comp, "Circuit": race["Circuit"],
                               "TyreAge": a.ravel().astype(float), "LapNo": l.ravel()})
                 for comp in DRY]
        g = pd.concat(parts, ignore_index=True)
        g["LapsRemaining"] = float(n_laps) - g["LapNo"]
        g["RaceProgress"] = g["LapNo"] / n_laps
        g["TrackTemp"] = float(race["TrackTemp"])
        g["AirTemp"] = float(race["AirTemp"])

        grid = self.model.predict(g[self.input_columns]).reshape(len(DRY), max_age, n_laps)
        if self.calibrate:
            for i, comp in enumerate(DRY):
                grid[i] += self.calibration(comp, ages)[:, None]
        return grid

    # ---------- evaluating plans ----------
    @staticmethod
    def _stint_time(grid, comp_idx, start_lap, n_laps, start_age):
        k = np.arange(n_laps)
        ages = np.minimum(start_age + 1 + k, grid.shape[1])   # TyreAge counts the current lap
        laps = start_lap + k                                    # lap numbers, starting at 1
        return grid[comp_idx, ages - 1, laps - 1].sum()

    def plan_time(self, grid, race, plan, start_age=0, stop_conditions=None):
        """Total predicted time (s, relative to qualifying pace) of a plan, including pit losses."""
        total, lap = 0.0, 1
        for i, (comp, n) in enumerate(plan):
            total += self._stint_time(grid, DRY.index(comp), lap, n, start_age if i == 0 else 0)
            lap += n
        conditions = stop_conditions or ["green"] * (len(plan) - 1)
        total += sum(self.pit_loss(race["Circuit"], c) for c in conditions)
        return total

    @staticmethod
    def candidate_plans(race_laps, max_stops=2, min_stint=5):
        """Every 1..max_stops plan using at least two different dry compounds."""
        plans = []
        for stops in range(1, max_stops + 1):
            for comps in product(DRY, repeat=stops + 1):
                if len(set(comps)) < 2:                      # the two-compound rule
                    continue
                for pits in combinations(range(min_stint, race_laps - min_stint + 1), stops):
                    lengths = np.diff((0,) + pits + (race_laps,))
                    if lengths.min() >= min_stint:
                        plans.append(tuple(zip(comps, lengths.tolist())))
        return plans

    def rank_plans(self, race, start_age=0, max_stops=2, min_stint=5, top=10):
        """All valid plans ranked by predicted total time, with each plan's gap to the best."""
        grid = self.lap_grid(race)
        rows = [{"Plan": p, "Stops": len(p) - 1, "Time": self.plan_time(grid, race, p, start_age)}
                for p in self.candidate_plans(int(race["RaceLaps"]), max_stops, min_stint)]
        ranked = pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)
        ranked["GapToBest"] = ranked["Time"] - ranked["Time"].iloc[0]
        ranked["Description"] = ranked["Plan"].apply(describe_plan)
        return ranked.head(top) if top else ranked


# ---------- helpers for real races ----------
def race_context(race_laps):
    """Circuit, distance and typical temperatures of one race, from its rows in laps_all_flags."""
    return {
        "Circuit": race_laps["Circuit"].iloc[0],
        "RaceLaps": int(race_laps["LapNumber"].max()),
        "TrackTemp": float(race_laps["TrackTemp"].median()),
        "AirTemp": float(race_laps["AirTemp"].median()),
    }


def actual_plan(driver_laps):
    """The stints one driver actually ran, and the tyre age they started the race on."""
    d = driver_laps.sort_values("LapNumber")
    d = d[d["StintNo"].notna()]
    stints = d.groupby("StintNo", sort=True).agg(
        Compound=("Compound", "first"), Laps=("LapNumber", "size"), FirstAge=("TyreAge", "first"))
    plan = [(str(c), int(n)) for c, n in zip(stints["Compound"], stints["Laps"])]
    first_age = stints["FirstAge"].iloc[0]
    start_age = max(int(first_age) - 1, 0) if pd.notna(first_age) else 0
    return plan, start_age


def describe_plan(plan):
    """Short text for a plan, e.g. 'M20 → H37 (pit lap 20)'."""
    parts = [f"{comp[0]}{n}" for comp, n in plan]
    pit_laps = np.cumsum([n for _, n in plan])[:-1]
    text = " → ".join(parts)
    return text + (f" (pit lap {', '.join(str(p) for p in pit_laps)})" if len(pit_laps) else "")


def plan_is_simulatable(plan, race_laps):
    """True if a plan uses only dry tyres and covers the full race distance."""
    return all(c in DRY for c, _ in plan) and sum(n for _, n in plan) == race_laps
