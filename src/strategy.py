"""Pit-stop strategy simulator built on the lap-time model, plus a model of teams' compound choices.

A plan is a list of stints, e.g. [("MEDIUM", 20), ("HARD", 37)] for a 57-lap race.
The simulator predicts ONE grid of lap times per race (compound x tyre age x lap number),
then adds up each plan's laps from that grid, so thousands of plans can be compared quickly.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DRY = ["SOFT", "MEDIUM", "HARD"]
LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}


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
        ranked["Sequence"] = ranked["Plan"].apply(plan_sequence)
        ranked["Description"] = ranked["Plan"].apply(describe_plan)
        return ranked.head(top) if top else ranked

    # ---------- safety cars and VSCs ----------
    def sample_neutralisations(self, circuit, race_laps, n_sims=200, seed=42, sc_laps=5, vsc_laps=3):
        """Random race scenarios. Each is a list of (start_lap, end_lap, 'sc' or 'vsc').

        Simplifications: at most one SC and one VSC per race, fixed lengths,
        start times drawn from the measured pattern by tenth of the race.
        """
        rng = np.random.default_rng(seed)
        p_sc = self.params["sc_prob_by_circuit"].get(circuit, self.params["sc_prob_default"])
        p_vsc = self.params["vsc_prob_by_circuit"].get(circuit, self.params["vsc_prob_default"])
        tenth_probs = np.array(self.params["sc_start_probs_by_tenth"], dtype=float)
        tenth_probs = tenth_probs / tenth_probs.sum()
        scenarios = []
        for _ in range(n_sims):
            events = []
            for kind, p, length in [("sc", p_sc, sc_laps), ("vsc", p_vsc, vsc_laps)]:
                if rng.random() < p:
                    tenth = rng.choice(10, p=tenth_probs)
                    start = int(np.clip(round((tenth + rng.random()) / 10 * race_laps), 1, race_laps - 1))
                    events.append((start, min(start + length - 1, race_laps), kind))
            scenarios.append(events)
        return scenarios

    @staticmethod
    def adapt_plan(plan, events, window=5, min_stint=5):
        """React like a strategist: if an SC/VSC starts up to `window` laps before a planned stop,
        pit under it instead. Returns (adjusted plan, condition of each stop)."""
        comps = [c for c, _ in plan]
        race_laps = sum(n for _, n in plan)
        pits = [int(p) for p in np.cumsum([n for _, n in plan])[:-1]]
        conditions = ["green"] * len(pits)
        for start, _end, kind in sorted(events):
            for i, p in enumerate(pits):
                if conditions[i] != "green" or not (start <= p <= start + window):
                    continue
                prev_pit = pits[i - 1] if i > 0 else 0
                next_pit = pits[i + 1] if i + 1 < len(pits) else race_laps
                if start - prev_pit >= min_stint and next_pit - start >= min_stint:
                    pits[i] = start
                    conditions[i] = kind
                break                                   # each SC/VSC can be used by one stop
        lengths = np.diff([0] + pits + [race_laps]).tolist()
        return list(zip(comps, lengths)), conditions

    def plan_time_in_scenario(self, grid, race, plan, events, start_age=0, window=5, min_stint=5):
        """Time of a plan in one SC/VSC scenario, after the strategist's reaction."""
        new_plan, conditions = self.adapt_plan(plan, events, window, min_stint)
        return self.plan_time(grid, race, new_plan, start_age, conditions)

    def rank_plans_with_safety_cars(self, race, start_age=0, per_sequence=8, n_sims=200,
                                    seed=42, window=5, max_stops=2, min_stint=5):
        """Re-rank the best plans of EVERY compound order by expected time over random SC/VSC scenarios."""
        ranked = self.rank_plans(race, start_age, max_stops, min_stint, top=None)
        cands = ranked.groupby(["Stops", "Sequence"]).head(per_sequence).reset_index(drop=True)
        grid = self.lap_grid(race)
        scenarios = self.sample_neutralisations(race["Circuit"], int(race["RaceLaps"]), n_sims, seed)
        cands["ExpectedTime"] = [
            float(np.mean([self.plan_time_in_scenario(grid, race, plan, ev, start_age, window, min_stint)
                           for ev in scenarios]))
            for plan in cands["Plan"]
        ]
        cands = cands.sort_values("ExpectedTime").reset_index(drop=True)
        cands["ExpectedGapToBest"] = cands["ExpectedTime"] - cands["ExpectedTime"].iloc[0]
        return cands


class TeamChoiceModel:
    """Predict the compound order top teams choose, from Pirelli's compound range and race facts.

    Step 1: a logistic regression predicts the starting compound (SOFT / MEDIUM / HARD).
    Step 2: the most common compound order in past races with that number of stops and that
            starting compound completes the plan.
    Teams know things our lap-time model can't see (practice long runs, tyre sets left),
    so their past choices are evidence about which compound order works.
    """

    def __init__(self, features, C=1.0):
        self.features = list(features)
        self.C = C

    def fit(self, choices):
        self.clf_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
        self.clf_.fit(choices[self.features], choices["StartCompound"])
        self.seq_by_stops_start_ = (choices.groupby(["Stops", "StartCompound"])["Sequence"]
                                    .agg(lambda s: s.value_counts().index[0]).to_dict())
        self.seq_by_stops_ = (choices.groupby("Stops")["Sequence"]
                              .agg(lambda s: s.value_counts().index[0]).to_dict())
        return self

    def predict_start(self, feats):
        X = pd.DataFrame([feats])[self.features]
        probs = dict(zip(self.clf_.classes_, self.clf_.predict_proba(X)[0]))
        return max(probs, key=probs.get), probs

    def predict_sequence(self, feats, stops):
        start, probs = self.predict_start(feats)
        seq = self.seq_by_stops_start_.get((stops, start)) or self.seq_by_stops_.get(stops)
        return seq, start, probs


def recommend(sim, team_model, race, feats, start_age=3, n_sims=200, per_sequence=8, seed=42):
    """Final recommendation: number of stops (safety-car-aware simulator) + compound order
    (team-choice model), using the best-timed plan with that order."""
    ranked = sim.rank_plans_with_safety_cars(race, start_age, per_sequence=per_sequence,
                                             n_sims=n_sims, seed=seed)
    fastest = ranked.iloc[0]
    stops = int(fastest["Stops"])
    seq, _start, probs = team_model.predict_sequence(feats, stops)
    match = ranked[(ranked["Stops"] == stops) & (ranked["Sequence"] == seq)]
    chosen = match.iloc[0] if len(match) else fastest
    return {"ranked": ranked, "fastest": fastest, "recommended": chosen, "stops": stops,
            "sequence": chosen["Sequence"], "start_probs": probs,
            "cost_vs_fastest": float(chosen["ExpectedTime"] - fastest["ExpectedTime"])}


# ---------- helpers for real races ----------
def race_features(race, nomination, params):
    """Features for the team-choice model: Pirelli's range (Hard, Medium, Soft C-numbers) + race facts."""
    hard, medium, soft = nomination
    return {
        "SoftC": soft, "HardC": hard,
        "HardMedGap": medium - hard, "MedSoftGap": soft - medium,   # 2 = Pirelli skipped a compound
        "RaceLaps": race["RaceLaps"], "TrackTemp": race["TrackTemp"],
        "PitLoss": params["pit_loss_green_by_circuit"].get(race["Circuit"], params["pit_loss_green_default"]),
    }


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


def plan_sequence(plan):
    """Compound order of a plan, e.g. 'S-H-H'."""
    return "-".join(LETTER.get(c, "?") for c, _ in plan)


def pit_laps(plan):
    """Laps at the end of which the driver pits, e.g. [17, 37]."""
    return [int(p) for p in np.cumsum([n for _, n in plan])[:-1]]


def describe_plan(plan):
    """Short text for a plan, e.g. 'M20 → H37 (pit lap 20)'."""
    parts = [f"{comp[0]}{n}" for comp, n in plan]
    laps = pit_laps(plan)
    text = " → ".join(parts)
    return text + (f" (pit lap {', '.join(str(p) for p in laps)})" if laps else "")


def plan_is_simulatable(plan, race_laps):
    """True if a plan uses only dry tyres and covers the full race distance."""
    return all(c in DRY for c, _ in plan) and sum(n for _, n in plan) == race_laps
