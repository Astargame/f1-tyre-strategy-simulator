"""F1 Tyre Strategy Simulator — Streamlit app.

Run locally from the project folder (with the venv active):
    python -m streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))   # the saved models use classes from src/
from strategy import (DRY, StrategySimulator, describe_plan, pit_laps,  # noqa: E402
                      plan_is_simulatable, race_features, recommend)

APP_DATA = ROOT / "app" / "app_data"
MODELS = ROOT / "models"
COLOURS = {"SOFT": "#d62728", "MEDIUM": "#e6b800", "HARD": "#7f7f7f"}

# Races where special rules changed strategy (the simulator doesn't model race-specific rules)
ONE_OFF_RULES = {
    (2023, "Lusail"): "Pirelli limited each set of tyres to 18 laps, so drivers pushed flat out and made extra stops.",
    (2024, "Monaco"): "A lap-1 red flag let everyone change tyres for free, and almost nobody stopped again.",
    (2025, "Monaco"): "Two pit stops were mandatory at Monaco in 2025.",
    (2025, "Lusail"): "Pirelli limited how many laps each set of tyres could run, forcing extra stops.",
}
DATA_ISSUES = {
    (2025, "Montréal"): "FastF1's stint numbers look wrong for this race (e.g. M–H–H–H–H–H), "
                        "so 'actual' plans may show stops that didn't happen.",
}

st.set_page_config(page_title="F1 Tyre Strategy Simulator", page_icon="🏎️", layout="wide")


# ---------- loading (cached, so it only happens once) ----------
@st.cache_resource
def load_models():
    model = joblib.load(MODELS / "lap_time_model.joblib")
    meta = json.loads((MODELS / "lap_time_model_meta.json").read_text())
    params = json.loads((MODELS / "strategy_params.json").read_text())
    team = joblib.load(MODELS / "team_choice_model.joblib")
    sim = StrategySimulator(model, params, meta["input_columns"], calibrate=True)
    seen_circuits = set(model.named_steps["terms"].circuits_)
    return model, meta, params, team, sim, seen_circuits


@st.cache_data
def load_data():
    races = pd.read_parquet(APP_DATA / "races.parquet")
    drivers = pd.read_parquet(APP_DATA / "drivers.parquet")
    laps = pd.read_parquet(APP_DATA / "laps.parquet")
    return races, drivers, laps


def parse_plan(text):
    """'MEDIUM:26|HARD:30' -> [('MEDIUM', 26), ('HARD', 30)]"""
    plan = []
    for part in str(text).split("|"):
        comp, n = part.rsplit(":", 1)
        plan.append((comp, int(n)))
    return plan


def race_ctx(r):
    return {"Circuit": r["Circuit"], "RaceLaps": int(r["RaceLaps"]),
            "TrackTemp": float(r["TrackTemp"]), "AirTemp": float(r["AirTemp"])}


def get_race_row(year, rnd):
    races, _, _ = load_data()
    return races[(races["Year"] == year) & (races["Round"] == rnd)].iloc[0]


@st.cache_data(show_spinner="Simulating every 1–2 stop plan across 200 safety-car scenarios…")
def get_recommendation(year, rnd):
    _, _, params, team, sim, _ = load_models()
    r = get_race_row(year, rnd)
    ctx = race_ctx(r)
    feats = race_features(ctx, (int(r["HardC"]), int(r["MediumC"]), int(r["SoftC"])), params)
    rec = recommend(sim, team, ctx, feats, start_age=3, n_sims=200)
    alts = rec["ranked"][["Description", "Stops", "Sequence", "ExpectedGapToBest"]].head(8).copy()
    return {"plan": tuple(rec["recommended"]["Plan"]), "description": rec["recommended"]["Description"],
            "fastest": rec["fastest"]["Description"], "stops": rec["stops"], "cost": rec["cost_vs_fastest"],
            "start_probs": rec["start_probs"], "alternatives": alts}


@st.cache_data
def get_grid(year, rnd):
    _, _, _, _, sim, _ = load_models()
    return sim.lap_grid(race_ctx(get_race_row(year, rnd)))


# ---------- page ----------
model, meta, params, team, sim, seen_circuits = load_models()
races, drivers, laps = load_data()

st.title("🏎️ F1 Tyre Strategy Simulator")
st.caption("Predicts how lap times change as tyres wear and fuel burns off, then searches every 1–2 stop "
           "strategy, including random safety cars. Built on official F1 timing data (FastF1), 2022–2025 dry races.")

with st.sidebar:
    st.header("Choose a race")
    year = int(st.selectbox("Season", sorted(races["Year"].unique(), reverse=True)))
    season = races[races["Year"] == year].sort_values("Round")
    event = st.selectbox("Race", season["EventName"].tolist())
    r = season[season["EventName"] == event].iloc[0]
    rnd = int(r["Round"])
    field = (drivers[(drivers["Year"] == year) & (drivers["Round"] == rnd)]
             .sort_values("FinishPosition", na_position="last"))
    labels = [f"P{int(p)}  {d} ({t})" if pd.notna(p) else f"—  {d} ({t})"
              for d, p, t in zip(field["Driver"], field["FinishPosition"], field["Team"])]
    pick = st.selectbox("Driver", labels)
    drv = field.iloc[labels.index(pick)]
    st.divider()
    if year == 2025:
        st.success("2025 is the **test season**: none of it was used to build the model.")
    else:
        st.info(f"{year} was used to build the model.")

ctx = race_ctx(r)
circuit = r["Circuit"]
st.subheader(f"{year} {event}")

cols = st.columns(4)
cols[0].metric("Race laps", ctx["RaceLaps"])
cols[1].metric("Pirelli range (hard · medium · soft)", f"C{int(r['HardC'])} · C{int(r['MediumC'])} · C{int(r['SoftC'])}")
cols[2].metric("Pit-stop time loss", f"{sim.pit_loss(circuit):.1f} s")
cols[3].metric("Chance of a safety car", f"{params['sc_prob_by_circuit'].get(circuit, params['sc_prob_default']):.0%}")

if circuit not in seen_circuits:
    st.warning(f"**{circuit} wasn't in the model's training races**, so predictions here use "
               "average-track behaviour and are less reliable.")
if (year, circuit) in ONE_OFF_RULES:
    st.warning("**One-off rule:** " + ONE_OFF_RULES[(year, circuit)])
if (year, circuit) in DATA_ISSUES:
    st.warning("**Data issue:** " + DATA_ISSUES[(year, circuit)])
if bool(r["WetQuali"]):
    st.warning("**Qualifying was wet**, so absolute lap times are off (they're measured against a wet "
               "qualifying lap). Comparing strategies within this race is still meaningful.")

# ---------- recommendation ----------
rec = get_recommendation(year, rnd)
left, right = st.columns([3, 2])
with left:
    st.markdown("#### Recommended strategy")
    st.markdown(f"### {rec['description']}")
    st.caption("Number of stops and pit laps: lap-time model + 200 random safety-car scenarios. "
               "Compound order: what top-10 teams chose in past races with similar Pirelli ranges.")
    if rec["fastest"] != rec["description"]:
        st.info(f"The lap-time model on its own slightly prefers **{rec['fastest']}** "
                f"({rec['cost']:.1f} s faster on average). Its compound choices are much less reliable than "
                "team behaviour (start tyre right in 0% vs 90% of 2025 races), so the teams' order is shown.")
with right:
    st.markdown("#### Likely starting tyre (top-10 teams)")
    probs = pd.Series(rec["start_probs"]).reindex(DRY).fillna(0.0)
    probs.index = [c.title() for c in probs.index]
    st.bar_chart(probs)

with st.expander("Other strong options (average over 200 safety-car scenarios)"):
    alts = rec["alternatives"].rename(columns={"Description": "Plan", "Sequence": "Compounds",
                                               "ExpectedGapToBest": "Slower than best (s)"})
    st.dataframe(alts.round(1), hide_index=True)

# ---------- the driver's actual race vs the recommendation ----------
st.markdown(f"#### {drv['Driver']}'s actual race vs the recommendation")
plan = parse_plan(drv["PlanText"])
rec_plan = list(rec["plan"])
if plan_is_simulatable(plan, ctx["RaceLaps"]):
    grid = get_grid(year, rnd)
    start_age = int(drv["StartAge"])

    # Compare both plans on the SAME basis: the same 200 safety-car scenarios the recommendation used (same seed)
    scenarios = sim.sample_neutralisations(circuit, ctx["RaceLaps"], n_sims=200, seed=42)
    exp_actual = np.mean([sim.plan_time_in_scenario(grid, ctx, plan, ev, start_age) for ev in scenarios])
    exp_rec = np.mean([sim.plan_time_in_scenario(grid, ctx, rec_plan, ev, 3) for ev in scenarios])
    clean_actual = sim.plan_time(grid, ctx, plan, start_age)
    clean_rec = sim.plan_time(grid, ctx, rec_plan, 3)

    a, b, c = st.columns(3)
    a.metric(f"{drv['Driver']}'s strategy", describe_plan(plan))
    b.metric("vs recommendation — average with safety cars", f"{exp_actual - exp_rec:+.1f} s")
    c.metric("vs recommendation — if no safety car", f"{clean_actual - clean_rec:+.1f} s")
    st.caption("Positive = the driver's strategy is predicted to be slower. The recommendation is chosen on the "
               "safety-car average, so with no safety car a different plan can come out ahead. Differences under "
               "about 10 s are within the model's uncertainty (2025 check: 11 s average error).")

    d = laps[(laps["Year"] == year) & (laps["Round"] == rnd) & (laps["Driver"] == drv["Driver"])].sort_values("LapNumber")
    qref = float(drv["QualiRef"])

    def predicted_stints(p, first_age):
        """Model lap times for each stint of plan p: yields (compound, lap numbers, lap times)."""
        lap0 = 1
        for i, (comp, n) in enumerate(p):
            k = np.arange(n)
            ages = np.minimum((first_age if i == 0 else 0) + 1 + k, grid.shape[1])
            lap_numbers = lap0 + k
            yield comp, lap_numbers, grid[DRY.index(comp), ages - 1, lap_numbers - 1] + qref
            lap0 += n

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for lap in d.loc[d["IsNeutralised"].astype(bool), "LapNumber"]:
        ax.axvspan(lap - 0.5, lap + 0.5, color="gold", alpha=0.25, lw=0)
    ax.scatter(d["LapNumber"], d["LapTimeSeconds"], s=12, color="black", alpha=0.6, label="Actual lap", zorder=3)
    for comp, xs, ys in predicted_stints(plan, start_age):
        ax.plot(xs, ys, color=COLOURS[comp], lw=2.5, label=f"{drv['Driver']}'s strategy (model): {comp.title()}")
    for comp, xs, ys in predicted_stints(rec_plan, 3):
        ax.plot(xs, ys, color=COLOURS[comp], lw=1.8, ls=":", label=f"Recommended (model): {comp.title()}")
    for p in pit_laps(plan):
        ax.axvline(p + 0.5, color="black", ls="--", alpha=0.45)
    for p in pit_laps(rec_plan):
        ax.axvline(p + 0.5, color="tab:blue", ls=":", alpha=0.7)

    clean = d.loc[d["IsClean"].astype(bool), "LapTimeSeconds"]
    if len(clean):
        ax.set_ylim(clean.min() - 1.5, clean.quantile(0.98) + 3)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title(f"Solid: {drv['Driver']}'s strategy · dotted: recommended · dots: actual laps · "
                 "black dashes / blue dots: pit stops · yellow: safety car / VSC", fontsize=10)
    handles, names = ax.get_legend_handles_labels()
    unique = dict(zip(names, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    st.pyplot(fig)
else:
    st.warning(f"{drv['Driver']}'s race can't be simulated: they didn't complete the full distance on dry "
               f"tyres ({describe_plan(plan)}).")
    
# ---------- about ----------
with st.expander("How it works, how accurate it is, and its limits"):
    st.markdown(f"""
**Lap-time model.** A Ridge regression with tyre-wear terms for each compound and circuit, trained on
{meta['train_laps']:,} clean dry laps (2022–2024). It predicts how much slower each race lap is than the
driver's own qualifying lap. On the unseen 2025 season: **{meta['test_2025']['MAE']:.2f} s** average error
per lap vs **{meta['test_2025_B1_MAE']:.2f} s** for a track + compound baseline (−15.8%).

**Strategy.** Every legal 1–2 stop plan is scored, then re-ranked over 200 random safety-car / VSC scenarios
(pit stops under a safety car cost about 64% of a normal stop). The compound order comes from a model of what
top-10 teams chose, given Pirelli's compound range.

**2025 back-test (20 dry races).** Number of stops matched the top 10 in 60% of races (baselines: 30–35%);
starting tyre in 90% (lap-time model alone: 0%).

**Limits.** Dry races only · 1–2 stops only · no traffic, tyre management or team orders · race-specific
rules aren't modelled · teams prefer a 1-stop when it's close, the model doesn't · at most one safety car and
one VSC per simulated race.
""")
st.caption("Built by Brandon Richardson · [Code on GitHub](https://github.com/Astargame/f1-tyre-strategy-simulator) · Data: FastF1, Pirelli press releases")

