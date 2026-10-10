"""
Download F1 race laps (2022–2025) with FastF1 and save one Parquet file per race.

Usage (from the project root, with the venv active):
    python src/download_data.py --test   # first 2 races of 2022 only
    python src/download_data.py          # all seasons

Safe to re-run: races already saved (with qualifying times) are skipped.
If FastF1's rate limit (500 calls/hour) is reached, the script waits and retries.
"""
import sys
import time
from pathlib import Path

import fastf1
import pandas as pd

YEARS = [2022, 2023, 2024, 2025]
RATE_LIMIT_WAIT_MIN = 60  # minutes to wait after hitting the hourly limit
MAX_RATE_LIMIT_WAITS = 3  # per race, before giving up on it

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "fastf1_cache"
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "races"
FAILED_LOG = PROJECT_ROOT / "data" / "raw" / "failed_races.txt"

# Lap columns we keep from FastF1 (originals, unchanged)
LAP_COLUMNS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "Stint",
    "LapTime", "LapStartTime", "PitInTime", "PitOutTime",
    "Compound", "TyreLife", "FreshTyre", "TrackStatus",
    "IsAccurate", "FastF1Generated", "Position", "Deleted",
]
# Weather at the time of each lap
WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]


def is_rate_limit(error):
    """True if FastF1 refused the request because of the hourly limit."""
    return type(error).__name__ == "RateLimitExceededError"


def wait_for_rate_limit():
    resume_at = time.strftime("%H:%M", time.localtime(time.time() + RATE_LIMIT_WAIT_MIN * 60))
    print(f"\n*** Rate limit reached (500 calls/hour). Waiting {RATE_LIMIT_WAIT_MIN} min, "
          f"resuming at about {resume_at}. Leave this window open. ***\n", flush=True)
    time.sleep(RATE_LIMIT_WAIT_MIN * 60)


def get_schedule(year):
    """Season calendar, waiting and retrying if the rate limit is hit."""
    while True:
        try:
            return fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            if is_rate_limit(e):
                wait_for_rate_limit()
            else:
                raise


def already_done(path):
    """A race counts as done only if its file exists AND it has some qualifying times."""
    if not path.exists():
        return False
    quali = pd.read_parquet(path, columns=["BestQualiSeconds"])
    return bool(quali["BestQualiSeconds"].notna().any())


def best_quali_times(year, rnd):
    """Each driver's best lap across Q1, Q2 and Q3, in seconds."""
    quali = fastf1.get_session(year, rnd, "Q")
    quali.load(laps=False, telemetry=False, weather=False, messages=False)
    res = quali.results[["Abbreviation", "Q1", "Q2", "Q3"]].copy()
    best = res[["Q1", "Q2", "Q3"]].apply(pd.to_timedelta).min(axis=1)
    res["BestQualiSeconds"] = best.dt.total_seconds()
    return res.rename(columns={"Abbreviation": "Driver"})[["Driver", "BestQualiSeconds"]]


def load_race(year, rnd):
    """Load one race and return (laps table, summary info for the progress line)."""
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps
    df = laps[LAP_COLUMNS].reset_index(drop=True)

    # Weather matched to each lap (same order and length as the laps)
    weather = laps.get_weather_data().reset_index(drop=True)
    assert len(weather) == len(df), "Weather rows don't match lap rows"
    df = pd.concat([df, weather[WEATHER_COLUMNS]], axis=1)

    # Race results: grid slot, finishing position, status (Finished / +1 Lap / DNF reason)
    results = session.results[["Abbreviation", "GridPosition", "Position", "Status"]]
    results = results.rename(columns={"Abbreviation": "Driver", "Position": "FinishPosition"})
    df = df.merge(results, on="Driver", how="left")

    # Qualifying reference for DeltaToQuali
    try:
        df = df.merge(best_quali_times(year, rnd), on="Driver", how="left")
    except Exception as e:
        if is_rate_limit(e):
            raise  # never save a race with blank quali just because of the rate limit
        print(f"   warning: no qualifying times ({type(e).__name__}); BestQualiSeconds left blank")
        df["BestQualiSeconds"] = float("nan")

    # Identify the race on every row
    df.insert(0, "Year", year)
    df.insert(1, "Round", rnd)
    df.insert(2, "EventName", session.event["EventName"])
    df.insert(3, "Location", session.event["Location"])

    # New column in seconds (original LapTime kept)
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    winner = results.sort_values("FinishPosition").iloc[0]["Driver"]
    info = {
        "event": session.event["EventName"],
        "drivers": df["Driver"].nunique(),
        "laps": int(df["LapNumber"].max()),
        "winner": winner,
        "quali_found": int(df.drop_duplicates("Driver")["BestQualiSeconds"].notna().sum()),
    }
    return df, info


def main():
    test_mode = "--test" in sys.argv

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level("WARNING")  # hide FastF1's INFO messages; we print our own progress

    failed = []
    start = time.time()

    for year in YEARS:
        schedule = get_schedule(year)
        rounds = [int(r) for r in schedule["RoundNumber"]]
        if test_mode:
            rounds = rounds[:2]
        print(f"\n=== {year}: {len(rounds)} races ===")

        for rnd in rounds:
            out_path = OUT_DIR / f"{year}_{rnd:02d}.parquet"
            if already_done(out_path):
                print(f"{year} R{rnd:02d} already saved, skipping")
                continue
            if out_path.exists():
                print(f"{year} R{rnd:02d} saved without qualifying times, downloading again")

            waits = 0
            while True:
                t0 = time.time()
                try:
                    df, info = load_race(year, rnd)
                    df.to_parquet(out_path, index=False)
                    print(
                        f"{year} R{rnd:02d} {info['event']:<28} | {info['drivers']} drivers | "
                        f"{info['laps']} laps | winner {info['winner']} | "
                        f"quali {info['quali_found']}/{info['drivers']} | {time.time() - t0:.0f}s",
                        flush=True,
                    )
                    break
                except Exception as e:
                    if is_rate_limit(e) and waits < MAX_RATE_LIMIT_WAITS:
                        waits += 1
                        wait_for_rate_limit()
                        continue  # retry the same race
                    msg = f"{year} R{rnd:02d}: {type(e).__name__}: {e}"
                    print("FAILED", msg, flush=True)
                    failed.append(msg)
                    break

        if test_mode:
            break  # test mode: only the first season

    saved = len(list(OUT_DIR.glob("*.parquet")))
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Race files saved in total: {saved}")

    if failed:
        FAILED_LOG.write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} race(s) failed (listed in {FAILED_LOG.name}):")
        for msg in failed:
            print("  ", msg)
        print("Re-run the same command to retry only the missing races.")
    else:
        if FAILED_LOG.exists():
            FAILED_LOG.unlink()
        print("No failures.")


if __name__ == "__main__":
    main()
