"""
Download qualifying lap data (2022–2025) and save one small Parquet summary per event:
each driver's best qualifying lap, the compound it was set on, and whether qualifying was wet.

Usage (from the project root, with the venv active):
    python src/download_quali.py --test   # first 2 events of 2022 only
    python src/download_quali.py          # all seasons

Safe to re-run: events already saved are skipped. Waits if FastF1's rate limit is hit.
"""
import sys
import time

import fastf1
import pandas as pd

# Reuse settings and rate-limit helpers from the race download script
from download_data import (
    YEARS, CACHE_DIR, PROJECT_ROOT, MAX_RATE_LIMIT_WAITS,
    is_rate_limit, wait_for_rate_limit, get_schedule,
)

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "quali"
FAILED_LOG = PROJECT_ROOT / "data" / "raw" / "failed_quali.txt"
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]


def load_quali(year, rnd):
    """Best qualifying lap per driver, plus session-level wet indicators."""
    session = fastf1.get_session(year, rnd, "Q")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = session.laps

    # Did the weather station record rain at any point in qualifying?
    weather = session.weather_data
    any_rain = False
    if weather is not None and len(weather) > 0:
        any_rain = bool(weather["Rainfall"].fillna(False).astype(bool).any())

    # Timed laps only, excluding laps deleted for track limits
    timed = laps[laps["LapTime"].notna() & ~laps["Deleted"].eq(True)].copy()
    timed["LapTimeSeconds"] = timed["LapTime"].dt.total_seconds()

    # Integrity check: on a real lap the three sector times add up to the lap time.
    # Timing-feed glitches can create impossible "laps" (e.g. 73.670 s at Jeddah 2022).
    sector_sum = (
        timed["Sector1Time"] + timed["Sector2Time"] + timed["Sector3Time"]
    ).dt.total_seconds()
    timed = timed[(sector_sum - timed["LapTimeSeconds"]).abs() < 0.1]

    # Share of valid timed laps set on intermediate or wet tyres
    timed["IsWetTyre"] = timed["Compound"].isin(WET_COMPOUNDS)
    wet_share = float(timed["IsWetTyre"].mean()) if len(timed) else float("nan")

    # Each driver's fastest valid lap, and the compound it was set on
    best_idx = timed.groupby("Driver")["LapTimeSeconds"].idxmin()
    best = timed.loc[best_idx, ["Driver", "Team", "LapTimeSeconds", "Compound"]].rename(
        columns={"LapTimeSeconds": "BestQualiLapSeconds", "Compound": "BestQualiCompound"}
    )

    # Session-level wet indicators, repeated on every driver row
    best["QualiWetTyreShare"] = wet_share
    best["QualiRainfall"] = any_rain

    best.insert(0, "Year", year)
    best.insert(1, "Round", rnd)
    best.insert(2, "EventName", session.event["EventName"])
    best = best.reset_index(drop=True)

    fastest = best.sort_values("BestQualiLapSeconds").iloc[0]
    info = {
        "event": session.event["EventName"],
        "drivers": len(best),
        "fastest": f"{fastest['Driver']} {fastest['BestQualiLapSeconds']:.3f}",
        "wet_share": wet_share,
        "rain": any_rain,
    }
    return best, info


def main():
    test_mode = "--test" in sys.argv

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level("WARNING")

    failed = []
    start = time.time()

    for year in YEARS:
        schedule = get_schedule(year)
        rounds = [int(r) for r in schedule["RoundNumber"]]
        if test_mode:
            rounds = rounds[:2]
        print(f"\n=== {year}: {len(rounds)} qualifying sessions ===")

        for rnd in rounds:
            out_path = OUT_DIR / f"{year}_{rnd:02d}_quali.parquet"
            if out_path.exists():
                print(f"{year} R{rnd:02d} already saved, skipping")
                continue

            waits = 0
            while True:
                t0 = time.time()
                try:
                    best, info = load_quali(year, rnd)
                    best.to_parquet(out_path, index=False)
                    print(
                        f"{year} R{rnd:02d} {info['event']:<28} | {info['drivers']} drivers | "
                        f"fastest {info['fastest']} | wet-tyre laps {info['wet_share']:.0%} | "
                        f"rain {info['rain']} | {time.time() - t0:.0f}s",
                        flush=True,
                    )
                    break
                except Exception as e:
                    if is_rate_limit(e) and waits < MAX_RATE_LIMIT_WAITS:
                        waits += 1
                        wait_for_rate_limit()
                        continue
                    msg = f"{year} R{rnd:02d}: {type(e).__name__}: {e}"
                    print("FAILED", msg, flush=True)
                    failed.append(msg)
                    break

        if test_mode:
            break

    saved = len(list(OUT_DIR.glob("*.parquet")))
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Qualifying files saved in total: {saved}")

    if failed:
        FAILED_LOG.write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} session(s) failed (listed in {FAILED_LOG.name}):")
        for msg in failed:
            print("  ", msg)
        print("Re-run the same command to retry only the missing sessions.")
    else:
        if FAILED_LOG.exists():
            FAILED_LOG.unlink()
        print("No failures.")


if __name__ == "__main__":
    main()
