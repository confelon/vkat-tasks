"""Collects daily historical weather for a city into yearly CSV files.

Weather is strongly seasonal, so we collect 10 years of daily data by
default. Data is fetched year by year, newest first, one CSV per year in
app/data/raw/ (e.g. Almaty-2026.csv). Years that are already fully
fetched are skipped, so re-running the script only downloads what is
missing — increase --years to extend history further back.

The city is looked up in the CITIES dict (European and ex-USSR capitals);
pass --lat/--lon to collect for a place that is not in the dict.

Run inside the container:
    docker compose run --rm weather python -m app.collector [--city Moscow] [--years 10]
    docker compose run --rm weather python -m app.collector --city Karaganda --lat 49.8 --lon 73.1
"""

import argparse
import csv
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from .cities import CITIES
from .openmeteo import COLUMNS, OpenMeteoProvider

RAW_DIR = Path(__file__).parent / "data" / "raw"


class WeatherCollector:
    """Fetches one CSV per year, newest first, skipping years already on disk."""

    def __init__(self, provider: OpenMeteoProvider, city: str, raw_dir: Path) -> None:
        self.provider = provider
        self.city = city
        self.raw_dir = raw_dir

    def collect(self, years: int) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        current_year = date.today().year
        for year in range(current_year, current_year - years, -1):
            self.collect_year(year)

    def collect_year(self, year: int) -> None:
        year_start = date(year, 1, 1)
        # Data for today is still incomplete, so only fetch up to yesterday
        last_day = min(date(year, 12, 31), date.today() - timedelta(days=1))
        if last_day < year_start:
            return
        expected_days = (last_day - year_start).days + 1

        out_file = self.raw_dir / f"{self.city}-{year}.csv"
        if self.already_fetched(out_file, expected_days):
            print(f"{out_file.name}: complete, skipping")
            return

        rows = self.provider.fetch_range(year_start, last_day)
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        print(f"{out_file.name}: fetched {len(rows)} days")

    def already_fetched(self, out_file: Path, expected_days: int) -> bool:
        if not out_file.exists():
            return False
        data_rows = len(out_file.read_text().splitlines()) - 1
        return data_rows >= expected_days


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical weather for a city")
    parser.add_argument("--city", default="Moscow", help="city from the CITIES dict, a label for --lat/--lon, or 'all'")
    parser.add_argument("--lat", type=float, help="latitude, overrides the CITIES lookup (requires --lon)")
    parser.add_argument("--lon", type=float, help="longitude, overrides the CITIES lookup (requires --lat)")
    parser.add_argument("--years", type=int, default=10, help="how many years back to fetch")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        cities = {args.city: (args.lat, args.lon)}
    elif args.city == "all":
        cities = CITIES
    else:
        cities = {args.city: CITIES[args.city]}

    for city, (latitude, longitude) in cities.items():
        print(f"=== {city} ===")
        provider = OpenMeteoProvider(latitude, longitude)
        WeatherCollector(provider, city, RAW_DIR).collect(args.years)


if __name__ == "__main__":
    main()
