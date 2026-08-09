"""Fleet experiments: train ridge per city across a set of variants and compare.

Both experiments share one skeleton (FleetExperiment): for every city with raw
CSVs, train ridge once per variant — every run validating on the same most
recent year — then aggregate fleet-wide, flag anomalies, and persist a JSON
report that the fleet page renders (the UI only ever reads the persisted file).

- depth:  variants are history depths in years (10/20/30/40) — does more
  training history help? (Finding: no — it adds cold bias.)
- weight: variants are oldest-sample weights (0.2/0.5/1.0, where 1.0 means no
  recency weighting) on the fixed 10-year window — does recency weighting help?
  (Finding: only in fast-warming continental cities.)

Heavy — one model per city per variant. Run offline:
    docker compose run --rm weather python -m app.experiments depth
    docker compose run --rm weather python -m app.experiments weight
"""

import argparse
import contextlib
import io
import json
import statistics
from datetime import date

from .train import DATA_DIR, RAW_DIR, DatasetBuilder, ModelTrainer

MODELS = ["ridge"]
FLEET_WINDOW_YEARS = 10  # the standard training window used for the fleet
NOISE_FLOOR = 0.1  # °C — MAE differences below this on a single validation year are noise


class FleetExperiment:
    """Skeleton: run variants per city, aggregate, flag anomalies, persist JSON."""

    name: str
    description: str
    variants: list
    json_file: str

    def label(self, variant) -> str:
        return str(variant)

    def city_rows(self, city: str) -> list[dict]:
        raise NotImplementedError

    def anomalies(self, results: dict[str, list[dict]]) -> list[str]:
        raise NotImplementedError

    def ridge_scores(self, df, city: str, **trainer_args) -> dict:
        # ModelTrainer prints a full report — too noisy for a fleet run
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = ModelTrainer(df, city, MODELS, **trainer_args)
            best = trainer.compare()["report"]["ridge"]
        return {
            "samples": len(trainer.X_train),
            "temp_mae": round(best["mae"]["temperature"], 2),
            "temp_bias": round(best["bias"]["temperature"], 2),
            "overall": round(best["overall"], 3),
        }

    def run(self) -> None:
        cities = sorted({f.name.rsplit("-", 1)[0] for f in RAW_DIR.glob("*.csv")})
        print(f"{self.name} experiment: {len(cities)} cities, variants {self.variants}, models {MODELS}")
        results = {}
        for city in cities:
            rows = self.city_rows(city)
            results[city] = rows
            print(
                f"{city}: "
                + "  ".join(f"{self.label(r['variant'])} {r['temp_mae']:.2f} (bias {r['temp_bias']:+.2f})" for r in rows),
                flush=True,
            )

        aggregate = self.aggregate(results)
        anomalies = self.anomalies(results)
        self.persist(results, aggregate, anomalies)
        self.print_summary(aggregate, anomalies)

    def aggregate(self, results: dict[str, list[dict]]) -> list[dict]:
        rows = []
        for i, variant in enumerate(self.variants):
            at_variant = [city_rows[i] for city_rows in results.values()]
            rows.append(
                {
                    "variant": variant,
                    "mean_temp_mae": round(statistics.mean(r["temp_mae"] for r in at_variant), 3),
                    "median_temp_mae": round(statistics.median(r["temp_mae"] for r in at_variant), 3),
                    "mean_temp_bias": round(statistics.mean(r["temp_bias"] for r in at_variant), 3),
                    "mean_overall": round(statistics.mean(r["overall"] for r in at_variant), 3),
                    "best_in_cities": sum(
                        1 for city_rows in results.values()
                        if min(city_rows, key=lambda r: r["temp_mae"])["variant"] == variant
                    ),
                }
            )
        return rows

    def persist(self, results: dict, aggregate: list[dict], anomalies: list[str]) -> None:
        report = {
            "generated_at": str(date.today()),
            "description": self.description,
            "variants": self.variants,
            "models": MODELS,
            "cities": results,
            "aggregate": aggregate,
            "anomalies": anomalies,
        }
        out_file = DATA_DIR / self.json_file
        out_file.write_text(json.dumps(report, indent=2))
        print(f"\nPersisted to {out_file}")

    def print_summary(self, aggregate: list[dict], anomalies: list[str]) -> None:
        print("\naggregate:")
        for row in aggregate:
            print(
                f"  {self.label(row['variant'])}: mean temp MAE {row['mean_temp_mae']:.3f}, "
                f"mean bias {row['mean_temp_bias']:+.2f}, mean overall {row['mean_overall']:.3f}, "
                f"best in {row['best_in_cities']} cities"
            )
        print(f"\nAnomalies ({len(anomalies)}):" if anomalies else "\nNo anomalies detected.")
        for flag in anomalies:
            print(f"  {flag}")


class DepthExperiment(FleetExperiment):
    """How does training-history depth affect quality?"""

    name = "depth"
    description = "ridge per history depth; every depth validates on the same most recent year"
    variants = [10, 20, 30, 40]
    json_file = "depth_benchmark.json"

    def label(self, variant) -> str:
        return f"{variant}y"

    def city_rows(self, city: str) -> list[dict]:
        rows = []
        for years in self.variants:
            df = DatasetBuilder(city, max_years=years).build()
            rows.append({"variant": years} | self.ridge_scores(df, city))
        return rows

    def anomalies(self, results: dict[str, list[dict]]) -> list[str]:
        flags = []
        shallow_maes = [rows[0]["temp_mae"] for rows in results.values()]
        mean, sd = statistics.mean(shallow_maes), statistics.stdev(shallow_maes)
        for city, rows in results.items():
            shallow, deep = rows[0], rows[-1]
            if abs(shallow["temp_mae"] - mean) > 2 * sd:
                flags.append(
                    f"{city}: temp MAE outlier at {self.label(shallow['variant'])}: "
                    f"{shallow['temp_mae']:.2f} (fleet {mean:.2f} ± {sd:.2f})"
                )
            if deep["temp_mae"] < shallow["temp_mae"] - NOISE_FLOOR:
                flags.append(f"{city}: deeper history helps: {shallow['temp_mae']:.2f} -> {deep['temp_mae']:.2f}")
            for previous, current in zip(rows, rows[1:]):
                if current["samples"] == previous["samples"]:
                    flags.append(
                        f"{city}: {self.label(current['variant'])} trained on the same {current['samples']} "
                        f"samples as {self.label(previous['variant'])} — not enough history on disk"
                    )
        return flags


class WeightExperiment(FleetExperiment):
    """Does recency weighting still help once the window is 10 years?"""

    name = "weight"
    description = (
        f"ridge on the last {FLEET_WINDOW_YEARS} years, oldest-sample weight per variant "
        "(1.0 = no weighting); same validation year for every run"
    )
    variants = [0.2, 0.5, 1.0]
    json_file = "weight_benchmark.json"

    def label(self, variant) -> str:
        return f"w{variant}"

    def city_rows(self, city: str) -> list[dict]:
        df = DatasetBuilder(city, max_years=FLEET_WINDOW_YEARS).build()
        return [{"variant": w} | self.ridge_scores(df, city, oldest_weight=w) for w in self.variants]

    def anomalies(self, results: dict[str, list[dict]]) -> list[str]:
        flags = []
        for city, rows in results.items():
            weighted, unweighted = rows[0], rows[-1]
            if weighted["temp_mae"] < unweighted["temp_mae"] - NOISE_FLOOR:
                flags.append(
                    f"{city}: weighting matters: MAE {unweighted['temp_mae']:.2f} without -> "
                    f"{weighted['temp_mae']:.2f} with it"
                )
            if unweighted["temp_mae"] < weighted["temp_mae"] - NOISE_FLOOR:
                flags.append(
                    f"{city}: weighting hurts: MAE {weighted['temp_mae']:.2f} with -> "
                    f"{unweighted['temp_mae']:.2f} without it"
                )
        return flags


EXPERIMENTS = {"depth": DepthExperiment, "weight": WeightExperiment}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fleet-wide training experiment")
    parser.add_argument("experiment", choices=EXPERIMENTS, help="which experiment to run")
    args = parser.parse_args()
    EXPERIMENTS[args.experiment]().run()


if __name__ == "__main__":
    main()
