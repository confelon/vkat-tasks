"""Fleet benchmark: one comparison table for all trained city models.

Aggregates the validation metrics stored in every app/data/<City>-<model>.pkl
bundle at training time — no retraining happens here. Prints the table, flags
anomalies, and persists two artifacts: app/data/benchmark.csv (for spreadsheet
inspection) and app/data/benchmark.json (table + anomalies + generated_at,
served read-only by GET /api/benchmark for the fleet page). The benchmark
itself always runs offline; the UI only ever reads the persisted report.

Anomaly checks (thresholds calibrated so "anomaly" means unusual for this
fleet — near-zero skill and small biases are normal at a 30-day horizon):
- dummy winning a city (data pipeline failure until proven otherwise)
- temperature skill clearly negative, below -0.05 (model loses to climatology;
  a near-zero tie is normal in mild climates)
- |temperature bias| over 1 °C (a degree of systematic offset is a real problem)
- train samples far below the fleet median (gaps in collected history)
- overall score outside median ± 1.5×IQR (statistical outlier)
- validation_days / horizon_days differing from the current constants
  (all cities must be validated identically to stay comparable)

Run inside the container:
    docker compose run --rm weather python -m app.benchmark
"""

import json
import pickle
from datetime import date

import pandas as pd

from .train import DATA_DIR, HORIZON_DAYS, VALIDATION_DAYS

SUMMARY_FILE = DATA_DIR / "benchmark.csv"
JSON_FILE = DATA_DIR / "benchmark.json"
BIAS_LIMIT = 1.0  # °C of systematic temperature offset worth investigating
SKILL_FLOOR = -0.05  # below this the model clearly loses to climatology (near zero = tie, normal)
LOW_SAMPLES_FACTOR = 0.7  # of the fleet median


class FleetBenchmark:
    """Collects per-city metrics from the model bundles and flags anomalies."""

    def collect(self) -> pd.DataFrame:
        rows = []
        for model_file in sorted(DATA_DIR.glob("*.pkl")):
            with open(model_file, "rb") as f:
                bundle = pickle.load(f)
            metrics = bundle["metrics"]
            training = metrics["training"]
            winner = metrics["models"][metrics["winner"]]
            rows.append(
                {
                    "city": bundle["city"],
                    "model": bundle["model_name"],
                    "trained_through": training["last_day"],
                    "train_samples": training["train_samples"],
                    "validation_days": training["validation_days"],
                    "horizon_days": training["horizon_days"],
                    "temp_mae": round(winner["mae"]["temperature"], 2),
                    "temp_bias": round(winner["bias"]["temperature"], 2),
                    "temp_skill": round(winner["skill"]["temperature"], 3),
                    "overall": round(winner["overall"], 3),
                }
            )
        return pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)

    def anomalies(self, fleet: pd.DataFrame) -> list[str]:
        flags = []
        median_samples = fleet["train_samples"].median()
        q1, q3 = fleet["overall"].quantile([0.25, 0.75])
        fence = 1.5 * (q3 - q1)
        for row in fleet.itertuples():
            if row.model == "dummy":
                flags.append(f"{row.city}: dummy won — data pipeline failure until proven otherwise")
            if row.temp_skill < SKILL_FLOOR:
                flags.append(f"{row.city}: temperature skill {row.temp_skill:+.3f} — model clearly loses to climatology")
            if abs(row.temp_bias) > BIAS_LIMIT:
                flags.append(f"{row.city}: temperature bias {row.temp_bias:+.2f} °C — systematic offset")
            if row.train_samples < LOW_SAMPLES_FACTOR * median_samples:
                flags.append(
                    f"{row.city}: only {row.train_samples} train samples vs fleet median {median_samples:.0f} — gaps in history?"
                )
            if not (q1 - fence) <= row.overall <= (q3 + fence):
                flags.append(f"{row.city}: overall {row.overall:.3f} is an IQR outlier for the fleet")
            if row.validation_days != VALIDATION_DAYS or row.horizon_days != HORIZON_DAYS:
                flags.append(
                    f"{row.city}: validated with {row.validation_days}d/{row.horizon_days}d "
                    f"instead of the fleet's {VALIDATION_DAYS}d/{HORIZON_DAYS}d — retrain to stay comparable"
                )
        return flags

    def run(self) -> None:
        fleet = self.collect()
        print(fleet.to_string(index=False))
        flags = self.anomalies(fleet)

        fleet.to_csv(SUMMARY_FILE, index=False)
        report = {
            "generated_at": str(date.today()),
            "models": len(fleet),
            "fleet": fleet.to_dict(orient="records"),
            "anomalies": flags,
        }
        JSON_FILE.write_text(json.dumps(report, indent=2))
        print(f"\nSummary persisted to {SUMMARY_FILE} and {JSON_FILE}")

        print(f"\nAnomalies ({len(flags)}):" if flags else "\nNo anomalies detected.")
        for flag in flags:
            print(f"  {flag}")


def main() -> None:
    FleetBenchmark().run()


if __name__ == "__main__":
    main()
