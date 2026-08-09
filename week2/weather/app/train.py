"""Trains the weather prediction model from the collected historical data.

Each training sample is one day: the features summarize the previous 30 days
of weather (rolling averages, trends, seasonality), and the targets are the
5 weather values for each of the next 30 days (150 outputs at once).

A ladder of models is compared on a held-out validation set (the most recent
year): Dummy (overall mean) → climatology (seasonal mean) → Ridge → Random
Forest → gradient boosting. Each model gets a skill score against climatology
(1 - MAE_model / MAE_climatology; positive = beats the seasonal average).
Older samples get linearly smaller training weight, so models trust recent
years more.

The best model is saved to app/data/<City>-<model>.pkl (e.g. Almaty-ridge.pkl)
together with its validation metrics, which the web app serves via /api/models.

Run inside the container:
    docker compose run --rm weather python -m app.train [--city Moscow] [--max-years 10]

Learning-curve experiment (stdout only, nothing is saved or exposed via API):
    docker compose run --rm weather python -m app.train --city Almaty --compare-years 10 20 30
Each depth trains on the last N years only, but all depths validate on the same
most recent year — otherwise the numbers would not be comparable.
"""

import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Uncomment together with the gradient_boosting candidate below:
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.multioutput import MultiOutputRegressor

RAW_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR = Path(__file__).parent / "data"

WEATHER_COLUMNS = ["temp_min", "temp_max", "temperature", "humidity", "pressure", "wind_speed", "precipitation"]
TARGETS = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
HORIZON_DAYS = 30
VALIDATION_DAYS = 365
OLDEST_SAMPLE_WEIGHT = 0.2

FEATURE_COLUMNS = (
    WEATHER_COLUMNS
    + [f"{col}_avg{window}" for col in WEATHER_COLUMNS for window in (7, 14, 30)]
    + ["temp_trend", "precip_total7", "precip_total30", "season_sin", "season_cos", "month"]
)
TARGET_COLUMNS = [f"{target}_plus{h}" for h in range(1, HORIZON_DAYS + 1) for target in TARGETS]


class DatasetBuilder:
    """Turns one city's raw daily CSVs into a DataFrame with feature and target columns."""

    def __init__(self, city: str, max_years: int | None = None) -> None:
        self.city = city
        self.max_years = max_years

    def build(self) -> pd.DataFrame:
        df = self.load_raw()
        df = self.add_features(df)
        df = self.add_targets(df)
        # Drop edge rows: the first 30 days have no full history, the last 30 no full future
        return df.dropna().reset_index(drop=True)

    def load_raw(self) -> pd.DataFrame:
        frames = [pd.read_csv(f, parse_dates=["date"]) for f in sorted(RAW_DIR.glob(f"{self.city}-*.csv"))]
        df = pd.concat(frames).sort_values("date")
        if self.max_years:
            cutoff = df["date"].max() - pd.Timedelta(days=round(self.max_years * 365.25))
            df = df[df["date"] > cutoff]
        return df.reset_index(drop=True)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in WEATHER_COLUMNS:
            for window in (7, 14, 30):
                df[f"{col}_avg{window}"] = df[col].rolling(window).mean()
        df["temp_trend"] = df["temperature_avg7"] - df["temperature_avg30"]
        df["precip_total7"] = df["precipitation"].rolling(7).sum()
        df["precip_total30"] = df["precipitation"].rolling(30).sum()
        # Seasonality as sin/cos so December 31 and January 1 are neighbours, not opposites
        day_of_year = df["date"].dt.dayofyear
        df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        df["month"] = df["date"].dt.month
        return df

    def add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        shifted = {f"{target}_plus{h}": df[target].shift(-h) for h in range(1, HORIZON_DAYS + 1) for target in TARGETS}
        return pd.concat([df, pd.DataFrame(shifted)], axis=1)


class ClimatologyBaseline:
    """Naive baseline: predict the multi-year average for the target date's day of year."""

    def __init__(self, train_df: pd.DataFrame) -> None:
        day_of_year = train_df["date"].dt.dayofyear
        self.averages = {}
        for target in TARGETS:
            by_day = train_df.groupby(day_of_year)[target].mean().reindex(range(1, 367)).ffill()
            # Smooth over ±7 days so single-day spikes don't dominate
            self.averages[target] = by_day.rolling(15, center=True, min_periods=1).mean()

    def predict_frame(self, dates: pd.Series) -> pd.DataFrame:
        columns = {}
        for h in range(1, HORIZON_DAYS + 1):
            target_days = (dates + pd.Timedelta(days=h)).dt.dayofyear.values
            for target in TARGETS:
                columns[f"{target}_plus{h}"] = self.averages[target].loc[target_days].values
        return pd.DataFrame(columns, index=dates.index)


class ModelTrainer:
    """Compares models against a climatology baseline and saves the best model."""

    def __init__(
        self, df: pd.DataFrame, city: str, models: list[str] | None = None,
        oldest_weight: float = OLDEST_SAMPLE_WEIGHT,
    ) -> None:
        self.city = city
        self.models = models  # None = the whole ladder
        self.oldest_weight = oldest_weight
        cutoff = df["date"].max() - timedelta(days=VALIDATION_DAYS)
        self.train_df = df[df["date"] <= cutoff]
        validation = df[df["date"] > cutoff]
        self.X_train, self.Y_train = self.train_df[FEATURE_COLUMNS], self.train_df[TARGET_COLUMNS]
        self.X_val, self.Y_val = validation[FEATURE_COLUMNS], validation[TARGET_COLUMNS]
        self.val_dates = validation["date"]
        print(f"Train: {len(self.train_df)} samples up to {cutoff.date()}, validation: {len(validation)} samples after")

    def run(self) -> None:
        results = self.compare()
        metrics = {
            "training": {
                "first_day": str(self.train_df["date"].min().date()),
                "last_day": str(self.train_df["date"].max().date()),
                "train_samples": len(self.X_train),
                "validation_samples": len(self.X_val),
                "horizon_days": HORIZON_DAYS,
                "validation_days": VALIDATION_DAYS,
                "features": len(FEATURE_COLUMNS),
                "outputs": len(TARGET_COLUMNS),
                "oldest_sample_weight": self.oldest_weight,
            },
            "climatology": results["climatology"],
            "models": results["report"],
            "winner": results["winner"],
            "temperature_mae_by_horizon": results["horizon_curve"],
        }
        self.save(results["model"], results["winner"], metrics)

    def compare(self) -> dict:
        """Fits and evaluates the whole model ladder; prints reports, saves nothing."""
        baseline = ClimatologyBaseline(self.train_df)
        climatology = self.evaluate("climatology (seasonal average)", baseline.predict_frame(self.val_dates))

        candidates = {
            "dummy": DummyRegressor(),  # predicts each target's overall mean — the bottom rung
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(
                n_estimators=50, min_samples_leaf=20, n_jobs=-1, random_state=42
            ),
            # HistGradientBoosting only handles one output, so the wrapper fits one boosted
            # model per each of the 150 outputs — minutes of CPU per city for a ~2% gain
            # over Ridge. Disabled for fleet-scale training; uncomment to re-benchmark.
            # "gradient_boosting": MultiOutputRegressor(
            #     HistGradientBoostingRegressor(max_iter=50, random_state=42), n_jobs=-1
            # ),
        }
        if self.models:
            candidates = {name: candidates[name] for name in self.models}
        # Linear weights: oldest sample counts oldest_weight, newest 1.0 (rows are date-sorted)
        weights = np.linspace(self.oldest_weight, 1.0, len(self.X_train))

        report = {}
        all_predictions = {}
        for name, model in candidates.items():
            model.fit(self.X_train, self.Y_train, sample_weight=weights)
            predictions = pd.DataFrame(model.predict(self.X_val), columns=TARGET_COLUMNS, index=self.Y_val.index)
            report[name] = self.evaluate(name, predictions, climatology["mae"])
            all_predictions[name] = predictions

        winner = min(report, key=lambda name: report[name]["overall"])
        horizon_curve = self.temperature_mae_by_horizon(all_predictions[winner])
        print(f"\nWinner: {winner}")
        print("Temperature MAE by horizon (how quality degrades with distance):")
        for h in (1, 5, 10, 15, 20, 25, 30):
            print(f"  +{h:>2} days: {horizon_curve[h - 1]:.2f}")
        return {
            "report": report,
            "winner": winner,
            "model": candidates[winner],
            "climatology": climatology,
            "horizon_curve": horizon_curve,
        }

    def evaluate(self, name: str, predictions: pd.DataFrame, climatology_mae: dict | None = None) -> dict:
        print(f"\n{name} — validation MAE (averaged over all 30 horizons):")
        maes = {}
        biases = {}
        skills = {}
        for target in TARGETS:
            columns = [f"{target}_plus{h}" for h in range(1, HORIZON_DAYS + 1)]
            maes[target] = mean_absolute_error(self.Y_val[columns], predictions[columns])
            # Bias: mean signed error (prediction − actual). Near zero is healthy;
            # a big value means a systematic offset that MAE alone can't reveal
            biases[target] = float((predictions[columns].values - self.Y_val[columns].values).mean())
            line = f"  {target}: {maes[target]:.2f} (bias {biases[target]:+.2f})"
            if climatology_mae:
                # Skill: how much better than the seasonal average (0 = same, negative = worse)
                skills[target] = 1 - maes[target] / climatology_mae[target]
                line += f"  (skill {skills[target]:+.2f})"
            print(line)
        # One overall score across targets of different units: MAE relative to each target's spread
        relative_errors = (self.Y_val - predictions).abs().mean() / self.Y_val.std()
        overall = float(relative_errors.mean())
        print(f"  overall (MAE / std): {overall:.3f}")
        return {"mae": maes, "bias": biases, "skill": skills, "overall": overall}

    def temperature_mae_by_horizon(self, predictions: pd.DataFrame) -> list[float]:
        return [
            mean_absolute_error(self.Y_val[f"temperature_plus{h}"], predictions[f"temperature_plus{h}"])
            for h in range(1, HORIZON_DAYS + 1)
        ]

    def save(self, model, model_name: str, metrics: dict) -> None:
        # A retrain can crown a different winner; the previous bundle would linger
        # on disk and still be served by the API, so replace all of the city's bundles
        for old_file in DATA_DIR.glob(f"{self.city}-*.pkl"):
            print(f"Removing {old_file.name}")
            old_file.unlink()

        bundle = {
            "model": model,
            "city": self.city,
            "model_name": model_name,
            "metrics": metrics,
        }
        model_file = DATA_DIR / f"{self.city}-{model_name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(bundle, f)
        print(f"Saved to {model_file}")


class YearsExperiment:
    """Learning-curve experiment: how does training-history depth affect quality?

    Trains the ladder on the last N years for each depth, validating every run
    on the same most recent year, and prints a comparison table. Stdout only —
    nothing is saved to disk or exposed through the API.
    """

    def __init__(self, city: str, depths: list[int], models: list[str] | None = None) -> None:
        self.city = city
        self.depths = depths
        self.models = models

    def run(self) -> list[dict]:
        rows = []
        for years in self.depths:
            print(f"\n===== {self.city}: training on the last {years} years =====")
            df = DatasetBuilder(self.city, max_years=years).build()
            trainer = ModelTrainer(df, self.city, self.models)
            results = trainer.compare()
            best = results["report"][results["winner"]]
            rows.append(
                {
                    "years": years,
                    "samples": len(trainer.X_train),
                    "winner": results["winner"],
                    "temp_mae": round(best["mae"]["temperature"], 2),
                    "temp_bias": round(best["bias"]["temperature"], 2),
                    "overall": round(best["overall"], 3),
                }
            )
        print(f"\n===== Summary for {self.city}: history depth vs quality =====")
        print(f"{'years':>5} {'samples':>8} {'winner':>15} {'temp MAE':>9} {'bias':>6} {'overall':>8}")
        for r in rows:
            print(
                f"{r['years']:>5} {r['samples']:>8} {r['winner']:>15} "
                f"{r['temp_mae']:>9.2f} {r['temp_bias']:>+6.2f} {r['overall']:>8.3f}"
            )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a weather prediction model for a city")
    parser.add_argument("--city", default="Moscow", help="city whose raw CSVs to train on")
    parser.add_argument("--max-years", type=int, default=None, help="train on the last N years of history only")
    parser.add_argument(
        "--compare-years", type=int, nargs="+", metavar="N",
        help="experiment: compare training-history depths (e.g. 10 20 30), print a summary, save nothing",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME",
        help="restrict the candidate ladder, e.g. --models ridge",
    )
    args = parser.parse_args()

    if args.compare_years:
        YearsExperiment(args.city, args.compare_years, args.models).run()
        return
    df = DatasetBuilder(args.city, args.max_years).build()
    ModelTrainer(df, args.city, args.models).run()


if __name__ == "__main__":
    main()
