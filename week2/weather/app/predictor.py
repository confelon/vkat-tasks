"""Runtime prediction: recent weather + a trained city model = forecast.

PredictorRegistry discovers <City>-<model>.pkl files in app/data/ and hands
out one WeatherPredictor per file. Each predictor fetches the last 45 days
of real weather for its city from Open-Meteo (cached until the day changes),
reuses DatasetBuilder's feature engineering, and decodes the model outputs.

Pressure is stored and predicted in hPa (the API's native unit) and converted
to mmHg here, since weather in Europe is displayed in millimetres of mercury.
"""

import pickle
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .cities import CITIES
from .openmeteo import OpenMeteoProvider
from .train import FEATURE_COLUMNS, HORIZON_DAYS, TARGETS, DatasetBuilder

DATA_DIR = Path(__file__).parent / "data"
HISTORY_DAYS = 45  # enough real days to compute the 30-day rolling features
HPA_TO_MMHG = 0.750062


class WeatherPredictor:
    def __init__(self, model_file: Path) -> None:
        with open(model_file, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.city = bundle["city"]
        latitude, longitude = CITIES[self.city]
        self.provider = OpenMeteoProvider(latitude, longitude)
        self.builder = DatasetBuilder(self.city)
        # cached_features and last_history_day are set by refresh_features
        self.cached_day: date | None = None

    def predict_month(self) -> list[dict]:
        self.refresh_features()
        outputs = self.model.predict(self.cached_features)[0]
        month = []
        for horizon in range(1, HORIZON_DAYS + 1):
            day = self.last_history_day + timedelta(days=horizon)
            # The 150 outputs are grouped day by day: 5 values for day+1, then 5 for day+2, ...
            first_output = (horizon - 1) * len(TARGETS)
            values = {target: round(float(outputs[first_output + i]), 1) for i, target in enumerate(TARGETS)}
            values["pressure"] = round(values["pressure"] * HPA_TO_MMHG, 1)
            # Linear models can predict impossible values (negative rain, >100% humidity) — clip them
            values["precipitation"] = max(0.0, values["precipitation"])
            values["wind_speed"] = max(0.0, values["wind_speed"])
            values["humidity"] = min(100.0, max(0.0, values["humidity"]))
            month.append({"date": day.isoformat()} | values)
        return month

    def refresh_features(self) -> None:
        if self.cached_day == date.today():
            return
        rows = self.provider.fetch_recent(HISTORY_DAYS)
        df = pd.DataFrame([asdict(row) for row in rows])
        df["date"] = pd.to_datetime(df["date"])
        last_row = self.builder.add_features(df).dropna().iloc[[-1]]
        self.cached_features = last_row[FEATURE_COLUMNS]
        self.last_history_day = last_row["date"].iloc[0].date()
        self.cached_day = date.today()


class PredictorRegistry:
    """Discovers <City>-<model>.pkl files on disk and caches predictors for them."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.predictors: dict[Path, WeatherPredictor] = {}

    def available(self) -> dict[str, list[str]]:
        cities: dict[str, list[str]] = {}
        for model_file in sorted(self.data_dir.glob("*.pkl")):
            city, model_name = model_file.stem.split("-", 1)
            cities.setdefault(city, []).append(model_name)
        return cities

    def get(self, city: str, model_name: str | None = None) -> WeatherPredictor:
        model_name = model_name or self.available()[city][0]
        model_file = self.data_dir / f"{city}-{model_name}.pkl"
        if model_file not in self.predictors:
            self.predictors[model_file] = WeatherPredictor(model_file)
        return self.predictors[model_file]

    def metrics(self, city: str) -> dict[str, dict]:
        """Validation metrics stored in each of the city's model bundles at training time."""
        result = {}
        for model_file in sorted(self.data_dir.glob(f"{city}-*.pkl")):
            with open(model_file, "rb") as f:
                bundle = pickle.load(f)
            result[bundle["model_name"]] = bundle["metrics"]
        return result
