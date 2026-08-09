import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .predictor import DATA_DIR, PredictorRegistry

app = FastAPI()
registry = PredictorRegistry(DATA_DIR)

STATIC_DIR = Path(__file__).parent / "static"
I18N_DIR = STATIC_DIR / "i18n"
README_FILE = Path(__file__).parent.parent / "README.md"
README_RU_FILE = Path(__file__).parent.parent / "README.ru.md"


@app.get("/readme")
def get_readme(lang: str = "en"):
    readme = README_RU_FILE if lang == "ru" else README_FILE
    return PlainTextResponse(readme.read_text(encoding="utf-8"))


@app.get("/api/languages")
def get_languages():
    return {
        f.stem: json.loads(f.read_text(encoding="utf-8"))["language"]
        for f in sorted(I18N_DIR.glob("*.json"))
    }


@app.get("/api/cities")
def get_cities():
    return registry.available()


@app.get("/api/benchmark")
def get_benchmark():
    return json.loads((DATA_DIR / "benchmark.json").read_text())


@app.get("/api/depth-benchmark")
def get_depth_benchmark():
    return json.loads((DATA_DIR / "depth_benchmark.json").read_text())


@app.get("/api/weight-benchmark")
def get_weight_benchmark():
    return json.loads((DATA_DIR / "weight_benchmark.json").read_text())


@app.get("/api/models")
def get_models(city: str = "Moscow"):
    return registry.metrics(city)


@app.get("/api/forecast")
def get_forecast(city: str = "Moscow", model: str | None = None):
    return registry.get(city, model).predict_month()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
