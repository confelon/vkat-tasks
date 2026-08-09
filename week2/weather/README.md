# Weather Forecast (student project)

Student ML project: predict weather 1 month ahead for 52 European and ex-USSR capitals.
Focus is on integration (data fetching, pipeline, serving), not on advanced ML.

Weather data by [Open-Meteo](https://open-meteo.com/) (CC-BY, free, no API key).

## Findings

- A 30-day forecast is mostly a refined seasonal norm: fleet mean temperature MAE is
  **2.58 °C**, but the advantage over plain climatology (multi-year seasonal average) is thin —
  in ~40% of cities, mostly mild climates (Athens, Rome, Lisbon…), climatology alone ties or wins.
- **More history makes models worse, not better.** Each extra decade of training data adds
  ~0.1 °C of systematic cold bias (old, cooler climate leaking in) — see the depth benchmark below.
- Predictability is climate, not modeling: Valletta (Mediterranean island) scores 1.40 °C MAE,
  Astana (extreme continental steppe) 4.17 °C — a 3× spread no model choice can close.

## Pipeline

Historical data + training (offline):

```
Open-Meteo → collector.py → data/raw/<City>-YYYY.csv → train.py → <City>-<model>.pkl
```

Prediction (runtime, triggered by the browser):

```
Browser → FastAPI → Open-Meteo (recent weather) → ML model → prediction → Browser
```

Weather data is cached locally (`app/data/raw/`) so training stays reproducible, the data is
inspectable, and retraining needs no re-fetching.

## Stack

- Backend: FastAPI (URL handlers only), static files served by Python itself (no nginx).
- Frontend: plain HTML + jQuery + plotly.js, no build step. All JS vendored in
  `app/static/vendor/` — no CDN at runtime.
- Everything runs via docker-compose: `docker-compose.yml` (dev, `app/` bind-mounted, uvicorn
  `--reload`) and `docker-compose.prod.yml` (prod, code baked into the image).

```bash
# dev
docker compose up --build -d

# prod
docker compose -f docker-compose.prod.yml up --build -d
```

App: http://localhost:8000 · model verification: `/model.html` · fleet benchmark: `/fleet.html` ·
this document rendered: `/readme.html`

## Deploy to a real server

The prod image is self-contained: code, all locally trained models (`app/data/*.pkl`), and the
benchmark reports are baked in at build time. Raw training CSVs stay out (`.dockerignore`) —
training happens on the dev box, the server only serves. Build locally, ship one file, run on
any Linux box with Docker:

```bash
# on the dev box: build the prod image and bundle it into a single file
docker compose -f docker-compose.prod.yml build
docker save -o weather.tar weather-weather

# copy to the server and start it (plain HTTP on port 80 — see security posture)
scp weather.tar user@server:/tmp/
ssh user@server "docker load -i /tmp/weather.tar && \
  docker run -d --name weather --restart unless-stopped -p 80:8000 weather-weather"
```

To update: rebuild and re-copy, then on the server
`docker rm -f weather`, `docker load`, `docker run` again. (A single `docker run` replaces
docker-compose on the server — one container, nothing to compose.)

## 1. Collect data

One CSV per city per year lands in `app/data/raw/`. Complete years are skipped on re-run, so the
command is resumable — if it stops (rate limit, network), just run it again. ~52 cities × 40
archive calls with a pause between calls: expect a few hours on the first run.

```bash
docker compose run --rm weather python -m app.collector --city all --years 40
docker compose run --rm weather python -m app.collector --city Riga            # one city
docker compose run --rm weather python -m app.collector --city Karaganda --lat 49.8 --lon 73.1
```

The Open-Meteo archive goes back to 1940 and lags a few days behind real time; re-run later to
fill in the freshest days. An optional SOCKS5 proxy for API calls goes into `.env`:

```
# from inside the container, use host.docker.internal to reach a proxy on the host
# ALL_PROXY=socks5h://host.docker.internal:1080
```

## 2. Train a model for every city

One sample = one day: ~30 features summarizing the previous 30 days (rolling averages, trend,
seasonality) predict all 5 weather values (temperature, humidity, pressure, wind, precipitation)
for each of the next 30 days. A ladder of models — Dummy (overall mean), climatology (seasonal
average, the skill reference), Ridge, Random Forest — is compared on the most recent year, never
seen in training; older samples get linearly smaller weight (0.2 for the oldest — validated by
the sample-weight benchmark below: fast-warming continental cities like Almaty or Tashkent need
it, some European cities are marginally better without; 0.2 is the fleet-wide compromise). The winner is saved with its
validation metrics to `app/data/<City>-<model>.pkl`, replacing the city's previous bundle so
stale winners never linger.

Windows (PowerShell):

```powershell
Get-ChildItem app\data\raw\*.csv |
  ForEach-Object { $_.BaseName -replace '-\d+$', '' } |
  Sort-Object -Unique |
  ForEach-Object { docker compose run --rm weather python -m app.train --city $_ --max-years 10 }
```

Linux / macOS (bash; also works in Git Bash on Windows):

```bash
for city in $(ls app/data/raw | sed 's/-[0-9]*\.csv//' | sort -u); do
  docker compose run --rm weather python -m app.train --city "$city" --max-years 10
done
```

`--max-years 10` follows the benchmark recommendation (below). Expect 30–60 minutes for the fleet.

## 3. Benchmarks

Single-city history-depth experiment (stdout only, nothing saved):

```bash
docker compose run --rm weather python -m app.train --city Moscow --compare-years 10 20 30 --models ridge
```

Fleet-wide depth benchmark — one model per city per depth, persists
`app/data/depth_benchmark.json`, rendered on `/fleet.html`:

```bash
docker compose run --rm weather python -m app.experiments depth
```

Fleet-wide sample-weight benchmark — does recency weighting (oldest sample 0.2 → newest 1.0)
still help once the window is 10 years? Trains ridge per city with weights 0.2 / 0.5 / 1.0
(1.0 = no weighting), persists `app/data/weight_benchmark.json`, rendered on `/fleet.html`:

```bash
docker compose run --rm weather python -m app.experiments weight
```

Fleet summary + anomaly detection over the trained models — no retraining, reads the metrics
stored in each bundle, persists `app/data/benchmark.csv` + `benchmark.json` (also on
`/fleet.html`); flags dummy winning, clearly negative skill (< −0.05 — a near-zero tie with
climatology is normal at this horizon), |bias| > 1 °C, sample gaps, IQR outliers, and
validation-config drift:

```bash
docker compose run --rm weather python -m app.benchmark
```

## Depth benchmark results (2026-08-09, ridge, 52 cities)

Every depth validates on the same most recent year. Bias = mean signed error (prediction − actual).

| Depth | Mean MAE, °C | Median MAE, °C | Mean bias, °C | Mean overall | Best in cities |
|-------|-------------|----------------|---------------|--------------|----------------|
| 10 years | 2.576 | 2.565 | −0.08 | 0.614 | 25 |
| 20 years | 2.566 | 2.545 | −0.35 | 0.607 | 19 |
| 30 years | 2.581 | 2.550 | −0.48 | 0.608 | 8 |
| 40 years | 2.593 | 2.570 | −0.55 | 0.610 | 0 |

**Recommendation: train the fleet with `--max-years 10`.** Accuracy stops improving past 20
years (10y/20y are 0.01 °C apart — noise), while cold bias grows monotonically with depth —
old decades teach the model yesterday's climate. 10y is the best depth in 25 of 52 cities, 40y
in none, and a quarter of the data means the fastest training run. Known exception: Baltic/Nordic
and Caucasus climates improve with depth (Tallinn 3.00 → 2.85), but 20y captures most of that,
and per-city depth tuning would buy ~0.05 °C at the price of per-city config.

## Prediction API

`GET /api/cities` — trained models on disk as `{city: [models]}` · `GET /api/forecast?city=&model=`
— the 30-day forecast (5 values per day) · `GET /api/models?city=` — validation metrics from the
city's bundle · `GET /api/benchmark`, `GET /api/depth-benchmark`, `GET /api/weight-benchmark` — the persisted offline reports ·
`GET /api/languages` — available translations.

Features are built from the last 45 days of real weather fetched from Open-Meteo, cached until
the day changes — at most one upstream call per city per day. Display units are European:
pressure in mmHg (stored and modeled in hPa), °C, m/s, mm. Predictions are clipped to physically
possible ranges (no negative rain, humidity 0–100%).

The web UI shows the monthly forecast as a table plus plotly charts per parameter, with city and
language dropdowns (43 languages, auto-picked from the browser's preferences, remembered in
localStorage). `/model.html` shows the model ladder with MAE/bias/skill and the per-horizon error
curve; `/fleet.html` renders both offline benchmark reports.

## Security posture

No writes, no sensitive data — the prod container is exposed to the internet over plain HTTP
with no auth, same as a static `hello_world.html`. HTTPS, auth, and rate limiting are deliberately skipped.
