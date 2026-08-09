"""Open-Meteo weather client: daily aggregates for one location.

Free, no API key. The archive endpoint serves full past years; the forecast
endpoint serves the most recent days (the archive lags a few days behind).
All HTTP honors the ALL_PROXY env var (socks5h:// supported).
"""

import os
import time
from dataclasses import dataclass, fields
from datetime import date

import httpx

RETRY_ATTEMPTS = 5

DAILY_FIELDS = [
    "temperature_2m_min",
    "temperature_2m_max",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
]


@dataclass
class DailyWeather:
    date: date
    temp_min: float
    temp_max: float
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    precipitation: float


COLUMNS = [field.name for field in fields(DailyWeather)]


class OpenMeteoProvider:
    """Fetches whole date ranges in single calls to the Open-Meteo APIs."""

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude
        # Set ALL_PROXY (e.g. socks5h://host:port) to route API calls through a proxy
        self.http = httpx.Client(timeout=30, proxy=os.environ.get("ALL_PROXY") or None)

    def fetch_range(self, start: date, end: date) -> list[DailyWeather]:
        # Year-long archive queries count as multiple "weighted" calls against
        # the free tier's per-minute cap, so pause between them
        time.sleep(2)
        params = self.common_params() | {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        response = self.get_archive_with_retries(params)
        response.raise_for_status()
        return self.parse_daily(response.json()["daily"])

    def get_archive_with_retries(self, params: dict) -> httpx.Response:
        """Retries transient failures: dropped connections and 429 rate limiting."""
        for _ in range(RETRY_ATTEMPTS - 1):
            try:
                response = self.http.get(self.ARCHIVE_URL, params=params)
            except httpx.TransportError as error:
                print(f"Connection failed ({error}), retrying in 10 seconds...")
                time.sleep(10)
                continue
            if response.status_code != 429:
                return response
            print("Open-Meteo rate limit hit (429), waiting 60 seconds...")
            time.sleep(60)
        # Last attempt runs bare so a persistent failure surfaces naturally
        return self.http.get(self.ARCHIVE_URL, params=params)

    def fetch_recent(self, days: int) -> list[DailyWeather]:
        """Last `days` days up to yesterday — the archive lags behind, the forecast API doesn't."""
        params = self.common_params() | {"past_days": days, "forecast_days": 1}
        response = self.http.get(self.FORECAST_URL, params=params)
        response.raise_for_status()
        rows = self.parse_daily(response.json()["daily"])
        return [row for row in rows if row.date < date.today()]

    def common_params(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": ",".join(DAILY_FIELDS),
            "wind_speed_unit": "ms",
            "timezone": "auto",
        }

    def parse_daily(self, daily: dict) -> list[DailyWeather]:
        days = []
        for i, day in enumerate(daily["time"]):
            # The archive lags a few days behind real time; those days come back as null
            if daily["temperature_2m_mean"][i] is None:
                continue
            days.append(
                DailyWeather(
                    date=date.fromisoformat(day),
                    temp_min=daily["temperature_2m_min"][i],
                    temp_max=daily["temperature_2m_max"][i],
                    temperature=daily["temperature_2m_mean"][i],
                    humidity=daily["relative_humidity_2m_mean"][i],
                    pressure=daily["surface_pressure_mean"][i],
                    wind_speed=daily["wind_speed_10m_max"][i],
                    precipitation=daily["precipitation_sum"][i],
                )
            )
        return days
