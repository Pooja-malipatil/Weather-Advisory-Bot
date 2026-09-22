"""
Open-Meteo client.

Design decision: this module returns a plain, flat dict of *actual numbers
returned by the API*. Nothing here is invented, estimated, or passed through
an LLM. graph/sop_matcher.py evaluates numeric SOPs directly against this
dict in plain Python (no LLM in the loop for numeric matching), and
graph/compose.py is instructed to only ever restate values that appear in
this dict, never to "recall" a number.

We request both `current` (for the instantaneous reading) and `hourly`
(for precipitation_probability and uv_index, which Open-Meteo does not
reliably expose on the `current` block) plus `daily` aggregates
(precipitation_sum, precipitation_probability_max, wind_speed_10m_max) which
we need to recognize a "sustained, day-long severe system" signal
(SOP-012) rather than judging severity off a single instantaneous reading.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT_SECONDS = 8


class WeatherLookupError(Exception):
    """Raised for any failure resolving a location or fetching weather.

    The graph treats every subclass of this identically: route to the
    honest-fallback node. We keep subclasses only so logs/evals can tell
    geocoding failures apart from forecast-API failures.
    """


class GeocodingError(WeatherLookupError):
    pass


class ForecastError(WeatherLookupError):
    pass


@dataclass
class ResolvedLocation:
    name: str
    country: Optional[str]
    latitude: float
    longitude: float
    candidate_count: int  # how many results geocoding returned; 1 = unambiguous


@dataclass
class WeatherSnapshot:
    location: ResolvedLocation
    time: str
    temperature_2m: Optional[float]
    wind_speed_10m: Optional[float]
    precipitation: Optional[float]
    precipitation_probability: Optional[float]
    uv_index: Optional[float]
    # Day-level aggregates, used for "sustained system" style judgments.
    daily_precipitation_sum: Optional[float]
    daily_precipitation_probability_max: Optional[float]
    daily_wind_speed_10m_max: Optional[float]
    raw: dict = field(default_factory=dict, repr=False)

    def as_fact_dict(self) -> dict:
        """Flat dict of the fields SOP conditions are evaluated against."""
        return {
            "temperature_2m": self.temperature_2m,
            "wind_speed_10m": self.wind_speed_10m,
            "precipitation": self.precipitation,
            "precipitation_probability": self.precipitation_probability,
            "uv_index": self.uv_index,
            "daily_precipitation_sum": self.daily_precipitation_sum,
            "daily_precipitation_probability_max": self.daily_precipitation_probability_max,
            "daily_wind_speed_10m_max": self.daily_wind_speed_10m_max,
        }


def geocode_city(city_name: str) -> ResolvedLocation:
    """Resolve a free-text city name to coordinates.

    If geocoding returns zero results, or the HTTP call fails/errors, we
    raise GeocodingError uniformly -- the graph routes this exactly like a
    forecast-API outage (an honest "can't resolve this" fallback), per the
    assignment's explicit instruction that these are the same failure mode.
    """
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"name": city_name, "count": 5, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise GeocodingError(f"Geocoding request failed for '{city_name}': {exc}") from exc
    except ValueError as exc:  # bad JSON
        raise GeocodingError(f"Geocoding returned unparseable data for '{city_name}': {exc}") from exc

    results = data.get("results") or []
    if not results:
        raise GeocodingError(f"No location found for '{city_name}'.")

    # Picking the first candidate silently is the assignment's stated
    # reasonable default. We keep candidate_count so the answer can
    # mention ambiguity ("there are multiple places named X") if useful.
    top = results[0]
    return ResolvedLocation(
        name=top.get("name", city_name),
        country=top.get("country"),
        latitude=top["latitude"],
        longitude=top["longitude"],
        candidate_count=len(results),
    )


def fetch_weather(location: ResolvedLocation) -> WeatherSnapshot:
    """Fetch current + hourly + daily weather for a resolved location."""
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": "temperature_2m,wind_speed_10m,precipitation",
        "hourly": "precipitation_probability,uv_index",
        "daily": "precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise ForecastError(f"Forecast request failed: {exc}") from exc
    except ValueError as exc:
        raise ForecastError(f"Forecast API returned unparseable data: {exc}") from exc

    current = data.get("current")
    hourly = data.get("hourly")
    daily = data.get("daily")
    if not current or not hourly:
        # This is the "got metadata, no actual values" failure mode the
        # assignment warns about explicitly -- treat it as an honest failure,
        # not a KeyError that crashes the graph.
        raise ForecastError("Forecast API response missing expected current/hourly weather fields.")

    current_time = current.get("time")
    precip_prob, uv = _match_hourly_to_current_time(hourly, current_time)

    daily_precip_sum = _first_or_none(daily, "precipitation_sum")
    daily_precip_prob_max = _first_or_none(daily, "precipitation_probability_max")
    daily_wind_max = _first_or_none(daily, "wind_speed_10m_max")

    return WeatherSnapshot(
        location=location,
        time=current_time,
        temperature_2m=current.get("temperature_2m"),
        wind_speed_10m=current.get("wind_speed_10m"),
        precipitation=current.get("precipitation"),
        precipitation_probability=precip_prob,
        uv_index=uv,
        daily_precipitation_sum=daily_precip_sum,
        daily_precipitation_probability_max=daily_precip_prob_max,
        daily_wind_speed_10m_max=daily_wind_max,
        raw=data,
    )


def _match_hourly_to_current_time(hourly: dict, current_time: Optional[str]):
    """Find the hourly index closest to `current_time` and pull values there."""
    times = hourly.get("time") or []
    if not times:
        return None, None

    idx = 0
    if current_time:
        try:
            target = dt.datetime.fromisoformat(current_time)
            best_idx, best_diff = 0, None
            for i, t in enumerate(times):
                try:
                    parsed = dt.datetime.fromisoformat(t)
                except ValueError:
                    continue
                diff = abs((parsed - target).total_seconds())
                if best_diff is None or diff < best_diff:
                    best_diff, best_idx = diff, i
            idx = best_idx
        except ValueError:
            idx = 0

    precip_prob_list = hourly.get("precipitation_probability") or []
    uv_list = hourly.get("uv_index") or []
    precip_prob = precip_prob_list[idx] if idx < len(precip_prob_list) else None
    uv = uv_list[idx] if idx < len(uv_list) else None
    return precip_prob, uv


def _first_or_none(daily: Optional[dict], key: str):
    if not daily:
        return None
    values = daily.get(key) or []
    return values[0] if values else None
