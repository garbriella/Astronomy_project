"""NOAA IGRA 2.2 다운로드, 고정폭 파싱 및 열역학 계산 도구."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import re
from typing import Any, Callable, Iterable
from urllib.parse import urljoin
import zipfile

from metpy.calc import (
    el,
    k_index,
    lcl,
    lfc,
    lifted_index,
    parcel_profile,
    precipitable_water,
    surface_based_cape_cin,
    total_totals_index,
)
from metpy.units import units
import numpy as np
import pandas as pd
import requests


STATION_LIST_URL = (
    "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
    "doc/igra2-station-list.txt"
)
RECENT_DATA_DIRECTORY_URL = (
    "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
    "access/data-y2d/"
)
DATA_FORMAT_URL = (
    "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
    "doc/igra2-data-format.txt"
)
KST = "Asia/Seoul"
REQUEST_HEADERS = {"User-Agent": "AtmosScope/1.0 (educational IGRA analysis)"}
MISSING_CODES = {-8888, -9999}
MIN_THERMO_LEVELS = 6


@dataclass(frozen=True)
class Station:
    """IGRA 관측소 목록의 한 행."""

    station_id: str
    latitude: float
    longitude: float
    elevation_m: float
    name: str
    first_year: int | None
    last_year: int | None
    observations: int | None


def _get_text(url: str, timeout: int = 30) -> str:
    """공개 HTTPS 텍스트 자료를 내려받고 HTTP 오류를 발생시킨다."""

    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_station_list() -> pd.DataFrame:
    """NOAA IGRA 고정폭 관측소 목록을 DataFrame으로 반환한다."""

    return parse_station_list(_get_text(STATION_LIST_URL))


def parse_station_list(text: str) -> pd.DataFrame:
    """IGRA station-list의 공식 고정폭 열을 파싱한다."""

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if len(line) < 71:
            continue
        station_id = line[0:11].strip()
        if not station_id:
            continue
        try:
            latitude = float(line[12:20])
            longitude = float(line[21:30])
            elevation = float(line[31:37])
        except ValueError:
            continue

        def as_int(value: str) -> int | None:
            try:
                return int(value.strip())
            except ValueError:
                return None

        rows.append(
            {
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": elevation,
                "name": line[41:71].strip(),
                "first_year": as_int(line[72:76]),
                "last_year": as_int(line[77:81]),
                "observations": as_int(line[82:88]),
            }
        )
    return pd.DataFrame(rows)


def fetch_recent_directory() -> str:
    """NOAA IGRA 최근자료 디렉터리 HTML을 반환한다."""

    return _get_text(RECENT_DATA_DIRECTORY_URL)


def discover_recent_filename(directory_html: str, station_id: str) -> str:
    """디렉터리 HTML에서 관측소별 data-begYYYY ZIP 파일명을 찾는다."""

    pattern = re.compile(
        rf"({re.escape(station_id)}-data-beg\d{{4}}\.txt\.zip)", re.IGNORECASE
    )
    matches = sorted(set(pattern.findall(directory_html)))
    if not matches:
        raise FileNotFoundError(f"{station_id}의 NOAA 최근자료 파일을 찾지 못했습니다.")
    return matches[-1]


def available_recent_station_ids(directory_html: str) -> set[str]:
    """현재 디렉터리에 ZIP 자료가 있는 모든 관측소 ID를 반환한다."""

    return set(
        re.findall(
            r"([A-Z0-9]{11})-data-beg\d{4}\.txt\.zip",
            directory_html,
            flags=re.IGNORECASE,
        )
    )


def recent_data_url(directory_html: str, station_id: str) -> str:
    """탐색된 최근자료 파일의 절대 URL을 만든다."""

    return urljoin(
        RECENT_DATA_DIRECTORY_URL,
        discover_recent_filename(directory_html, station_id),
    )


def download_zip_bytes(url: str, timeout: int = 60) -> bytes:
    """IGRA ZIP 파일을 메모리용 bytes로 내려받는다."""

    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.content


def extract_igra_text(zip_bytes: bytes) -> str:
    """디스크 저장 없이 메모리에서 첫 IGRA TXT 파일을 해제한다."""

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if not names:
                raise ValueError("ZIP 안에 IGRA TXT 파일이 없습니다.")
            with archive.open(names[0]) as source:
                return source.read().decode("ascii", errors="replace")
    except zipfile.BadZipFile as exc:
        raise ValueError("NOAA에서 받은 ZIP 파일이 손상되었습니다.") from exc


def _fixed_int(line: str, start: int, end: int) -> int | None:
    """고정폭 정수 필드를 안전하게 읽는다."""

    try:
        return int(line[start:end].strip())
    except (TypeError, ValueError):
        return None


def _valid_scaled(value: int | None, scale: float = 1.0) -> float:
    """IGRA 누락/QA 제거 코드를 NaN으로 바꾸고 배율을 적용한다."""

    if value is None or value in MISSING_CODES:
        return np.nan
    return float(value) / scale


def _observation_datetimes(
    year: int | None, month: int | None, day: int | None, hour: int | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """명목 관측 시각으로 UTC와 KST Timestamp를 만든다."""

    if None in (year, month, day, hour) or hour == 99:
        return pd.NaT, pd.NaT
    try:
        utc = pd.Timestamp(datetime(year, month, day, hour), tz="UTC")
        return utc, utc.tz_convert(KST)
    except (TypeError, ValueError):
        return pd.NaT, pd.NaT


def parse_igra_soundings(text: str) -> dict[pd.Timestamp, pd.DataFrame]:
    """IGRA 2.0~2.2 sounding 텍스트를 관측 시각별 DataFrame으로 변환한다.

    슬라이스는 NOAA 문서의 1-based 열 번호를 Python의 0-based, end-exclusive
    인덱스로 직접 변환한다. 기압은 hPa, 기온·이슬점차는 °C, 풍속은 m/s이다.
    """

    lines = text.splitlines()
    soundings: dict[pd.Timestamp, pd.DataFrame] = {}
    index = 0
    while index < len(lines):
        header = lines[index]
        if not header.startswith("#"):
            index += 1
            continue
        if len(header) < 36:
            raise ValueError(f"IGRA 헤더 길이가 너무 짧습니다(행 {index + 1}).")

        station_id = header[1:12].strip()
        year = _fixed_int(header, 13, 17)
        month = _fixed_int(header, 18, 20)
        day = _fixed_int(header, 21, 23)
        hour = _fixed_int(header, 24, 26)
        release_time = _fixed_int(header, 27, 31)
        num_levels = _fixed_int(header, 32, 36)
        if num_levels is None or num_levels < 0:
            raise ValueError(f"IGRA NUMLEV 필드를 읽을 수 없습니다(행 {index + 1}).")
        utc, kst = _observation_datetimes(year, month, day, hour)
        rows: list[dict[str, Any]] = []

        for level_line in lines[index + 1 : index + 1 + num_levels]:
            if level_line.startswith("#"):
                break
            padded = level_line.ljust(51)
            pressure = _valid_scaled(_fixed_int(padded, 9, 15), 100.0)
            height = _valid_scaled(_fixed_int(padded, 16, 21))
            temperature = _valid_scaled(_fixed_int(padded, 22, 27), 10.0)
            humidity = _valid_scaled(_fixed_int(padded, 28, 33), 10.0)
            depression = _valid_scaled(_fixed_int(padded, 34, 39), 10.0)
            wind_direction = _valid_scaled(_fixed_int(padded, 40, 45))
            wind_speed = _valid_scaled(_fixed_int(padded, 46, 51), 10.0)

            if pressure <= 0:
                pressure = np.nan
            if not 0 <= humidity <= 100:
                humidity = np.nan
            if depression < 0:
                depression = np.nan
            if not 0 <= wind_direction <= 360:
                wind_direction = np.nan
            if wind_speed < 0:
                wind_speed = np.nan
            dewpoint = temperature - depression
            if not np.isfinite(temperature) or not np.isfinite(depression):
                dewpoint = np.nan
            # 음의 이슬점차 또는 반올림 허용치를 넘는 Td > T는 물리적으로 부적절하다.
            if np.isfinite(dewpoint) and dewpoint > temperature + 0.05:
                dewpoint = np.nan

            pressure_flag = padded[15:16].strip()
            height_flag = padded[21:22].strip()
            temperature_flag = padded[27:28].strip()
            rows.append(
                {
                    "station_id": station_id,
                    "observation_year": year,
                    "observation_month": month,
                    "observation_day": day,
                    "observation_hour_utc": hour,
                    "release_time_utc_hhmm": release_time,
                    "datetime_utc": utc,
                    "datetime_kst": kst,
                    "pressure_hpa": pressure,
                    "geopotential_height_m": height,
                    "temperature_c": temperature,
                    "relative_humidity_pct": humidity,
                    "dewpoint_depression_c": depression,
                    "dewpoint_c": dewpoint,
                    "wind_direction_deg": wind_direction,
                    "wind_speed_ms": wind_speed,
                    "level_type_major": _fixed_int(padded, 0, 1),
                    "level_type_minor": _fixed_int(padded, 1, 2),
                    "pressure_qc_flag": pressure_flag,
                    "height_qc_flag": height_flag,
                    "temperature_qc_flag": temperature_flag,
                    "quality_flags": "/".join(
                        flag or "-"
                        for flag in (pressure_flag, height_flag, temperature_flag)
                    ),
                }
            )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            pressure_rows = frame[frame["pressure_hpa"].notna()].copy()
            pressure_rows = pressure_rows.sort_values("pressure_hpa", ascending=False)
            pressure_rows = pressure_rows.drop_duplicates("pressure_hpa", keep="first")
            other_rows = frame[frame["pressure_hpa"].isna()].copy()
            frame = pd.concat([pressure_rows, other_rows], ignore_index=True)
            key = utc if not pd.isna(utc) else pd.Timestamp(f"1970-01-01", tz="UTC") + pd.Timedelta(index, "ns")
            soundings[key] = frame
        index += num_levels + 1
    return dict(sorted(soundings.items(), key=lambda item: item[0]))


def thermodynamic_profile(frame: pd.DataFrame, min_pressure_hpa: float = 100.0) -> pd.DataFrame:
    """열역학 계산에 쓸 유효·고유 기압 레벨만 높은 기압부터 반환한다."""

    required = ["pressure_hpa", "temperature_c", "dewpoint_c"]
    if frame.empty or any(column not in frame for column in required):
        return pd.DataFrame(columns=frame.columns)
    profile = frame.dropna(subset=required).copy()
    profile = profile[profile["pressure_hpa"] >= min_pressure_hpa]
    profile = profile[profile["dewpoint_c"] <= profile["temperature_c"] + 0.05]
    profile = profile.sort_values("pressure_hpa", ascending=False)
    return profile.drop_duplicates("pressure_hpa", keep="first").reset_index(drop=True)


def _quantity_value(value: Any, unit: str | None = None) -> float:
    """MetPy/Pint 스칼라 Quantity를 float로 바꾼다."""

    if value is None:
        return np.nan
    try:
        converted = value.to(unit) if unit else value
        result = float(np.asarray(converted.magnitude).squeeze())
        return result if np.isfinite(result) else np.nan
    except (AttributeError, TypeError, ValueError):
        return np.nan


def _interp_at_pressure(profile: pd.DataFrame, column: str, target: float) -> float:
    """로그 기압 좌표에서 한 변수를 선형 보간한다."""

    subset = profile[["pressure_hpa", column]].dropna().drop_duplicates("pressure_hpa")
    if len(subset) < 2:
        return np.nan
    pressures = subset["pressure_hpa"].to_numpy(dtype=float)
    values = subset[column].to_numpy(dtype=float)
    if target < pressures.min() or target > pressures.max():
        return np.nan
    order = np.argsort(np.log(pressures))
    return float(np.interp(np.log(target), np.log(pressures[order]), values[order]))


def _height_at_pressure(profile: pd.DataFrame, target: float) -> float:
    """관측 지위고도로 특정 기압의 고도를 로그 보간한다."""

    return _interp_at_pressure(profile, "geopotential_height_m", target)


def calculate_sounding_metrics(
    frame: pd.DataFrame, min_pressure_hpa: float = 100.0
) -> dict[str, Any]:
    """한 sounding의 MetPy 지표를 계산하고 개별 실패 이유를 함께 반환한다."""

    metrics: dict[str, Any] = {
        "valid_levels": 0,
        "cape_jkg": np.nan,
        "cin_jkg": np.nan,
        "lcl_pressure_hpa": np.nan,
        "lcl_height_m": np.nan,
        "lfc_pressure_hpa": np.nan,
        "lfc_height_m": np.nan,
        "el_pressure_hpa": np.nan,
        "el_height_m": np.nan,
        "precipitable_water_mm": np.nan,
        "k_index_c": np.nan,
        "total_totals_c": np.nan,
        "lifted_index_c": np.nan,
        "lapse_rate_850_500_c_km": np.nan,
        "lowest_temperature_c": np.nan,
        "lowest_dewpoint_c": np.nan,
        "lowest_depression_c": np.nan,
        "temperature_850_c": np.nan,
        "dewpoint_850_c": np.nan,
        "temperature_500_c": np.nan,
        "quality_status": "계산 불가",
        "errors": [],
        "parcel_temperature_c": [],
        "parcel_pressure_hpa": [],
    }
    profile = thermodynamic_profile(frame, min_pressure_hpa)
    metrics["valid_levels"] = len(profile)
    if profile.empty:
        metrics["errors"].append("기압·기온·이슬점이 모두 유효한 레벨이 없습니다.")
        return metrics

    lowest = profile.iloc[0]
    metrics["lowest_temperature_c"] = float(lowest["temperature_c"])
    metrics["lowest_dewpoint_c"] = float(lowest["dewpoint_c"])
    metrics["lowest_depression_c"] = float(lowest["temperature_c"] - lowest["dewpoint_c"])
    for pressure_level, prefix in ((850.0, "850"), (500.0, "500")):
        metrics[f"temperature_{prefix}_c"] = _interp_at_pressure(
            profile, "temperature_c", pressure_level
        )
        if pressure_level == 850:
            metrics["dewpoint_850_c"] = _interp_at_pressure(
                profile, "dewpoint_c", pressure_level
            )

    if len(profile) < MIN_THERMO_LEVELS:
        metrics["quality_status"] = "유효 레벨 부족"
        metrics["errors"].append("유효한 연직 관측 레벨이 부족하여 계산할 수 없음")
        return metrics

    pressure = profile["pressure_hpa"].to_numpy() * units.hPa
    temperature = profile["temperature_c"].to_numpy() * units.degC
    dewpoint = profile["dewpoint_c"].to_numpy() * units.degC

    parcel = None
    try:
        parcel = parcel_profile(pressure, temperature[0], dewpoint[0]).to("degC")
        metrics["parcel_temperature_c"] = parcel.magnitude.tolist()
        metrics["parcel_pressure_hpa"] = pressure.magnitude.tolist()
    except Exception as exc:  # MetPy는 자료별로 여러 예외 형식을 낸다.
        metrics["errors"].append(f"공기덩이 상승경로: {exc}")

    try:
        cape, cin = surface_based_cape_cin(pressure, temperature, dewpoint)
        metrics["cape_jkg"] = _quantity_value(cape, "joule / kilogram")
        metrics["cin_jkg"] = _quantity_value(cin, "joule / kilogram")
    except Exception as exc:
        metrics["errors"].append(f"CAPE/CIN: {exc}")

    try:
        lcl_p, _ = lcl(pressure[0], temperature[0], dewpoint[0])
        metrics["lcl_pressure_hpa"] = _quantity_value(lcl_p, "hPa")
        metrics["lcl_height_m"] = _height_at_pressure(profile, metrics["lcl_pressure_hpa"])
    except Exception as exc:
        metrics["errors"].append(f"LCL: {exc}")

    if parcel is not None:
        for label, function in (("lfc", lfc), ("el", el)):
            try:
                result_p, _ = function(
                    pressure, temperature, dewpoint, parcel_temperature_profile=parcel
                )
                result = _quantity_value(result_p, "hPa")
                metrics[f"{label}_pressure_hpa"] = result
                metrics[f"{label}_height_m"] = _height_at_pressure(profile, result)
            except Exception as exc:
                metrics["errors"].append(f"{label.upper()}: {exc}")

    calculations: Iterable[tuple[str, Callable[[], Any], str]] = (
        (
            "precipitable_water_mm",
            lambda: precipitable_water(pressure, dewpoint),
            "millimeter",
        ),
        ("k_index_c", lambda: k_index(pressure, temperature, dewpoint), "delta_degC"),
        (
            "total_totals_c",
            lambda: total_totals_index(pressure, temperature, dewpoint),
            "delta_degC",
        ),
        (
            "lifted_index_c",
            lambda: lifted_index(pressure, temperature, parcel)[0] if parcel is not None else np.nan,
            "delta_degC",
        ),
    )
    for name, calculation, unit in calculations:
        try:
            metrics[name] = _quantity_value(calculation(), unit)
        except Exception as exc:
            metrics["errors"].append(f"{name}: {exc}")

    t850 = metrics["temperature_850_c"]
    t500 = metrics["temperature_500_c"]
    z850 = _height_at_pressure(profile, 850.0)
    z500 = _height_at_pressure(profile, 500.0)
    if all(np.isfinite(value) for value in (t850, t500, z850, z500)) and z500 > z850:
        metrics["lapse_rate_850_500_c_km"] = (t850 - t500) / ((z500 - z850) / 1000)

    metrics["quality_status"] = "양호" if len(profile) >= 12 else "제한적"
    return metrics


def calculate_period_metrics(
    soundings: dict[pd.Timestamp, pd.DataFrame], min_pressure_hpa: float = 100.0
) -> pd.DataFrame:
    """여러 관측 시각의 지표를 시계열 DataFrame으로 계산한다."""

    rows: list[dict[str, Any]] = []
    for timestamp, frame in soundings.items():
        result = calculate_sounding_metrics(frame, min_pressure_hpa)
        result.pop("parcel_temperature_c", None)
        result.pop("parcel_pressure_hpa", None)
        errors = result.pop("errors", [])
        rows.append(
            {
                "datetime_utc": timestamp,
                "datetime_kst": timestamp.tz_convert(KST),
                **result,
                "calculation_notes": " | ".join(errors),
            }
        )
    return pd.DataFrame(rows).sort_values("datetime_utc") if rows else pd.DataFrame()
