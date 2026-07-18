"""FITS 이미지 탐색, 통계, 명암 변환 및 천체좌표 계산 도구."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u
import numpy as np
import pandas as pd


SEOUL_LOCATION = EarthLocation(
    lat=37.5665 * u.deg,
    lon=126.9780 * u.deg,
    height=50 * u.m,
)
iers.conf.auto_download = False


@dataclass(frozen=True)
class ImageHDUInfo:
    """이미지 배열을 가진 FITS HDU의 표시 정보."""

    index: int
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def label(self) -> str:
        """Streamlit 선택 상자용 문자열을 반환한다."""

        return f"#{self.index} {self.name} · {self.shape} · {self.dtype}"


HEADER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "object": ("OBJECT",),
    "exposure": ("EXPTIME", "EXPOSURE"),
    "observation_time": ("DATE-OBS", "DATEOBS"),
    "filter": ("FILTER", "FILTER1", "FILTER2"),
    "telescope": ("TELESCOP",),
    "instrument": ("INSTRUME",),
    "ra": ("RA", "OBJRA", "CRVAL1"),
    "dec": ("DEC", "OBJDEC", "CRVAL2"),
}


def find_image_hdus(hdulist: fits.HDUList) -> list[ImageHDUInfo]:
    """모든 HDU를 검사해 2차원 이상 숫자 배열을 가진 항목을 찾는다."""

    results: list[ImageHDUInfo] = []
    for index, hdu in enumerate(hdulist):
        try:
            data = hdu.data
        except Exception:
            continue
        if isinstance(data, np.ndarray) and data.ndim >= 2 and np.issubdtype(data.dtype, np.number):
            results.append(
                ImageHDUInfo(
                    index=index,
                    name=str(hdu.name or "PRIMARY"),
                    shape=tuple(int(value) for value in data.shape),
                    dtype=str(data.dtype),
                )
            )
    return results


def extract_image_plane(data: np.ndarray, flat_slice_index: int = 0) -> np.ndarray:
    """2D 배열을 반환하거나 N-D 배열의 앞쪽 축을 평탄화해 한 평면을 고른다."""

    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("선택한 HDU에는 2차원 이미지가 없습니다.")
    if array.ndim == 2:
        return np.asarray(array, dtype=np.float64)
    plane_count = int(np.prod(array.shape[:-2]))
    if not 0 <= flat_slice_index < plane_count:
        raise IndexError(f"슬라이스 인덱스는 0부터 {plane_count - 1}까지여야 합니다.")
    return np.asarray(array.reshape((plane_count, *array.shape[-2:]))[flat_slice_index], dtype=np.float64)


def first_header_value(header: fits.Header, candidates: Iterable[str]) -> Any | None:
    """후보 헤더 키를 순서대로 검사해 첫 유효 값을 반환한다."""

    for key in candidates:
        value = header.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def extract_header_metadata(header: fits.Header) -> dict[str, Any]:
    """여러 제조사 표기를 고려해 분석에 필요한 FITS 메타데이터를 추출한다."""

    return {
        name: first_header_value(header, candidates)
        for name, candidates in HEADER_CANDIDATES.items()
    }


def display_header_value(value: Any | None, unit: str = "") -> str:
    """누락 헤더를 사용자 친화적인 문자열로 표시한다."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "정보 없음"
    return f"{value}{unit}"


def image_statistics(image: np.ndarray) -> dict[str, float | int]:
    """유효 픽셀만 사용해 FITS 이미지 요약 통계를 계산한다."""

    array = np.asarray(image, dtype=np.float64)
    finite = array[np.isfinite(array)]
    total = int(array.size)
    valid = int(finite.size)
    missing_ratio = 100.0 * (total - valid) / total if total else 100.0
    if not valid:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "valid_pixels": 0,
            "missing_ratio_pct": missing_ratio,
            "percentile_1": np.nan,
            "percentile_99_5": np.nan,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "valid_pixels": valid,
        "missing_ratio_pct": missing_ratio,
        "percentile_1": float(np.percentile(finite, 1)),
        "percentile_99_5": float(np.percentile(finite, 99.5)),
    }


def scale_image(
    image: np.ndarray,
    lower: float,
    upper: float,
    scale: str = "선형",
) -> np.ndarray:
    """선택 명암 범위를 0~1로 정규화하고 선형·로그·제곱근 변환을 적용한다."""

    array = np.asarray(image, dtype=np.float64)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return np.full_like(array, np.nan, dtype=np.float64)
    if upper <= lower:
        normalized = np.zeros_like(array, dtype=np.float64)
        normalized[~np.isfinite(array)] = np.nan
        return normalized
    normalized = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
    if scale == "로그":
        normalized = np.log1p(999.0 * normalized) / np.log(1000.0)
    elif scale == "제곱근":
        normalized = np.sqrt(normalized)
    elif scale != "선형":
        raise ValueError(f"지원하지 않는 명암 스케일입니다: {scale}")
    normalized[~np.isfinite(array)] = np.nan
    return normalized


def header_to_dataframe(header: fits.Header) -> pd.DataFrame:
    """전체 FITS 헤더를 키·값·설명 표로 변환한다."""

    rows = [
        {"키": card.keyword, "값": str(card.value), "설명": card.comment}
        for card in header.cards
        if card.keyword
    ]
    return pd.DataFrame(rows)


def _as_float(value: Any) -> float | None:
    """숫자형 또는 숫자 문자열을 float로 바꾼다."""

    try:
        result = float(str(value).strip())
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def parse_skycoord(ra_value: Any, dec_value: Any) -> SkyCoord:
    """다양한 RA/DEC 표기를 ICRS SkyCoord로 안전하게 해석한다.

    두 값이 단순 숫자이면 도 단위로 해석한다. RA가 시·분·초 표기라면
    hourangle, DEC는 degree로 해석하며 콜론/공백/hms 표기를 지원한다.
    """

    if ra_value is None or dec_value is None:
        raise ValueError("적경과 적위가 모두 필요합니다.")
    ra_number, dec_number = _as_float(ra_value), _as_float(dec_value)
    if ra_number is not None and dec_number is not None:
        return SkyCoord(ra=ra_number * u.deg, dec=dec_number * u.deg, frame="icrs")

    ra_text = str(ra_value).strip()
    dec_text = str(dec_value).strip()
    attempts = (
        lambda: SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg), frame="icrs"),
        lambda: SkyCoord(f"{ra_text} {dec_text}", unit=(u.hourangle, u.deg), frame="icrs"),
        lambda: SkyCoord(ra_text, dec_text, unit=(u.deg, u.deg), frame="icrs"),
    )
    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("RA/DEC 형식을 해석하지 못했습니다. 시:분:초/도:분:초 또는 도 단위 숫자를 입력하세요.")


def parse_observation_time(value: Any | None) -> Time | None:
    """FITS 관측 시각을 Astropy Time으로 해석하고 실패하면 None을 반환한다."""

    if value is None or not str(value).strip():
        return None
    try:
        return Time(value)
    except Exception:
        try:
            return Time(pd.Timestamp(value).to_pydatetime())
        except Exception:
            return None


def altitude_azimuth(
    coordinate: SkyCoord,
    observation_time: Time | None = None,
    location: EarthLocation = SEOUL_LOCATION,
) -> tuple[float, float]:
    """서울 기준 지정 시각의 고도와 방위각을 도 단위로 반환한다."""

    when = observation_time or Time.now()
    altaz = coordinate.transform_to(AltAz(obstime=when, location=location))
    return float(altaz.alt.deg), float(altaz.az.deg)
