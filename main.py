"""AstroWeather Lab 통합 Streamlit 애플리케이션."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from astropy.io import fits
import numpy as np
import pandas as pd
import streamlit as st

from fits_utils import (
    altitude_azimuth,
    display_header_value,
    extract_header_metadata,
    extract_image_plane,
    find_image_hdus,
    header_to_dataframe,
    image_statistics,
    parse_observation_time,
    parse_skycoord,
)
from plot_utils import (
    comparison_overlay_figure,
    fits_diagnostic_figure,
    fits_image_figure,
    make_skewt_figure,
    time_series_figure,
    vertical_profile_figures,
)
from weather_utils import (
    DATA_FORMAT_URL,
    KST,
    MIN_THERMO_LEVELS,
    RECENT_DATA_DIRECTORY_URL,
    STATION_LIST_URL,
    available_recent_station_ids,
    calculate_period_metrics,
    calculate_sounding_metrics,
    discover_recent_filename,
    download_zip_bytes,
    extract_igra_text,
    fetch_recent_directory,
    fetch_station_list,
    parse_igra_soundings,
    recent_data_url,
)


SAMPLE_FITS_URL = "https://drive.google.com/drive/folders/1PH6OfSRX5SUtImmrFtKbd4P6cl3e92hw"

st.set_page_config(
    page_title="AstroWeather Lab | 천문 이미지와 고층대기 분석기",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --navy:#132a4f; --sky:#eaf5fb; --teal:#0f8b8d; }
      .stApp { background: linear-gradient(180deg, #f4f8fc 0%, #ffffff 34rem); }
      h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
      [data-testid="stMetric"] {
        background:rgba(255,255,255,.94); border:1px solid #d8e8f1;
        border-radius:14px; padding:.78rem 1rem; box-shadow:0 5px 18px rgba(32,74,109,.05);
      }
      [data-testid="stSidebar"] { background:#eef5fa; }
      .source-box { margin-top:2rem; padding:1rem 1.2rem; border-radius:14px;
        background:#edf7fa; border:1px solid #cae4ea; }
      @media (max-width:700px) { [data-testid="stMetric"] { padding:.55rem .65rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_station_list() -> pd.DataFrame:
    """NOAA 관측소 목록을 하루 동안 캐시한다."""

    return fetch_station_list()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_recent_directory() -> str:
    """NOAA 최근자료 디렉터리 HTML을 6시간 동안 캐시한다."""

    return fetch_recent_directory()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_zip(url: str) -> bytes:
    """선택 관측소의 ZIP payload를 메모리에 캐시한다."""

    return download_zip_bytes(url)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_soundings(zip_payload: bytes) -> dict[pd.Timestamp, pd.DataFrame]:
    """메모리 ZIP 해제와 IGRA 파싱 결과를 캐시한다."""

    return parse_igra_soundings(extract_igra_text(zip_payload))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_period_metrics(
    profiles: dict[pd.Timestamp, pd.DataFrame], min_pressure_hpa: float
) -> pd.DataFrame:
    """선택 기간의 관측 시각별 열역학 지표를 캐시한다."""

    return calculate_period_metrics(profiles, min_pressure_hpa)


def format_number(value: Any, unit: str = "", digits: int = 1) -> str:
    """유효 숫자에는 단위를 붙이고 결측은 계산 불가로 표시한다."""

    try:
        if not np.isfinite(float(value)):
            return "계산 불가"
        return f"{float(value):,.{digits}f}{unit}"
    except (TypeError, ValueError):
        return "계산 불가"


def format_timestamp(value: Any, timezone: str = "UTC") -> str:
    """Timestamp를 UTC 또는 KST 표시 문자열로 바꾼다."""

    if value is None or pd.isna(value):
        return "자료 없음"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(timezone).strftime("%Y-%m-%d %H:%M %Z")


def show_error(message: str, exc: Exception | None = None) -> None:
    """간단한 사용자 오류와 선택적인 기술 상세를 분리해 표시한다."""

    st.error(message)
    if exc is not None:
        with st.expander("기술적 오류 내용"):
            st.code(f"{type(exc).__name__}: {exc}")


def metric_grid(items: list[tuple[str, str]], columns_per_row: int = 4) -> None:
    """라벨·값 쌍을 반응형 Streamlit 지표 카드로 표시한다."""

    for start in range(0, len(items), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, (label, value) in zip(columns, items[start : start + columns_per_row]):
            column.metric(label, value)


def display_stability_cards(metrics: dict[str, Any]) -> None:
    """주요 안정도 지표와 한 문장 설명을 카드로 표시한다."""

    cards = (
        ("CAPE", "cape_jkg", " J/kg", "상승 공기덩이의 양의 부력 에너지입니다."),
        ("CIN", "cin_jkg", " J/kg", "대류 시작을 억제하는 음의 에너지입니다."),
        ("LCL", "lcl_pressure_hpa", " hPa", "상승 공기가 포화되는 기압입니다."),
        ("LFC", "lfc_pressure_hpa", " hPa", "자유대류가 시작되는 기압입니다."),
        ("EL", "el_pressure_hpa", " hPa", "상승 공기덩이의 부력이 다시 0이 되는 기압입니다."),
        ("가강수량", "precipitable_water_mm", " mm", "대기 기둥 전체 수증기의 물 환산 깊이입니다."),
        ("K 지수", "k_index_c", " °C", "중층 기온감률과 습기를 함께 보는 지수입니다."),
        ("Total Totals", "total_totals_c", " °C", "850–500 hPa 불안정을 나타내는 경험 지수입니다."),
        ("Lifted Index", "lifted_index_c", " °C", "500 hPa 환경과 상승 공기덩이의 온도 차입니다."),
        ("850–500 기온감률", "lapse_rate_850_500_c_km", " °C/km", "850–500 hPa 층의 관측 기온감률입니다."),
        ("최하층 T-Td", "lowest_depression_c", " °C", "최하층 공기의 건조 정도를 나타냅니다."),
    )
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, (label, key, unit, explanation) in zip(columns, cards[start : start + 3]):
            with column:
                st.metric(label, format_number(metrics.get(key), unit))
                st.caption(explanation)


def combine_kst(day_value: date, time_value: time) -> pd.Timestamp:
    """날짜와 시각 입력을 timezone-aware KST Timestamp로 결합한다."""

    return pd.Timestamp(datetime.combine(day_value, time_value), tz=KST)


def normalize_event_time(value: Any) -> pd.Timestamp:
    """업로드된 사례 시각을 timezone-aware KST Timestamp로 정규화한다."""

    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(KST) if timestamp.tzinfo is None else timestamp.tz_convert(KST)


def choose_event_soundings(
    metrics: pd.DataFrame,
    start_kst: pd.Timestamp,
    end_kst: pd.Timestamp,
    tolerance_hours: int,
) -> tuple[pd.Series | None, pd.Series | None, str | None]:
    """허용범위 안에서 시작 전·종료 후 가장 가까운 유효 sounding을 고른다."""

    if end_kst <= start_kst:
        return None, None, "강수 종료 시각은 시작 시각보다 늦어야 합니다."
    valid = metrics[metrics["valid_levels"] >= MIN_THERMO_LEVELS].copy()
    before = valid[valid["datetime_kst"] < start_kst].sort_values("datetime_kst")
    after = valid[valid["datetime_kst"] > end_kst].sort_values("datetime_kst")
    if before.empty or after.empty:
        return None, None, "강수 전후에 비교 가능한 유효 sounding이 없습니다."
    before_row, after_row = before.iloc[-1], after.iloc[0]
    tolerance = pd.Timedelta(hours=tolerance_hours)
    if start_kst - before_row["datetime_kst"] > tolerance:
        return None, None, "강수 시작 전 허용범위 안에 sounding이 없습니다."
    if after_row["datetime_kst"] - end_kst > tolerance:
        return None, None, "강수 종료 후 허용범위 안에 sounding이 없습니다."
    return before_row, after_row, None


def delta_value(after: pd.Series, before: pd.Series, key: str) -> float:
    """두 값이 유효할 때 후-전 변화량을 반환한다."""

    first, second = before.get(key, np.nan), after.get(key, np.nan)
    return float(second - first) if np.isfinite(first) and np.isfinite(second) else np.nan


st.title("AstroWeather Lab | 천문 이미지와 고층대기 분석기")
st.markdown(
    "FITS 천문 이미지 처리 기능을 기반으로 NOAA 라디오존데 자료를 활용한 "
    "대기 수직구조와 강수 전후 안정도 변화를 함께 탐색합니다."
)
st.caption("FITS 분석은 기본 앱 제작 기능이며 NOAA 분석은 독립적인 기상학 심화탐구 기능입니다. 두 자료 사이의 직접적인 인과관계를 가정하지 않습니다.")

tabs = st.tabs(
    [
        "FITS 이미지 분석",
        "천체 위치",
        "NOAA 고층대기",
        "대기 안정도",
        "강수 전후 비교",
        "자료 및 계산 방법",
    ]
)

# FITS 상태는 NOAA 연결과 완전히 분리한다.
fits_file = None
fits_image: np.ndarray | None = None
fits_header: fits.Header | None = None
fits_metadata: dict[str, Any] = {}
fits_hdu_label = "정보 없음"
fits_error: Exception | None = None

with tabs[0]:
    st.subheader("FITS 이미지 분석")
    st.link_button("샘플 FITS 파일 받기", SAMPLE_FITS_URL, icon="↗")
    st.markdown(
        "1. 공유 폴더에서 FITS 파일을 다운로드합니다.  \n"
        "2. 다운로드한 파일을 아래 업로드 창에 올립니다.  \n"
        "3. 파일은 GitHub에 올릴 필요가 없습니다.  \n"
        "4. 앱은 업로드된 파일을 현재 실행 세션에서만 처리합니다."
    )
    fits_file = st.file_uploader(
        "FITS 파일 업로드",
        type=["fits", "fit", "fts", "fz"],
        key="fits_uploader",
        help="파일은 서버나 외부 데이터베이스에 영구 저장하지 않습니다.",
    )
    if fits_file is None:
        st.info("FITS 파일을 업로드하면 이미지·헤더·밝기 통계를 분석할 수 있습니다.")
    else:
        try:
            fits_file.seek(0)
            with fits.open(fits_file, memmap=False) as hdulist:
                image_hdus = find_image_hdus(hdulist)
                if not image_hdus:
                    raise ValueError("모든 HDU를 검사했지만 실제 이미지 배열을 찾지 못했습니다.")
                hdu_labels = {item.label: item for item in image_hdus}
                selected_hdu_label = st.selectbox("이미지 HDU 선택", list(hdu_labels))
                selected_hdu = hdu_labels[selected_hdu_label]
                fits_hdu_label = f"#{selected_hdu.index} {selected_hdu.name}"
                hdu = hdulist[selected_hdu.index]
                raw_data = np.asarray(hdu.data)
                plane_count = int(np.prod(raw_data.shape[:-2])) if raw_data.ndim > 2 else 1
                slice_index = 0
                if raw_data.ndim > 2:
                    st.info(
                        f"{raw_data.ndim}차원 배열의 앞쪽 축 {raw_data.shape[:-2]}을 평탄화한 "
                        f"{plane_count}개 2차원 평면 중 하나를 사용합니다. 기본값은 첫 번째 평면입니다."
                    )
                    slice_index = st.slider("2차원 평면 슬라이스 인덱스", 0, plane_count - 1, 0)
                fits_image = extract_image_plane(raw_data, slice_index).copy()
                # 확장 HDU를 선택해도 기본 HDU의 공통 관측 메타데이터를 잃지 않는다.
                fits_header = hdulist[0].header.copy()
                if selected_hdu.index != 0:
                    fits_header.extend(hdu.header, update=True, unique=False)
                fits_metadata = extract_header_metadata(fits_header)

            stats = image_statistics(fits_image)
            metric_grid(
                [
                    ("파일명", fits_file.name),
                    ("HDU", fits_hdu_label),
                    ("이미지 크기", f"{fits_image.shape[1]} × {fits_image.shape[0]} px"),
                    ("데이터 차원", f"{raw_data.ndim}D"),
                    ("데이터 형식", str(raw_data.dtype)),
                    ("관측 대상", display_header_value(fits_metadata.get("object"))),
                    ("노출 시간", display_header_value(fits_metadata.get("exposure"), " s")),
                    ("관측 시각", display_header_value(fits_metadata.get("observation_time"))),
                    ("필터", display_header_value(fits_metadata.get("filter"))),
                    ("망원경/기기", f"{display_header_value(fits_metadata.get('telescope'))} / {display_header_value(fits_metadata.get('instrument'))}"),
                    ("평균 밝기", format_number(stats["mean"], digits=3)),
                    ("중앙값 밝기", format_number(stats["median"], digits=3)),
                    ("표준편차", format_number(stats["std"], digits=3)),
                    ("최솟값", format_number(stats["min"], digits=3)),
                    ("최댓값", format_number(stats["max"], digits=3)),
                    ("유효 픽셀", f"{stats['valid_pixels']:,}개"),
                    ("결측 픽셀 비율", format_number(stats["missing_ratio_pct"], "%", 2)),
                ],
                columns_per_row=4,
            )
            if stats["valid_pixels"] == 0:
                st.warning("유효한 숫자 픽셀이 없어 명암 변환과 그래프를 만들 수 없습니다.")
            else:
                controls = st.columns(5)
                lower = controls[0].number_input("명암 하한", value=float(stats["percentile_1"]), format="%.6g")
                upper = controls[1].number_input("명암 상한", value=float(stats["percentile_99_5"]), format="%.6g")
                scale = controls[2].selectbox("명암 스케일", ["선형", "로그", "제곱근"])
                colorscale = controls[3].selectbox("컬러맵", ["Gray", "Viridis", "Cividis", "Turbo", "Hot"])
                orientation = controls[4].selectbox("이미지 방향", ["원본 방향", "상하 반전"])
                if upper <= lower:
                    st.warning("명암 상한은 하한보다 커야 합니다. 같은 값이면 검은 영상으로 안전하게 처리합니다.")
                st.plotly_chart(
                    fits_image_figure(fits_image, lower, upper, scale, colorscale, orientation == "상하 반전"),
                    use_container_width=True,
                )
                st.plotly_chart(fits_diagnostic_figure(fits_image), use_container_width=True)
            with st.expander("전체 FITS 헤더 보기"):
                st.dataframe(header_to_dataframe(fits_header), width="stretch", hide_index=True)
        except Exception as exc:
            fits_error = exc
            fits_image = None
            fits_header = None
            fits_metadata = {}
            show_error("FITS 파일을 처리하지 못했습니다. 파일 또는 선택 HDU 형식을 확인해 주세요.", exc)

# 천체좌표 탭은 헤더 좌표가 실패해도 수동 입력으로 독립 동작한다.
coordinate = None
coordinate_error: Exception | None = None
current_altaz: tuple[float, float] | None = None
observed_altaz: tuple[float, float] | None = None
observation_time = parse_observation_time(fits_metadata.get("observation_time"))
header_ra, header_dec = fits_metadata.get("ra"), fits_metadata.get("dec")
try:
    if header_ra is not None and header_dec is not None:
        coordinate = parse_skycoord(header_ra, header_dec)
except Exception as exc:
    coordinate_error = exc

with tabs[1]:
    st.subheader("서울 기준 천체 위치")
    st.caption("관측 위치: 위도 37.5665°, 경도 126.9780°, 고도 50 m")
    if fits_file is not None:
        st.write(
            f"FITS 헤더 좌표 — RA: `{display_header_value(header_ra)}`, "
            f"DEC: `{display_header_value(header_dec)}`"
        )
    if coordinate_error is not None:
        st.warning(f"FITS 헤더 좌표를 해석하지 못했습니다: {coordinate_error}")
    use_manual = coordinate is None or st.checkbox("헤더 좌표 대신 직접 입력", value=False)
    if use_manual:
        coordinate_columns = st.columns(2)
        manual_ra = coordinate_columns[0].text_input(
            "적경 RA",
            value=str(header_ra) if header_ra is not None else "",
            placeholder="예: 12:34:56 또는 188.733",
        )
        manual_dec = coordinate_columns[1].text_input(
            "적위 DEC",
            value=str(header_dec) if header_dec is not None else "",
            placeholder="예: +12:34:56 또는 12.582",
        )
        if manual_ra.strip() and manual_dec.strip():
            try:
                coordinate = parse_skycoord(manual_ra, manual_dec)
                coordinate_error = None
            except Exception as exc:
                coordinate = None
                coordinate_error = exc
                show_error("입력한 적경과 적위를 해석하지 못했습니다.", exc)
    if coordinate is None:
        st.info("FITS 헤더에 좌표가 없으면 적경과 적위를 직접 입력해 현재 천체 위치를 계산할 수 있습니다.")
    else:
        try:
            current_altaz = altitude_azimuth(coordinate)
            if observation_time is not None:
                observed_altaz = altitude_azimuth(coordinate, observation_time)
            metric_grid(
                [
                    ("적경 (ICRS)", coordinate.ra.to_string(unit="hour", sep=":", precision=2)),
                    ("적위 (ICRS)", coordinate.dec.to_string(unit="deg", sep=":", precision=2, alwayssign=True)),
                    ("현재 고도", f"{current_altaz[0]:.2f}°"),
                    ("현재 방위각", f"{current_altaz[1]:.2f}°"),
                    ("관측 시각", observation_time.isot if observation_time is not None else "정보 없음"),
                    ("관측 당시 고도", f"{observed_altaz[0]:.2f}°" if observed_altaz else "계산 불가"),
                    ("관측 당시 방위각", f"{observed_altaz[1]:.2f}°" if observed_altaz else "계산 불가"),
                ]
            )
            st.caption("고도는 지평선 위 각도, 방위각은 북쪽에서 동쪽 방향으로 잰 각도입니다.")
        except Exception as exc:
            show_error("서울 기준 고도와 방위각을 계산하지 못했습니다.", exc)

with st.sidebar:
    st.header("FITS 상태")
    st.write(f"파일: **{fits_file.name if fits_file else '업로드되지 않음'}**")
    if fits_error is not None:
        st.caption("FITS 처리 오류")
    st.write(f"관측 대상: {display_header_value(fits_metadata.get('object'))}")
    st.write(f"적경: {display_header_value(header_ra)}")
    st.write(f"적위: {display_header_value(header_dec)}")
    st.write(f"현재 고도: {f'{current_altaz[0]:.2f}°' if current_altaz else '계산 불가'}")
    st.write(f"현재 방위각: {f'{current_altaz[1]:.2f}°' if current_altaz else '계산 불가'}")
    st.write(f"관측 당시 고도: {f'{observed_altaz[0]:.2f}°' if observed_altaz else '계산 불가'}")
    st.write(f"관측 당시 방위각: {f'{observed_altaz[1]:.2f}°' if observed_altaz else '계산 불가'}")
    st.divider()
    st.header("NOAA 분석 조건")

# NOAA 네트워크/파싱 오류는 FITS 영역 밖에서만 처리한다.
noaa_errors: list[tuple[str, Exception]] = []
fallback_station = pd.DataFrame(
    [{"station_id": "KSM00047138", "latitude": 36.032, "longitude": 129.380, "elevation_m": 3.9, "name": "POHANG", "first_year": 1967, "last_year": None, "observations": None}]
)
try:
    all_stations = cached_station_list()
except Exception as exc:
    all_stations = fallback_station
    noaa_errors.append(("NOAA 관측소 목록 연결에 실패해 포항 기본정보만 사용합니다.", exc))
try:
    directory_html = cached_recent_directory()
    recent_ids = available_recent_station_ids(directory_html)
except Exception as exc:
    directory_html = ""
    recent_ids = {"KSM00047138"}
    noaa_errors.append(("NOAA 최근자료 디렉터리에 연결하지 못했습니다.", exc))

korean_stations = all_stations[all_stations["station_id"].str.startswith("KSM", na=False)].copy()
if directory_html:
    korean_stations = korean_stations[korean_stations["station_id"].isin(recent_ids)]
if korean_stations.empty:
    korean_stations = fallback_station
korean_stations["display_name"] = korean_stations.apply(lambda row: f"{row['name']} · {row['station_id']}", axis=1)
default_matches = korean_stations.index[korean_stations["station_id"] == "KSM00047138"].tolist()
default_index = korean_stations.index.get_loc(default_matches[0]) if default_matches else 0

with st.sidebar:
    selected_label = st.selectbox("관측소 선택", korean_stations["display_name"].tolist(), index=default_index)
    station = korean_stations[korean_stations["display_name"] == selected_label].iloc[0]
    st.markdown(
        f"**{station['name']}**  \nID `{station['station_id']}`  \n"
        f"위도 {station['latitude']:.4f}° · 경도 {station['longitude']:.4f}° · 고도 {station['elevation_m']:.1f} m"
    )
    st.map(pd.DataFrame({"lat": [station["latitude"]], "lon": [station["longitude"]]}), zoom=7)
    if st.button("NOAA 자료 새로고침", width="stretch"):
        st.cache_data.clear()
        st.rerun()

soundings: dict[pd.Timestamp, pd.DataFrame] = {}
source_url = ""
if directory_html:
    try:
        filename = discover_recent_filename(directory_html, station["station_id"])
        source_url = recent_data_url(directory_html, station["station_id"])
        with st.spinner(f"NOAA 최근자료({filename})를 불러오는 중…"):
            soundings = cached_soundings(cached_zip(source_url))
    except Exception as exc:
        noaa_errors.append(("선택 관측소의 최근 sounding 자료를 불러오지 못했습니다.", exc))

all_times = sorted(soundings)
if all_times:
    first_date, last_date = all_times[0].date(), all_times[-1].date()
    default_start = max(first_date, last_date - timedelta(days=14))
else:
    first_date = last_date = datetime.now().date()
    default_start = first_date

with st.sidebar:
    date_range = st.date_input("분석 시작일과 종료일 (UTC)", value=(default_start, last_date), min_value=first_date, max_value=last_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range if not isinstance(date_range, tuple) else default_start
    min_pressure = st.slider("표시할 최소 기압 (hPa)", 10, 500, 100, 10)

period_soundings = {timestamp: frame for timestamp, frame in soundings.items() if start_date <= timestamp.date() <= end_date}
period_times = sorted(period_soundings)
with st.sidebar:
    if period_times:
        labels = {f"{format_timestamp(ts, 'UTC')} / {format_timestamp(ts, KST)}": ts for ts in period_times}
        selected_time = labels[st.selectbox("단일 sounding 선택", list(labels), index=len(labels) - 1)]
    else:
        selected_time = None
        st.warning("선택 기간에 sounding이 없습니다.")
    st.caption("UTC는 협정세계시, KST는 UTC+9입니다. 고층관측은 주로 00/12 UTC에 시행되어 일정한 1시간 간격 자료가 아닙니다.")
    with st.expander("NOAA 분석 기준 도움말"):
        st.write(
            f"기압·기온·이슬점이 모두 유효한 최소 {MIN_THERMO_LEVELS}개 고유 기압 레벨이 "
            "있어야 주요 열역학 지표를 계산합니다. 중복 기압은 첫 레벨만 사용하고 "
            "기압이 높은 값에서 낮은 값 순서가 되도록 정렬합니다."
        )

selected_frame = period_soundings.get(selected_time, pd.DataFrame()) if selected_time is not None else pd.DataFrame()
display_frame = selected_frame[selected_frame["pressure_hpa"].isna() | (selected_frame["pressure_hpa"] >= min_pressure)].copy() if not selected_frame.empty else pd.DataFrame()
selected_metrics = calculate_sounding_metrics(display_frame, min_pressure) if not display_frame.empty else calculate_sounding_metrics(pd.DataFrame())
try:
    period_metrics = cached_period_metrics(period_soundings, min_pressure) if period_soundings else pd.DataFrame()
except Exception as exc:
    period_metrics = pd.DataFrame()
    noaa_errors.append(("관측 시각별 지표 계산 중 오류가 발생했습니다.", exc))

with tabs[2]:
    st.subheader("NOAA 고층대기")
    st.caption("NOAA IGRA 2.2 공개 라디오존데 자료의 관측 시각별 수직구조입니다. UTC와 KST를 함께 표시합니다.")
    for message, exc in noaa_errors:
        show_error(message, exc)
    top = st.columns(4)
    top[0].metric("관측소", station["name"])
    top[1].metric("기간 sounding", f"{len(period_soundings):,}개")
    top[2].metric("유효 연직 레벨", f"{selected_metrics['valid_levels']:,}개")
    top[3].metric("자료 품질", selected_metrics["quality_status"])
    st.caption(f"{station['station_id']} · 위도 {station['latitude']:.4f}° · 경도 {station['longitude']:.4f}° · 해발 {station['elevation_m']:.1f} m")
    if period_times:
        st.caption(f"최초 {format_timestamp(period_times[0], 'UTC')} / {format_timestamp(period_times[0], KST)} · 최종 {format_timestamp(period_times[-1], 'UTC')} / {format_timestamp(period_times[-1], KST)}")
    metric_grid(
        [
            ("최하층 기온", format_number(selected_metrics["lowest_temperature_c"], " °C")),
            ("최하층 이슬점", format_number(selected_metrics["lowest_dewpoint_c"], " °C")),
            ("CAPE", format_number(selected_metrics["cape_jkg"], " J/kg")),
            ("CIN", format_number(selected_metrics["cin_jkg"], " J/kg")),
            ("가강수량", format_number(selected_metrics["precipitable_water_mm"], " mm")),
        ],
        columns_per_row=5,
    )
    if display_frame.empty:
        st.warning("선택한 관측 시각에 표시할 연직 자료가 없습니다. NOAA 오류가 발생해도 FITS 탭은 계속 사용할 수 있습니다.")
    else:
        use_height = st.checkbox("고도 기반 축으로 표시", value=False, disabled=display_frame["geopotential_height_m"].notna().sum() < 2)
        try:
            profile_figures = vertical_profile_figures(display_frame, use_height)
            left, right = st.columns(2)
            left.plotly_chart(profile_figures[0], use_container_width=True)
            right.plotly_chart(profile_figures[1], use_container_width=True)
            st.plotly_chart(profile_figures[2], use_container_width=True)
            st.caption("풍향은 색으로, 풍속은 가로축으로 표시합니다. 바람깃은 대기 안정도 탭의 Skew-T에서 확인할 수 있습니다.")
        except Exception as exc:
            show_error("수직 대기 그래프를 만들지 못했습니다.", exc)
        with st.expander("선택 sounding 연직 자료"):
            st.dataframe(display_frame, width="stretch", hide_index=True)
            st.download_button("선택 sounding CSV 다운로드", display_frame.to_csv(index=False).encode("utf-8-sig"), file_name=f"{station['station_id']}_{selected_time.strftime('%Y%m%d%H')}_profile.csv", mime="text/csv")
    st.markdown("#### 선택 기간의 관측 시각별 변화")
    if period_metrics.empty:
        st.warning("선택 기간의 관측 시각별 지표가 없습니다.")
    else:
        time_basis = st.radio("시각 표시 기준", ["KST", "UTC"], horizontal=True)
        x_column = "datetime_kst" if time_basis == "KST" else "datetime_utc"
        series_specs = (
            ("cape_jkg", "CAPE 변화", "CAPE (J/kg)"),
            ("cin_jkg", "CIN 변화", "CIN (J/kg)"),
            ("precipitable_water_mm", "가강수량 변화", "가강수량 (mm)"),
            ("lcl_pressure_hpa", "LCL 기압 변화", "LCL (hPa)"),
            ("lapse_rate_850_500_c_km", "850–500 hPa 기온감률 변화", "기온감률 (°C/km)"),
            ("lowest_depression_c", "최하층 기온-이슬점 차 변화", "기온차 (°C)"),
        )
        for start in range(0, len(series_specs), 2):
            columns = st.columns(2)
            for column, (key, title, unit) in zip(columns, series_specs[start : start + 2]):
                column.plotly_chart(time_series_figure(period_metrics, x_column, key, title, unit), use_container_width=True)
        st.download_button("관측 시각별 전체 지표 CSV 다운로드", period_metrics.to_csv(index=False).encode("utf-8-sig"), file_name=f"{station['station_id']}_time_series.csv", mime="text/csv")

with tabs[3]:
    st.subheader("대기 안정도")
    chart_column, metric_column = st.columns([1.15, 1])
    with chart_column:
        if display_frame.empty:
            st.warning("Skew-T를 그릴 기온·이슬점 자료가 없습니다.")
        else:
            try:
                st.pyplot(make_skewt_figure(display_frame, f"{station['name']} · {format_timestamp(selected_time, 'UTC')}", min_pressure), width="stretch")
            except Exception as exc:
                show_error("Skew-T를 그리지 못했습니다.", exc)
    with metric_column:
        display_stability_cards(selected_metrics)
        if selected_metrics["errors"]:
            with st.expander("계산 불가 이유와 개별 지표 오류"):
                for error in selected_metrics["errors"]:
                    st.write(f"- {error}")
    st.info("CAPE가 크더라도 강수가 반드시 발생하는 것은 아닙니다. CIN, 수증기량, 강제상승, 전선과 기단 이동을 함께 살펴야 합니다.")

with tabs[4]:
    st.subheader("강수 전후 비교")
    st.caption("심화탐구: 강수 전후 포항 상공의 열역학적 안정도 변화 분석 — CAPE 감소율, CIN 회복과 수증기량 변화")
    input_column, upload_column = st.columns(2)
    events = pd.DataFrame()
    with input_column:
        event_name = st.text_input("강수 사례 이름", value="강수 사례 1")
        default_event_date = period_times[len(period_times) // 2].tz_convert(KST).date() if period_times else datetime.now().date()
        start_day = st.date_input("강수 시작 KST 날짜", value=default_event_date, key="rain_start_day")
        start_clock = st.time_input("강수 시작 KST 시각", value=time(9, 0), key="rain_start_time")
        end_day = st.date_input("강수 종료 KST 날짜", value=default_event_date, key="rain_end_day")
        end_clock = st.time_input("강수 종료 KST 시각", value=time(12, 0), key="rain_end_time")
        precipitation = st.number_input("총강수량 (mm, 선택)", min_value=0.0, value=None, step=0.1)
    with upload_column:
        tolerance_hours = st.slider("전후 sounding 탐색 허용 시간", 1, 72, 18)
        event_upload = st.file_uploader("강수 사례 CSV (선택)", type=["csv"], help="event_name, start_kst, end_kst, precipitation_mm 열")
        if event_upload is not None:
            try:
                uploaded_events = pd.read_csv(event_upload)
                required = ["event_name", "start_kst", "end_kst", "precipitation_mm"]
                missing = set(required) - set(uploaded_events.columns)
                if missing:
                    raise ValueError(f"필수 열 누락: {', '.join(sorted(missing))}")
                events = uploaded_events[required].copy()
            except Exception as exc:
                show_error("강수 사례 CSV를 읽을 수 없습니다.", exc)
    if events.empty:
        events = pd.DataFrame([{"event_name": event_name, "start_kst": combine_kst(start_day, start_clock), "end_kst": combine_kst(end_day, end_clock), "precipitation_mm": precipitation}])

    results: list[dict[str, Any]] = []
    pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    if period_metrics.empty:
        st.warning("강수 전후를 비교할 관측 시각별 지표가 없습니다.")
    else:
        for _, event in events.iterrows():
            try:
                start_kst = normalize_event_time(event["start_kst"])
                end_kst = normalize_event_time(event["end_kst"])
                before, after, warning = choose_event_soundings(period_metrics, start_kst, end_kst, tolerance_hours)
                if warning or before is None or after is None:
                    st.warning(f"{event['event_name']}: {warning}")
                    continue
                before_cape, after_cape = before["cape_jkg"], after["cape_jkg"]
                reduction = 100 * (before_cape - after_cape) / before_cape if np.isfinite(before_cape) and np.isfinite(after_cape) and abs(before_cape) > 1 else np.nan
                record = {
                    "event_name": event["event_name"], "start_kst": start_kst, "end_kst": end_kst, "precipitation_mm": event.get("precipitation_mm", np.nan),
                    "before_kst": before["datetime_kst"], "after_kst": after["datetime_kst"],
                    "cape_before_jkg": before_cape, "cape_after_jkg": after_cape, "cape_delta_jkg": delta_value(after, before, "cape_jkg"),
                    "cin_before_jkg": before["cin_jkg"], "cin_after_jkg": after["cin_jkg"], "cin_delta_jkg": delta_value(after, before, "cin_jkg"),
                    "pw_before_mm": before["precipitable_water_mm"], "pw_after_mm": after["precipitable_water_mm"], "pw_delta_mm": delta_value(after, before, "precipitable_water_mm"),
                    "lcl_before_hpa": before["lcl_pressure_hpa"], "lcl_after_hpa": after["lcl_pressure_hpa"], "lcl_delta_hpa": delta_value(after, before, "lcl_pressure_hpa"),
                    "lfc_before_hpa": before["lfc_pressure_hpa"], "lfc_after_hpa": after["lfc_pressure_hpa"],
                    "el_before_hpa": before["el_pressure_hpa"], "el_after_hpa": after["el_pressure_hpa"],
                    "lapse_before_c_km": before["lapse_rate_850_500_c_km"], "lapse_after_c_km": after["lapse_rate_850_500_c_km"],
                    "temperature_before_c": before["lowest_temperature_c"], "temperature_after_c": after["lowest_temperature_c"],
                    "dewpoint_before_c": before["lowest_dewpoint_c"], "dewpoint_after_c": after["lowest_dewpoint_c"],
                    "depression_before_c": before["lowest_depression_c"], "depression_after_c": after["lowest_depression_c"],
                    "cape_reduction_pct": reduction,
                }
                results.append(record)
                pairs[str(event["event_name"])] = (before, after)
            except Exception as exc:
                show_error(f"{event.get('event_name', '강수 사례')} 비교 중 오류가 발생했습니다.", exc)

    if results:
        comparisons = pd.DataFrame(results)
        st.dataframe(comparisons, width="stretch", hide_index=True)
        selected_event = st.selectbox("상세 비교 사례", list(pairs))
        before, after = pairs[selected_event]
        selected = comparisons[comparisons["event_name"].astype(str) == selected_event].iloc[0]
        metric_grid([
            ("Δ CAPE", format_number(selected["cape_delta_jkg"], " J/kg")),
            ("Δ CIN", format_number(selected["cin_delta_jkg"], " J/kg")),
            ("Δ 가강수량", format_number(selected["pw_delta_mm"], " mm")),
            ("Δ LCL", format_number(selected["lcl_delta_hpa"], " hPa")),
            ("CAPE 감소율", format_number(selected["cape_reduction_pct"], "%")),
        ], columns_per_row=5)
        interpretations: list[str] = []
        if np.isfinite(selected["cape_delta_jkg"]):
            interpretations.append("불안정 에너지가 감소한 패턴입니다." if selected["cape_delta_jkg"] < 0 else "강수 이후 CAPE가 증가한 패턴입니다.")
        if np.isfinite(selected["cin_before_jkg"]) and np.isfinite(selected["cin_after_jkg"]):
            interpretations.append("대류 억제가 강화된 패턴입니다." if abs(selected["cin_after_jkg"]) > abs(selected["cin_before_jkg"]) else "강수 이후 CIN의 절댓값이 감소한 패턴입니다.")
        if np.isfinite(selected["pw_delta_mm"]):
            interpretations.append("대기 기둥의 수증기량이 줄어든 패턴입니다." if selected["pw_delta_mm"] < 0 else "강수 이후 대기 기둥의 수증기량이 증가한 패턴입니다.")
        st.info(" ".join(interpretations) + " 강수가 직접 원인이라고 단정할 수 없으며 기단 이동, 일사 변화, 전선 통과 등의 영향도 가능합니다.")
        before_time, after_time = pd.Timestamp(before["datetime_utc"]), pd.Timestamp(after["datetime_utc"])
        before_frame, after_frame = period_soundings.get(before_time), period_soundings.get(after_time)
        if before_frame is not None and after_frame is not None:
            try:
                columns = st.columns(2)
                columns[0].pyplot(make_skewt_figure(before_frame, f"강수 전 · {format_timestamp(before_time, KST)}", min_pressure), width="stretch")
                columns[1].pyplot(make_skewt_figure(after_frame, f"강수 후 · {format_timestamp(after_time, KST)}", min_pressure), width="stretch")
                st.pyplot(comparison_overlay_figure(before_frame, after_frame, min_pressure), width="stretch")
            except Exception as exc:
                show_error("강수 전후 선도를 그릴 수 없습니다.", exc)
        st.download_button("강수 전후 비교 CSV 다운로드", comparisons.to_csv(index=False).encode("utf-8-sig"), file_name=f"{station['station_id']}_rain_comparison.csv", mime="text/csv")

with tabs[5]:
    st.subheader("자료 및 계산 방법")
    st.markdown(
        """
        **두 분석 영역의 관계**
        FITS 탭은 천문 이미지 처리와 서울 기준 천체 위치 계산이라는 기본 앱 기능입니다. NOAA 탭은 IGRA 라디오존데 자료를 이용한 기상학적 심화탐구입니다. 두 자료를 직접적인 인과관계가 있는 것으로 해석하지 않습니다.

        **FITS 처리**
        Astropy로 모든 HDU를 검사하고 첫 유효 이미지 HDU를 기본 선택합니다. 3차원 이상 배열은 앞쪽 축을 평탄화해 사용자가 고른 2차원 평면을 분석합니다. 유효 픽셀의 1백분위수와 99.5백분위수를 기본 명암 범위로 사용합니다. 헤더의 RA/DEC는 ICRS 좌표로 해석하고 서울(37.5665°N, 126.9780°E, 50 m)의 현재 시각 및 관측 시각 AltAz 좌표로 변환합니다.

        **라디오존데와 안정도**
        NOAA NCEI IGRA 2.2 최근자료 ZIP을 메모리에서 해제하고 공식 고정폭 형식대로 읽습니다. CAPE는 양의 부력 에너지, CIN은 대류 억제 에너지이며 LCL/LFC/EL과 가강수량을 함께 살펴야 합니다.

        **기본 관측 변수**
        기압은 위로 갈수록 작아지는 대기의 수직 좌표입니다. 기온은 공기의 온도, 이슬점은 포화가 시작되는 온도, 상대습도는 현재 수증기량과 포화 수증기량의 비입니다. IGRA 이슬점차를 °C로 변환한 뒤 `이슬점 = 기온 - 이슬점차`로 계산합니다.

        **과학적 한계**
        라디오존데 관측소와 실제 강수 지점 사이에는 공간적 차이가 있고 관측은 주로 약 12시간 간격이어서 짧은 강수 과정을 모두 포착하지 못합니다. CAPE 변화만으로 강수 원인을 확정할 수 없으며 기단 이동, 일사, 전선과 강제상승도 영향을 줍니다.
        """
    )
    st.markdown(
        f"- [FITS 샘플 공유 폴더]({SAMPLE_FITS_URL})\n"
        f"- [NOAA IGRA 관측소 목록]({STATION_LIST_URL})\n"
        f"- [NOAA IGRA 최근자료 디렉터리]({RECENT_DATA_DIRECTORY_URL})\n"
        f"- [IGRA 2.2 원시자료 형식]({DATA_FORMAT_URL})"
    )

st.divider()
st.subheader("댓글 및 피드백")
st.caption("댓글은 데이터베이스에 영구 저장되지 않으며 앱 재시작이나 브라우저 세션 종료 후 사라질 수 있습니다.")
if "comments" not in st.session_state:
    st.session_state.comments = []
with st.form("comment_form", clear_on_submit=True):
    commenter = st.text_input("이름")
    comment_text = st.text_area("댓글")
    submitted = st.form_submit_button("댓글 남기기")
    if submitted:
        if commenter.strip() and comment_text.strip():
            st.session_state.comments.append({"name": commenter.strip(), "comment": comment_text.strip(), "created_at": datetime.now().strftime("%Y-%m-%d %H:%M KST")})
            st.success("댓글을 현재 세션에 저장했습니다.")
        else:
            st.warning("이름과 댓글을 모두 입력해 주세요.")
for comment in reversed(st.session_state.comments):
    with st.container(border=True):
        st.markdown(f"**{comment['name']}** · {comment['created_at']}")
        st.write(comment["comment"])

st.markdown(
    f"""<div class="source-box"><strong>자료 출처</strong><br>
    천문 이미지: 사용자가 현재 세션에 업로드한 FITS 파일<br>
    기상 자료: NOAA NCEI Integrated Global Radiosonde Archive Version 2.2<br>
    <a href="{source_url or RECENT_DATA_DIRECTORY_URL}">선택 관측소 최근자료</a> ·
    <a href="{DATA_FORMAT_URL}">IGRA 자료 형식</a></div>""",
    unsafe_allow_html=True,
)
