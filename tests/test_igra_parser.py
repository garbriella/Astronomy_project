"""IGRA 2.2 고정폭 파서 회귀 테스트."""

from io import BytesIO
import zipfile

import numpy as np

from weather_utils import (
    discover_recent_filename,
    extract_igra_text,
    parse_igra_soundings,
    thermodynamic_profile,
)


def _put(line: list[str], start: int, end: int, value: object) -> None:
    """0-based end-exclusive 고정폭 필드에 우측 정렬 값을 넣는다."""

    width = end - start
    line[start:end] = list(str(value).rjust(width)[:width])


def _header(num_levels: int) -> str:
    line = [" "] * 71
    line[0] = "#"
    line[1:12] = list("KSM00047138")
    for start, end, value in (
        (13, 17, 2026),
        (18, 20, 7),
        (21, 23, 18),
        (24, 26, 0),
        (27, 31, 2330),
        (32, 36, num_levels),
        (55, 62, 360320),
        (63, 71, 1293800),
    ):
        _put(line, start, end, value)
    return "".join(line)


def _level(
    pressure_pa: int,
    height_m: int,
    temperature_tenths_c: int,
    rh_tenths_pct: int,
    depression_tenths_c: int,
    direction_deg: int = 180,
    speed_tenths_ms: int = 50,
) -> str:
    line = [" "] * 51
    line[0:2] = list("10")
    for start, end, value in (
        (2, 8, 0),
        (9, 15, pressure_pa),
        (16, 21, height_m),
        (22, 27, temperature_tenths_c),
        (28, 33, rh_tenths_pct),
        (34, 39, depression_tenths_c),
        (40, 45, direction_deg),
        (46, 51, speed_tenths_ms),
    ):
        _put(line, start, end, value)
    line[15] = "B"
    line[21] = "A"
    line[27] = "B"
    return "".join(line)


def sample_text() -> str:
    """정렬·중복·단위·비정상 이슬점 검사용 작은 IGRA fixture."""

    levels = [
        _level(85000, 1500, 120, 600, 30),
        _level(100000, 100, 200, 700, 20),
        _level(85000, 1510, 119, 610, 31),  # 중복 기압
        _level(70000, 3000, 20, 500, -10),  # 음의 이슬점차
        _level(50000, 5600, -150, -9999, 50),
    ]
    return "\n".join([_header(len(levels)), *levels])


def test_parser_applies_units_sorting_and_duplicate_removal() -> None:
    soundings = parse_igra_soundings(sample_text())
    assert len(soundings) == 1
    frame = next(iter(soundings.values()))
    assert frame["pressure_hpa"].tolist() == [1000.0, 850.0, 700.0, 500.0]
    assert frame.iloc[0]["temperature_c"] == 20.0
    assert frame.iloc[0]["dewpoint_depression_c"] == 2.0
    assert frame.iloc[0]["dewpoint_c"] == 18.0
    assert frame.iloc[0]["wind_speed_ms"] == 5.0
    assert frame.iloc[0]["pressure_qc_flag"] == "B"


def test_invalid_dewpoint_and_missing_humidity_become_nan() -> None:
    frame = next(iter(parse_igra_soundings(sample_text()).values()))
    row_700 = frame.loc[frame["pressure_hpa"] == 700].iloc[0]
    row_500 = frame.loc[frame["pressure_hpa"] == 500].iloc[0]
    assert np.isnan(row_700["dewpoint_c"])
    assert np.isnan(row_500["relative_humidity_pct"])
    profile = thermodynamic_profile(frame)
    assert 700.0 not in profile["pressure_hpa"].tolist()


def test_filename_discovery_does_not_fix_start_year() -> None:
    html = '<a href="KSM00047138-data-beg2031.txt.zip">file</a>'
    assert discover_recent_filename(html, "KSM00047138") == "KSM00047138-data-beg2031.txt.zip"


def test_zip_is_extracted_in_memory() -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("KSM00047138-data.txt", sample_text())
    assert extract_igra_text(payload.getvalue()).startswith("#KSM00047138")
