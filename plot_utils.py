"""FITS 이미지와 NOAA 고층대기 자료의 Plotly/Matplotlib 그래프 도구."""

from __future__ import annotations

from typing import Any

import koreanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
from metpy.calc import parcel_profile, wind_components
from metpy.plots import SkewT
from metpy.units import units
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fits_utils import scale_image
from weather_utils import thermodynamic_profile


def _downsample_2d(array: np.ndarray, max_side: int = 1000) -> np.ndarray:
    """대형 이미지를 브라우저 친화적인 크기로 균일 간격 축소한다."""

    rows, columns = array.shape
    stride = max(1, int(np.ceil(max(rows, columns) / max_side)))
    return array[::stride, ::stride]


def fits_image_figure(
    image: np.ndarray,
    lower: float,
    upper: float,
    scale: str,
    colorscale: str,
    flip_vertical: bool,
) -> go.Figure:
    """확대·이동 가능한 FITS 명암 이미지 Plotly 그래프를 만든다."""

    original = np.flipud(image) if flip_vertical else image
    displayed = scale_image(original, lower, upper, scale)
    displayed = _downsample_2d(displayed)
    original = _downsample_2d(original)
    figure = go.Figure(
        go.Heatmap(
            z=displayed,
            customdata=original,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            colorbar={"title": "변환 밝기"},
            hovertemplate="열 %{x}<br>행 %{y}<br>원본 밝기 %{customdata:.5g}<extra></extra>",
        )
    )
    figure.update_layout(
        title="FITS 이미지",
        xaxis_title="가로 픽셀",
        yaxis_title="세로 픽셀",
        template="plotly_white",
        height=650,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def fits_diagnostic_figure(image: np.ndarray) -> go.Figure:
    """밝기 히스토그램과 중앙 가로·세로 프로파일을 한 그림에 표시한다."""

    array = np.asarray(image, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        raise ValueError("유효 픽셀이 없어 밝기 진단 그래프를 만들 수 없습니다.")
    if finite.size > 300_000:
        finite = finite[:: int(np.ceil(finite.size / 300_000))]
    center_row = array[array.shape[0] // 2, :]
    center_column = array[:, array.shape[1] // 2]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("밝기 히스토그램", "중앙 가로 밝기", "중앙 세로 밝기"),
    )
    figure.add_trace(go.Histogram(x=finite, nbinsx=100, marker_color="#168aad", name="픽셀"), row=1, col=1)
    figure.add_trace(go.Scatter(x=np.arange(center_row.size), y=center_row, mode="lines", line={"color": "#0f8b8d"}, name="가로"), row=1, col=2)
    figure.add_trace(go.Scatter(x=np.arange(center_column.size), y=center_column, mode="lines", line={"color": "#e45756"}, name="세로"), row=1, col=3)
    figure.update_xaxes(title_text="밝기", row=1, col=1)
    figure.update_yaxes(title_text="픽셀 수", row=1, col=1)
    figure.update_xaxes(title_text="가로 픽셀", row=1, col=2)
    figure.update_yaxes(title_text="밝기", row=1, col=2)
    figure.update_xaxes(title_text="세로 픽셀", row=1, col=3)
    figure.update_yaxes(title_text="밝기", row=1, col=3)
    figure.update_layout(template="plotly_white", height=380, showlegend=False, margin={"l": 35, "r": 15, "t": 60, "b": 35})
    return figure


def vertical_profile_figures(frame: pd.DataFrame, use_height: bool) -> tuple[go.Figure, go.Figure, go.Figure]:
    """기온·습도·바람의 Plotly 연직 프로파일 세 개를 만든다."""

    valid = frame[frame["pressure_hpa"].notna()].copy()
    y_column = "geopotential_height_m" if use_height else "pressure_hpa"
    y_title = "지위고도 (m)" if use_height else "기압 (hPa)"
    valid = valid.dropna(subset=[y_column])
    custom_columns = ["pressure_hpa", "geopotential_height_m", "temperature_c", "dewpoint_c", "relative_humidity_pct", "wind_direction_deg", "wind_speed_ms"]
    custom = valid[custom_columns].to_numpy()
    hover = ("기압 %{customdata[0]:.1f} hPa<br>고도 %{customdata[1]:.0f} m<br>기온 %{customdata[2]:.1f} °C<br>이슬점 %{customdata[3]:.1f} °C<br>상대습도 %{customdata[4]:.1f}%<br>풍향 %{customdata[5]:.0f}°<br>풍속 %{customdata[6]:.1f} m/s<extra></extra>")
    temperature_figure = go.Figure()
    temperature_figure.add_trace(go.Scatter(x=valid["temperature_c"], y=valid[y_column], customdata=custom, hovertemplate=hover, mode="lines+markers", name="기온", line={"color": "#e45756", "width": 3}))
    temperature_figure.add_trace(go.Scatter(x=valid["dewpoint_c"], y=valid[y_column], customdata=custom, hovertemplate=hover, mode="lines+markers", name="이슬점", line={"color": "#159d9c", "width": 3}))
    temperature_figure.update_layout(title="기온과 이슬점의 수직 분포", xaxis_title="기온 (°C)", yaxis_title=y_title)
    humidity_figure = go.Figure(go.Scatter(x=valid["relative_humidity_pct"], y=valid[y_column], customdata=custom, hovertemplate=hover, mode="lines+markers", line={"color": "#2563eb", "width": 3}, name="상대습도"))
    humidity_figure.update_layout(title="상대습도의 수직 분포", xaxis_title="상대습도 (%)", yaxis_title=y_title)
    wind_figure = go.Figure(go.Scatter(x=valid["wind_speed_ms"], y=valid[y_column], customdata=custom, hovertemplate=hover, mode="markers+lines", marker={"size": 9, "color": valid["wind_direction_deg"], "colorscale": "Twilight", "colorbar": {"title": "풍향 (°)"}}, line={"color": "#8fa8b8"}, name="바람"))
    wind_figure.update_layout(title="풍속과 풍향의 수직 분포", xaxis_title="풍속 (m/s)", yaxis_title=y_title)
    for figure in (temperature_figure, humidity_figure, wind_figure):
        figure.update_layout(template="plotly_white", margin={"l": 30, "r": 20, "t": 60, "b": 30}, legend={"orientation": "h"})
        if not use_height:
            figure.update_yaxes(autorange="reversed")
    return temperature_figure, humidity_figure, wind_figure


def time_series_figure(data: pd.DataFrame, x_column: str, y_column: str, title: str, unit: str) -> go.Figure:
    """한 기상 지표의 관측 시각별 Plotly 선 그래프를 만든다."""

    figure = go.Figure(go.Scatter(x=data[x_column], y=data[y_column], mode="lines+markers", line={"color": "#168aad", "width": 2.5}, marker={"size": 7}, connectgaps=False))
    figure.update_layout(template="plotly_white", title=title, xaxis_title="관측 시각", yaxis_title=unit, margin={"l": 25, "r": 15, "t": 55, "b": 30})
    return figure


def make_skewt_figure(frame: pd.DataFrame, title: str = "Skew-T 대기선도", min_pressure_hpa: float = 100.0) -> plt.Figure:
    """관측 프로파일, 상승경로, 보조선과 바람깃을 포함한 Skew-T를 그린다."""

    profile = thermodynamic_profile(frame, min_pressure_hpa)
    if len(profile) < 2:
        raise ValueError("Skew-T를 그릴 기온·이슬점 자료가 부족합니다.")
    pressure = profile["pressure_hpa"].to_numpy() * units.hPa
    temperature = profile["temperature_c"].to_numpy() * units.degC
    dewpoint = profile["dewpoint_c"].to_numpy() * units.degC
    figure = plt.figure(figsize=(8.4, 8.4), constrained_layout=True)
    skew = SkewT(figure, rotation=45)
    skew.plot(pressure, temperature, color="#e45756", linewidth=2.2, label="기온")
    skew.plot(pressure, dewpoint, color="#159d9c", linewidth=2.2, label="이슬점")
    try:
        parcel = parcel_profile(pressure, temperature[0], dewpoint[0]).to("degC")
        skew.plot(pressure, parcel, color="#253b80", linewidth=1.8, label="표면 상승경로")
        try:
            skew.shade_cape(pressure, temperature, parcel, alpha=0.2)
            skew.shade_cin(pressure, temperature, parcel, dewpoint, alpha=0.16)
        except Exception:
            pass
    except Exception:
        pass
    wind = profile.dropna(subset=["wind_direction_deg", "wind_speed_ms"])
    if not wind.empty:
        wind = wind.iloc[:: max(1, len(wind) // 24)]
        speed = wind["wind_speed_ms"].to_numpy() * units("m/s")
        direction = wind["wind_direction_deg"].to_numpy() * units.degree
        u, v = wind_components(speed, direction)
        skew.plot_barbs(wind["pressure_hpa"].to_numpy() * units.hPa, u, v, xloc=1.0)
    skew.plot_dry_adiabats(alpha=0.22, color="#64748b")
    skew.plot_moist_adiabats(alpha=0.22, color="#2563eb")
    skew.plot_mixing_lines(alpha=0.18, color="#0f766e")
    skew.ax.set_ylim(max(profile["pressure_hpa"].max() + 20, 1000), min_pressure_hpa)
    skew.ax.set_xlim(-50, 45)
    skew.ax.set_xlabel("기온 (°C)")
    skew.ax.set_ylabel("기압 (hPa)")
    skew.ax.set_title(title, color="#132a4f", fontsize=14, pad=14)
    skew.ax.grid(alpha=0.16)
    skew.ax.legend(loc="best")
    return figure


def comparison_overlay_figure(before: pd.DataFrame, after: pd.DataFrame, min_pressure_hpa: float = 100.0) -> plt.Figure:
    """강수 전후 기온·이슬점 프로파일을 한 좌표계에 겹쳐 그린다."""

    before_profile = thermodynamic_profile(before, min_pressure_hpa)
    after_profile = thermodynamic_profile(after, min_pressure_hpa)
    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for profile, suffix, color in ((before_profile, "전", "#2563eb"), (after_profile, "후", "#e45756")):
        axis.plot(profile["temperature_c"], profile["pressure_hpa"], color=color, label=f"기온 ({suffix})")
        axis.plot(profile["dewpoint_c"], profile["pressure_hpa"], color=color, linestyle="--", label=f"이슬점 ({suffix})")
    axis.set_yscale("log")
    axis.set_ylim(max(before_profile["pressure_hpa"].max(), after_profile["pressure_hpa"].max()), min_pressure_hpa)
    axis.set_xlabel("기온 (°C)")
    axis.set_ylabel("기압 (hPa)")
    axis.set_title("강수 전후 기온·이슬점 수직 분포")
    axis.grid(alpha=0.2)
    axis.legend()
    return figure
