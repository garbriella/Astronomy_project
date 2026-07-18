"""저장소 파일을 만들지 않는 FITS 처리 및 좌표 계산 테스트."""

from io import BytesIO

from astropy.io import fits
from astropy.time import Time
import numpy as np
import pytest

from fits_utils import (
    altitude_azimuth,
    extract_header_metadata,
    extract_image_plane,
    find_image_hdus,
    image_statistics,
    parse_observation_time,
    parse_skycoord,
    scale_image,
)


def in_memory_hdulist() -> fits.HDUList:
    """3차원 기본 HDU와 2차원 확장 HDU를 메모리에 생성한다."""

    primary = fits.PrimaryHDU(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    primary.header["OBJECT"] = "TEST TARGET"
    primary.header["EXPTIME"] = 30.0
    primary.header["RA"] = "12:30:00"
    primary.header["DEC"] = "+10:00:00"
    primary.header["DATE-OBS"] = "2026-01-01T12:00:00"
    extension = fits.ImageHDU(np.ones((5, 6), dtype=np.int16), name="SCI")
    payload = BytesIO()
    fits.HDUList([primary, extension]).writeto(payload)
    payload.seek(0)
    return fits.open(payload, memmap=False)


def test_hdu_discovery_and_nd_slice() -> None:
    with in_memory_hdulist() as hdulist:
        image_hdus = find_image_hdus(hdulist)
        assert [item.index for item in image_hdus] == [0, 1]
        first = extract_image_plane(hdulist[0].data, 0)
        second = extract_image_plane(hdulist[0].data, 1)
        assert first.shape == (3, 4)
        assert first[0, 0] == 0
        assert second[0, 0] == 12


def test_statistics_and_safe_contrast_transform() -> None:
    image = np.array([[1.0, 1.0], [np.nan, 1.0]])
    stats = image_statistics(image)
    assert stats["mean"] == 1.0
    assert stats["valid_pixels"] == 3
    assert stats["missing_ratio_pct"] == 25.0
    scaled = scale_image(image, 1.0, 1.0, "로그")
    assert np.all(scaled[np.isfinite(scaled)] == 0)
    assert np.isnan(scaled[1, 0])


def test_missing_header_values_are_safe() -> None:
    metadata = extract_header_metadata(fits.Header())
    assert all(value is None for value in metadata.values())
    assert parse_observation_time(None) is None
    assert parse_observation_time("not-a-time") is None


def test_coordinate_formats_and_altaz() -> None:
    sexagesimal = parse_skycoord("12:30:00", "+10:00:00")
    decimal = parse_skycoord(187.5, 10.0)
    assert sexagesimal.separation(decimal).arcsecond < 1e-6
    altitude, azimuth = altitude_azimuth(decimal, Time("2026-01-01T12:00:00"))
    assert -90 <= altitude <= 90
    assert 0 <= azimuth <= 360


def test_invalid_coordinate_requires_manual_correction() -> None:
    with pytest.raises(ValueError, match="해석하지 못했습니다"):
        parse_skycoord("invalid", "also-invalid")
