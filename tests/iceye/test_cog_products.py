# SPDX-License-Identifier: MIT

"""ICEYE SLC-COG and GRD-COG reader unittests"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from perseo_core.geometry.geocoding import direct_geocoding_monostatic
from perseo_core.timing import PreciseDateTime
from scipy.constants import speed_of_light

from eo_products.common.utilities import SARProjection, SARRadiometricQuantity
from eo_products.iceye.raster import read_shadows_down_raster
from eo_products.iceye.reader import open_product, read_channel_data, read_channel_metadata
from eo_products.iceye.utilities import (
    ICEYEProductFormat,
    ICEYEProductLevel,
    InvalidICEYEProduct,
    is_iceye_product,
)

from .synthetic_products import (
    AMPLITUDE_SCALE,
    CALIBRATION_FACTOR,
    CARRIER_FREQUENCY,
    GRD_AMPLITUDE_SCALE,
    GRD_PIXEL_SPACING,
    GROUND_TO_SLANT_COEFF,
    INCIDENCE_COEFF,
    PHASE_OFFSET,
    PHASE_SCALE,
    RANGE_NEAR,
    RANGE_SAMPLING_RATE,
    raster_values,
    write_cog_product,
    write_legacy_slc_from_cog,
)

LOOKS = ["right", "left"]


def _channel(folder: Path):
    product = open_product(folder)
    name = product.channels_list[0]
    files = product.get_files_from_channel_name(name)
    metadata_file = next(f for f in files if f.suffix in (".json", ".h5"))
    raster_file = next(f for f in files if f.suffix in (".tif", ".h5"))
    return name, files, read_channel_metadata(metadata_file, name), raster_file


def _expected_slc(look: str, lines: int = 96, samples: int = 64) -> np.ndarray:
    """Expected decoded SLC samples in (lines, samples) time order, computed from the raw band values"""
    raw = raster_values("SLC", samples, lines)
    columns = np.arange(lines) if look == "right" else lines - 1 - np.arange(lines)
    ordered = raw[:, columns, :].transpose(1, 0, 2)
    amplitude = ordered[..., 0].astype(np.float32) * AMPLITUDE_SCALE
    phase = ordered[..., 1].astype(np.float32) * PHASE_SCALE + PHASE_OFFSET
    return amplitude * np.cos(phase) + 1j * amplitude * np.sin(phase)


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize("level", ["SLC", "GRD"])
def test_cog_product_detection(tmp_path, level):
    json_path, tif_path = write_cog_product(tmp_path, level=level)
    assert is_iceye_product(tmp_path)
    product = open_product(tmp_path)
    assert product.channels_list == [f"{level.lower()}_spotlight_vv"]
    assert product.get_files_from_channel_name(product.channels_list[0]) == [tif_path, json_path]
    assert product.footprint == (45.0, 45.1, 12.0, 12.1)
    assert product.acquisition_time == PreciseDateTime.fromisoformat("2026-01-01T11:59:58.995200")


def test_slc_and_grd_cog_in_same_folder(tmp_path):
    write_cog_product(tmp_path, level="SLC")
    write_cog_product(tmp_path, level="GRD")
    assert sorted(open_product(tmp_path).channels_list) == ["grd_spotlight_vv", "slc_spotlight_vv"]


def test_cog_product_type_without_cog_suffix(tmp_path):
    # older COG deliveries annotate sar:product_type "SLC"/"GRD"
    write_cog_product(tmp_path, level="GRD", product_type="GRD")
    assert open_product(tmp_path).channels_list == ["grd_spotlight_vv"]


def test_not_a_cog_product(tmp_path):
    json_path, tif_path = write_cog_product(tmp_path / "no_json")
    json_path.unlink()
    assert not is_iceye_product(tmp_path / "no_json")

    json_path, _ = write_cog_product(tmp_path / "other_type")
    item = json.loads(json_path.read_text())
    item["properties"]["sar:product_type"] = "CSI"
    json_path.write_text(json.dumps(item))
    assert not is_iceye_product(tmp_path / "other_type")

    json_path, tif_path = write_cog_product(tmp_path / "no_tif")
    tif_path.unlink()
    assert not is_iceye_product(tmp_path / "no_tif")


def test_legacy_and_cog_channels_in_same_folder(tmp_path):
    json_path, tif_path = write_cog_product(tmp_path)
    write_legacy_slc_from_cog(json_path, tif_path, tmp_path)
    product = open_product(tmp_path)
    assert sorted(product.channels_list) == ["slc_spotlight_vv", "slc_spotlight_vv_cog"]
    assert [f.suffix for f in product.get_files_from_channel_name("slc_spotlight_vv")] == [".h5"]
    assert product.get_files_from_channel_name("slc_spotlight_vv_cog") == [tif_path, json_path]


# ------------------------------------------------------------------ metadata


@pytest.mark.parametrize("look", LOOKS)
def test_slc_cog_metadata(tmp_path, look):
    json_path, _ = write_cog_product(tmp_path, look=look)
    properties = json.loads(json_path.read_text())["properties"]
    _, _, meta, _ = _channel(tmp_path)

    assert meta.product_format == ICEYEProductFormat.COG
    assert meta.general_info.product_level == ICEYEProductLevel.SLC
    assert meta.general_info.projection == SARProjection.SLANT_RANGE
    assert meta.dataset_info.side_looking == look.upper()
    assert meta.raster_layout.look_side == look.upper()
    assert (meta.raster_layout.azimuth_lines, meta.raster_layout.range_samples) == (96, 64)

    # azimuth timing from zero Doppler start/end, not from the processing PRF
    zd_start = PreciseDateTime.fromisoformat(properties["iceye:zero_doppler_start_datetime"])
    zd_end = PreciseDateTime.fromisoformat(properties["iceye:zero_doppler_end_datetime"])
    assert meta.raster_info.lines.start == zd_start
    assert meta.raster_info.lines.step == pytest.approx(100e-6, rel=1e-12)
    assert meta.raster_info.lines.start + meta.raster_info.lines.step * 95 - zd_end == pytest.approx(0, abs=1e-12)
    assert meta.sampling_constants.azimuth_freq_hz == pytest.approx(1e4, rel=1e-12)

    # range timing from near range and sampling rate
    assert meta.raster_info.samples.start == pytest.approx(2 * RANGE_NEAR / speed_of_light, rel=1e-15)
    assert meta.raster_info.samples.step == pytest.approx(1 / RANGE_SAMPLING_RATE, rel=1e-15)

    assert meta.image_calibration_factor == pytest.approx(np.sqrt(CALIBRATION_FACTOR))
    assert meta.image_radiometric_quantity == SARRadiometricQuantity.BETA_NOUGHT
    assert meta.coordinate_conversions.ground_to_slant is None
    assert meta.state_vectors.num == len(properties["iceye:orbit_states"])

    # spotlight: Doppler centroid averaged to a single polynomial, as for legacy products
    assert len(meta.doppler_centroid_poly.functions) == 1
    mid_range = meta.raster_info.samples.start + meta.raster_info.samples.step * 64 / 2
    assert meta.doppler_centroid_poly.evaluate(zd_start, mid_range) == pytest.approx(20.0)

    # incidence polynomial in range pixel index
    assert meta.incidence_angles_poly.evaluate_incidence_angle(zd_start, np.array([0, 63])) == pytest.approx(
        np.polynomial.Polynomial(INCIDENCE_COEFF)(np.array([0, 63]))
    )


def test_slc_cog_resampled_range_uses_pixel_spacing(tmp_path):
    write_cog_product(tmp_path, pixel_spacing_range=0.6)
    _, _, meta, _ = _channel(tmp_path)
    assert meta.raster_info.samples.step == pytest.approx(2 * 0.6 / speed_of_light, rel=1e-15)
    assert meta.sampling_constants.range_freq_hz == pytest.approx(speed_of_light / 1.2, rel=1e-15)


def test_stripmap_doppler_centroid_keeps_all_estimates(tmp_path):
    json_path, _ = write_cog_product(tmp_path, mode="stripmap")
    properties = json.loads(json_path.read_text())["properties"]
    _, _, meta, _ = _channel(tmp_path)
    times = [PreciseDateTime.fromisoformat(t) for t in properties["iceye:doppler_centroid_datetimes"]]
    assert len(meta.doppler_centroid_poly.functions) == len(times)
    mid_range = meta.raster_info.samples.start + meta.raster_info.samples.step * 64 / 2
    for k, time in enumerate(times):
        assert meta.doppler_centroid_poly.evaluate(time, mid_range + 1e-6) == pytest.approx(10.0 * k + 1e3 * 1e-6)


@pytest.mark.parametrize("look", LOOKS)
def test_doppler_rate_from_orbit(tmp_path, look):
    """Doppler rate polynomial vs -2/lambda * d2R/dt2 by finite differences on the orbit"""
    write_cog_product(tmp_path, look=look)
    _, _, meta, _ = _channel(tmp_path)
    trajectory = meta.orbit
    wavelength = speed_of_light / CARRIER_FREQUENCY
    mid_time = meta.raster_info.lines.start + meta.raster_info.lines.step * 95 / 2
    for sample in (0, 32, 63):
        range_time = meta.raster_info.samples.start + sample * meta.raster_info.samples.step
        ground = direct_geocoding_monostatic(
            sensor_positions=trajectory.position(mid_time),
            sensor_velocities=trajectory.velocity(mid_time),
            range_times=range_time,
            doppler_frequencies=0.0,
            wavelength=wavelength,
            look_direction=look.upper(),
            altitude=0.0,
        ).reshape(3)
        step = 0.05
        distance = [np.linalg.norm(trajectory.position(mid_time + k * step) - ground) for k in (-1, 0, 1)]
        expected = -2 / wavelength * (distance[0] - 2 * distance[1] + distance[2]) / step**2
        assert meta.doppler_rate_poly.evaluate(mid_time, range_time) == pytest.approx(expected, rel=1e-3)


def test_grd_cog_metadata(tmp_path):
    json_path, _ = write_cog_product(tmp_path, level="GRD", look="left")
    _, _, meta, _ = _channel(tmp_path)
    zd_start = meta.raster_info.lines.start

    assert meta.general_info.projection == SARProjection.GROUND_RANGE
    assert meta.image_radiometric_quantity == SARRadiometricQuantity.BETA_NOUGHT  # unlike legacy GRD (sigma nought)
    assert meta.raster_info.samples.start == 0
    assert meta.raster_info.samples.step == GRD_PIXEL_SPACING
    assert meta.raster_info.samples.step_unit == "m"

    # ground range [m] from first pixel -> two-way slant range time [s]
    ground = np.array([0.0, 10.0, 31.5])
    slant = np.polynomial.Polynomial(GROUND_TO_SLANT_COEFF)(ground)
    times = meta.coordinate_conversions.evaluate_ground_to_slant(zd_start, ground)
    assert times == pytest.approx(2 * slant / speed_of_light, rel=1e-15)
    assert meta.coordinate_conversions.evaluate_slant_to_ground(zd_start, times) == pytest.approx(ground, abs=1e-3)


# ------------------------------------------------------------------ raster


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("look", LOOKS)
def test_read_slc_cog_orientation(tmp_path, look, planar):
    write_cog_product(tmp_path, look=look, planar=planar)
    _, _, meta, raster = _channel(tmp_path)
    expected = _expected_slc(look)

    full = read_shadows_down_raster(raster, meta.raster_layout)
    assert full.dtype == np.complex64
    np.testing.assert_allclose(full, expected, rtol=1e-6)

    for block in ([0, 0, 10, 7], [86, 57, 10, 7], [40, 0, 1, 64], [0, 63, 96, 1], [17, 23, 31, 29]):
        data = read_shadows_down_raster(raster, meta.raster_layout, block)
        np.testing.assert_array_equal(data, full[block[0] : block[0] + block[2], block[1] : block[1] + block[3]])


def test_read_channel_data_cog_applies_scaling(tmp_path):
    write_cog_product(tmp_path, look="left")
    _, _, meta, raster = _channel(tmp_path)
    block = [5, 6, 20, 30]
    scaled = read_channel_data(raster, block, meta.image_calibration_factor, raster_layout=meta.raster_layout)
    np.testing.assert_array_equal(
        scaled, read_shadows_down_raster(raster, meta.raster_layout, block) * meta.image_calibration_factor
    )


@pytest.mark.parametrize("grd_dtype", ["uint16", "float32"])
@pytest.mark.parametrize("look", LOOKS)
def test_read_grd_cog(tmp_path, look, grd_dtype):
    write_cog_product(tmp_path, level="GRD", look=look, grd_dtype=grd_dtype)
    _, _, meta, raster = _channel(tmp_path)
    raw = raster_values("GRD", 64, 96)
    columns = np.arange(96) if look == "right" else 95 - np.arange(96)
    expected = raw[:, columns].T.astype(np.float32) * np.float32(GRD_AMPLITUDE_SCALE)
    np.testing.assert_allclose(read_shadows_down_raster(raster, meta.raster_layout), expected, rtol=1e-6)


def test_power_amplitude_mapping(tmp_path):
    write_cog_product(tmp_path, amplitude_mapping={"function": "power", "parameters": {"scale": 2.0, "exponent": 0.5}})
    _, _, meta, raster = _channel(tmp_path)
    stored = np.abs(_expected_slc("right"))  # stored (mapped) amplitude y
    np.testing.assert_allclose(np.abs(read_shadows_down_raster(raster, meta.raster_layout)), stored**2 / 2.0, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude_mapping": {"function": "arctan", "parameters": {"scale": 1}}},
        {"orientation": "north-up"},
        {"coordinate_frame": "eci"},
    ],
)
def test_unsupported_cog_products(tmp_path, kwargs):
    json_path, _ = write_cog_product(tmp_path, **kwargs)
    with pytest.raises(InvalidICEYEProduct):
        read_channel_metadata(json_path, "slc_spotlight_vv")


# ------------------------------------------------------------------ equivalence with legacy HDF5


@pytest.mark.parametrize("look", LOOKS)
def test_slc_cog_equivalent_to_legacy_hdf5(tmp_path, look):
    """The same scene stored as SLC-COG and as legacy HDF5 SLC gives the same metadata and the same samples"""
    json_path, tif_path = write_cog_product(tmp_path / "cog", look=look)
    write_legacy_slc_from_cog(json_path, tif_path, tmp_path / "legacy")
    cog_name, _, cog, cog_raster = _channel(tmp_path / "cog")
    h5_name, _, h5, h5_raster = _channel(tmp_path / "legacy")

    assert cog_name == h5_name
    assert cog.general_info.product_name == h5.general_info.product_name
    for field in ("product_level", "polarization", "projection", "acquisition_mode", "orbit_direction"):
        assert getattr(cog.general_info, field) == getattr(h5.general_info, field)
    assert cog.raster_info.lines.length == h5.raster_info.lines.length
    assert cog.raster_info.lines.start == h5.raster_info.lines.start
    assert cog.raster_info.lines.step == pytest.approx(h5.raster_info.lines.step, rel=1e-12)
    assert cog.raster_info.samples.length == h5.raster_info.samples.length
    assert cog.raster_info.samples.start == pytest.approx(h5.raster_info.samples.start, rel=1e-15)
    assert cog.raster_info.samples.step == pytest.approx(h5.raster_info.samples.step, rel=1e-15)
    assert cog.dataset_info == h5.dataset_info
    assert cog.swath_info == h5.swath_info
    np.testing.assert_array_equal(cog.state_vectors.positions, h5.state_vectors.positions)
    np.testing.assert_array_equal(cog.state_vectors.velocities, h5.state_vectors.velocities)
    assert all(a == b for a, b in zip(cog.state_vectors.time_axis, h5.state_vectors.time_axis, strict=True))
    for field in ("azimuth_freq_hz", "azimuth_bandwidth_freq_hz", "range_freq_hz", "range_bandwidth_freq_hz"):
        assert getattr(cog.sampling_constants, field) == pytest.approx(getattr(h5.sampling_constants, field), rel=1e-12)
    assert cog.image_calibration_factor == h5.image_calibration_factor
    assert cog.image_radiometric_quantity == h5.image_radiometric_quantity
    range_times = h5.raster_info.samples.start + np.arange(0, 64, 9) * h5.raster_info.samples.step
    assert cog.doppler_centroid_poly.evaluate(h5.raster_info.lines.start, range_times) == pytest.approx(
        h5.doppler_centroid_poly.evaluate(h5.raster_info.lines.start, range_times)
    )

    for block in (None, [0, 0, 96, 64], [0, 0, 1, 1], [95, 63, 1, 1], [3, 50, 40, 14], [60, 0, 36, 64]):
        np.testing.assert_array_equal(
            read_channel_data(cog_raster, block, cog.image_calibration_factor, raster_layout=cog.raster_layout),
            read_channel_data(h5_raster, block, h5.image_calibration_factor),
        )
