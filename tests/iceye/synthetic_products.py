# SPDX-License-Identifier: MIT

"""Synthetic ICEYE products for unit tests: SLC-COG/GRD-COG (COG + STAC .json) and legacy SLC (.h5).

Products are tiny and generated at test time from a circular orbit. Raster values encode their position so that
orientation errors are detectable. The legacy writer decodes the COG bands independently from eo_products, so that it
can be used as a reference for the COG reader.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import tifffile
from scipy.constants import speed_of_light

GM_EARTH = 3.986004418e14
OMEGA_EARTH = 7.2921150e-5
T_REF = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
RANGE_NEAR = 620_000.0
RANGE_SAMPLING_RATE = 300e6
AZIMUTH_INTERVAL_US = 100  # azimuth line interval in microseconds
CARRIER_FREQUENCY = 9.65e9
CALIBRATION_FACTOR = 0.05
AMPLITUDE_SCALE = 0.03125
PHASE_SCALE = 2 * np.pi / 65535
PHASE_OFFSET = -np.pi
GRD_AMPLITUDE_SCALE = 0.01
GRD_PIXEL_SPACING = 0.5
GROUND_TO_SLANT_COEFF = [RANGE_NEAR, 0.62, 2.0e-7, -1.0e-13, 0.0]
INCIDENCE_COEFF = [38.0, 1.5e-3, -2.0e-8, 0.0]


def iso(time: datetime, suffix: str = "Z") -> str:
    """ISO 8601 string with microseconds"""
    return time.strftime("%Y-%m-%dT%H:%M:%S.%f") + suffix


def parse_iso(value: str) -> datetime:
    """Parse ISO 8601 string with or without trailing Z"""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def circular_orbit(num: int = 61, step: float = 1.0) -> list[dict]:
    """Earth-fixed state vectors of a 500 km circular sun-synchronous-like orbit, centred on T_REF"""
    semi_major_axis = 6378137.0 + 500e3
    mean_motion = np.sqrt(GM_EARTH / semi_major_axis**3)
    inclination = np.deg2rad(97.4)
    states = []
    for k in range(num):
        t = (k - num // 2) * step
        u = 0.6 + mean_motion * t
        r = semi_major_axis * np.array([np.cos(u), np.sin(u) * np.cos(inclination), np.sin(u) * np.sin(inclination)])
        v = (
            semi_major_axis
            * mean_motion
            * np.array([-np.sin(u), np.cos(u) * np.cos(inclination), np.cos(u) * np.sin(inclination)])
        )
        theta = OMEGA_EARTH * t
        rot = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        states.append(
            {
                "time": iso(T_REF + timedelta(seconds=t)),
                "position": (rot @ r).tolist(),
                "velocity": (rot @ (v - np.cross([0, 0, OMEGA_EARTH], r))).tolist(),
            }
        )
    return states


def _gdal_metadata(scales: list[float], offsets: list[float], names: list[str]) -> str:
    items = []
    for sample, (scale, offset, name) in enumerate(zip(scales, offsets, names, strict=True)):
        items.append(f'  <Item name="NAME" sample="{sample}">{name}</Item>')
        items.append(f'  <Item name="OFFSET" sample="{sample}" role="offset">{offset!r}</Item>')
        items.append(f'  <Item name="SCALE" sample="{sample}" role="scale">{scale!r}</Item>')
    return "<GDALMetadata>\n" + "\n".join(items) + "\n</GDALMetadata>"


def raster_values(level: str, samples: int, lines: int) -> np.ndarray:
    """Raw COG band values with shape (range samples, azimuth columns[, bands]) encoding the pixel position"""
    rows = np.arange(samples)[:, None]
    cols = np.arange(lines)[None, :]
    amplitude = ((rows * 7 + cols * 131) % 60000 + 1).astype(np.uint16)
    if level == "GRD":
        return amplitude
    phase = ((rows * 977 + cols * 61) % 65535).astype(np.uint16)
    return np.stack([amplitude, phase], axis=-1)


def write_cog_product(  # noqa: PLR0913
    folder: Path,
    level: str = "SLC",
    look: str = "right",
    lines: int = 96,
    samples: int = 64,
    mode: str = "spotlight",
    product_type: str | None = None,
    amplitude_mapping: dict | None = None,
    orientation: str = "shadows-down",
    coordinate_frame: str = "ecef",
    planar: bool = False,
    overviews: bool = True,
    grd_dtype: str = "uint16",
    pixel_spacing_range: float | None = None,
) -> tuple[Path, Path]:
    """Write a synthetic ICEYE SLC-COG or GRD-COG product (COG .tif + STAC .json) in folder.

    Returns
    -------
    tuple[Path, Path]
        path to .json metadata file, path to .tif raster file
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"ICEYE_SYNTH_20260101T120000Z_1_X99_{mode.upper()[:3]}_{level}"
    tif_path, json_path = folder / f"{stem}.tif", folder / f"{stem}.json"

    data = raster_values(level, samples, lines)
    if level == "SLC":
        scales, offsets, names = [AMPLITUDE_SCALE, PHASE_SCALE], [0.0, PHASE_OFFSET], ["AMPLITUDE", "PHASE"]
    elif grd_dtype == "uint16":
        scales, offsets, names = [GRD_AMPLITUDE_SCALE], [0.0], ["AMPLITUDE"]
    else:
        data = data.astype(np.float32) * GRD_AMPLITUDE_SCALE
        scales, offsets, names = None, None, None
    extratags = [] if scales is None else [(42112, "s", 0, _gdal_metadata(scales, offsets, names), True)]
    if planar and data.ndim == 3:
        data_to_write, config = np.moveaxis(data, -1, 0), "separate"
    else:
        data_to_write, config = data, "contig"
    with tifffile.TiffWriter(tif_path) as tif:
        tif.write(
            data_to_write,
            tile=(16, 16),
            photometric="minisblack",
            planarconfig=config,
            extratags=extratags,
            metadata=None,
        )
        if overviews:
            reduced = data_to_write[..., ::2, ::2] if config == "separate" else data_to_write[::2, ::2]
            tif.write(
                reduced, tile=(16, 16), photometric="minisblack", planarconfig=config, subfiletype=1, metadata=None
            )

    zd_start = T_REF - timedelta(microseconds=AZIMUTH_INTERVAL_US * (lines // 2))
    zd_end = zd_start + timedelta(microseconds=AZIMUTH_INTERVAL_US * (lines - 1))
    dc_times = [zd_start + (zd_end - zd_start) * k / 4 for k in range(5)]
    if pixel_spacing_range is None:
        pixel_spacing_range = (
            round(speed_of_light / (2 * RANGE_SAMPLING_RATE), 6) if level == "SLC" else GRD_PIXEL_SPACING
        )
    properties = {
        "start_datetime": iso(zd_start - timedelta(seconds=1)),
        "end_datetime": iso(zd_end + timedelta(seconds=1)),
        "platform": "ICEYE-X99",
        "sat:orbit_state": "ascending",
        "sar:instrument_mode": mode,
        "sar:product_type": product_type or f"{level}-COG",
        "sar:polarizations": ["VV"],
        "sar:observation_direction": look,
        "sar:center_frequency": CARRIER_FREQUENCY,
        "sar:pixel_spacing_range": pixel_spacing_range,
        "sar:pixel_spacing_azimuth": 0.7,
        "iceye:filename": tif_path.name,
        "iceye:orientation": orientation,
        "iceye:zero_doppler_start_datetime": iso(zd_start),
        "iceye:zero_doppler_end_datetime": iso(zd_end),
        "iceye:range_near": RANGE_NEAR,
        "iceye:acquisition_range_sampling_rate": RANGE_SAMPLING_RATE,
        "iceye:acquisition_prf": 6000.0,
        "iceye:processing_prf": 1e6 / AZIMUTH_INTERVAL_US * 1.01,  # deliberately not the line rate
        "iceye:processing_bandwidth_azimuth": 8000.0,
        "iceye:processing_bandwidth_range": 3.0e8,
        "iceye:pulse_bandwidth": 3.0e8,
        "iceye:pulse_duration": 3.0e-5,
        "iceye:coordinate_frame": coordinate_frame,
        "iceye:orbit_precision": "precise",
        "iceye:orbit_states": circular_orbit(),
        "iceye:doppler_centroid_coeffs": [[10.0 * k, 1e3] for k in range(5)],
        "iceye:doppler_centroid_datetimes": [iso(t) for t in dc_times],
        "iceye:doppler_rate_coeffs": [0.01, -700.0, 2e5, -2e7],  # not an FM rate, must not be used as such
        "iceye:incidence_angle_coeffs": INCIDENCE_COEFF,
        "iceye:ground_to_slant_coeff": GROUND_TO_SLANT_COEFF,
        "iceye:average_scene_height": 0.0,
        "iceye:calibration_factor": CALIBRATION_FACTOR,
        "iceye:amplitude_mapping": amplitude_mapping
        or {"function": "minmax", "parameters": {"scale": 1, "exponent": 1}},
    }
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": stem,
        "bbox": [12.0, 45.0, 12.1, 45.1],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[12.0, 45.0], [12.1, 45.0], [12.1, 45.1], [12.0, 45.1], [12.0, 45.0]]],
        },
        "properties": properties,
        "assets": {
            "cog": {
                "href": tif_path.name,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"],
            },
            "metadata": {"href": json_path.name, "type": "application/geo+json", "roles": ["metadata"]},
        },
    }
    json_path.write_text(json.dumps(item))
    return json_path, tif_path


def write_legacy_slc_from_cog(json_path: Path, tif_path: Path, folder: Path) -> Path:
    """Write the legacy HDF5 SLC equivalent of a synthetic SLC-COG product (same scene, same samples).

    Decoding and reorientation are implemented here independently from eo_products: legacy line a (time order) is COG
    column a for right looking products and column (lines - 1 - a) for left looking ones.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    properties = json.loads(Path(json_path).read_text())["properties"]
    data = tifffile.imread(tif_path, level=0)
    if data.shape[0] == 2:  # planar
        data = np.moveaxis(data, 0, -1)
    samples, lines = data.shape[:2]
    columns = np.arange(lines)
    if properties["sar:observation_direction"] == "left":
        columns = lines - 1 - columns
    ordered = data[:, columns, :].transpose(1, 0, 2)  # (lines, samples, bands)
    amplitude = ordered[..., 0].astype(np.float32) * AMPLITUDE_SCALE + 0.0
    phase = ordered[..., 1].astype(np.float32) * PHASE_SCALE + PHASE_OFFSET

    zd_start = parse_iso(properties["iceye:zero_doppler_start_datetime"])
    zd_end = parse_iso(properties["iceye:zero_doppler_end_datetime"])
    states = properties["iceye:orbit_states"]
    dc = np.zeros((len(properties["iceye:doppler_centroid_coeffs"]), 4))
    for k, coeff in enumerate(properties["iceye:doppler_centroid_coeffs"]):
        dc[k, : len(coeff)] = coeff

    def text(value: str) -> np.bytes_:
        return np.bytes_(value)

    def legacy_time(value: str) -> str:
        return iso(parse_iso(value), suffix="")

    h5_path = folder / (Path(tif_path).stem + ".h5")
    with h5py.File(h5_path, "w") as h5:
        h5["product_file"] = text(h5_path.name)
        h5["product_name"] = text(Path(properties["iceye:filename"]).stem)
        h5["product_level"] = text("SLC")
        h5["satellite_name"] = text(properties["platform"])
        h5["acquisition_mode"] = text(properties["sar:instrument_mode"])
        h5["polarization"] = text(properties["sar:polarizations"][0])
        h5["look_side"] = text(properties["sar:observation_direction"])
        h5["orbit_direction"] = text(properties["sat:orbit_state"].upper())
        h5["orbit_processing_level"] = text(properties["iceye:orbit_precision"])
        h5["geo_ref_system"] = text("WGS84")
        h5["carrier_frequency"] = float(properties["sar:center_frequency"])
        h5["acquisition_prf"] = float(properties["iceye:acquisition_prf"])
        h5["chirp_bandwidth"] = float(properties["iceye:processing_bandwidth_range"])
        h5["chirp_duration"] = float(properties["iceye:pulse_duration"])
        h5["total_processed_bandwidth_azimuth"] = float(properties["iceye:processing_bandwidth_azimuth"])
        h5["range_sampling_rate"] = float(properties["iceye:acquisition_range_sampling_rate"])
        h5["acquisition_start_utc"] = text(legacy_time(properties["start_datetime"]))
        h5["acquisition_end_utc"] = text(legacy_time(properties["end_datetime"]))
        h5["first_pixel_time"] = 2 * float(properties["iceye:range_near"]) / speed_of_light
        h5["slant_range_to_first_pixel"] = float(properties["iceye:range_near"])
        h5["zerodoppler_start_utc"] = text(iso(zd_start, suffix=""))
        h5["zerodoppler_end_utc"] = text(iso(zd_end, suffix=""))
        h5["azimuth_time_interval"] = (zd_end - zd_start).total_seconds() / (lines - 1)
        h5["number_of_azimuth_samples"] = lines
        h5["number_of_range_samples"] = samples
        h5["number_of_state_vectors"] = len(states)
        h5["state_vector_time_utc"] = np.array([[text(legacy_time(s["time"]))] for s in states], dtype="S26")
        for axis, name in enumerate("XYZ"):
            h5[f"pos{name}"] = np.array([s["position"][axis] for s in states])
            h5[f"vel{name}"] = np.array([s["velocity"][axis] for s in states])
        h5["dc_estimate_coeffs"] = dc
        h5["dc_estimate_time_utc"] = np.array(
            [[text(legacy_time(t))] for t in properties["iceye:doppler_centroid_datetimes"]], dtype="S26"
        )
        h5["doppler_rate_coeffs"] = np.array([-3000.0, 0.0, 0.0, 0.0])
        h5["calibration_factor"] = float(properties["iceye:calibration_factor"])
        for name in ("coord_first_near", "coord_first_far", "coord_last_near", "coord_last_far"):
            h5[name] = np.array([0.0, 0.0, 45.05, 12.05])
        h5["s_i"] = amplitude * np.cos(phase)
        h5["s_q"] = amplitude * np.sin(phase)
    return h5_path
