# SPDX-License-Identifier: MIT

"""ICEYE SLC-COG and GRD-COG support module.

COG products are made of a Cloud Optimized GeoTiff raster and a STAC .json metadata file. Metadata are converted to the
same dataclasses used for legacy ICEYE products, so that downstream code can handle both formats in the same way.

Reference: ICEYE product documentation (https://sar.iceye.com), STAC product-spec schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from lxml import etree
from numpy.polynomial import Polynomial
from perseo_core.geometry.geocoding import direct_geocoding_monostatic
from perseo_core.timing import PreciseDateTime
from scipy.constants import speed_of_light

from eo_products.common.utilities import (
    ConversionFunction,
    CoordinatesConversions,
    DatasetInfo,
    DopplerEvaluator,
    OrbitDirection,
    PulseInfo,
    RasterInfo,
    RasterInfoAxis,
    SARPolarization,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
    StandardSARAcquisitionMode,
    StateVectors,
    SwathInfo,
)
from eo_products.iceye.raster import POWER_AMPLITUDE_MAPPING
from eo_products.iceye.utilities import (
    _COG_PRODUCT_TYPES,
    ICEYEAcquisitionMode,
    ICEYEChannelMetadata,
    ICEYEGeneralChannelInfo,
    ICEYEIncidenceAnglePolynomial,
    ICEYEOrbitProcessingLevel,
    ICEYEProductFormat,
    ICEYEProductLevel,
    ICEYERasterLayout,
    ICEYEStateVectorsReferenceSystem,
    InvalidICEYEProduct,
    burst_info_from_metadata,
    cog_raster_file_from_stac_item,
    read_stac_item,
)

_SUPPORTED_ORIENTATIONS = ("shadows-down",)
_IDENTITY_AMPLITUDE_MAPPINGS = ("minmax", "clip")  # values are already linear amplitude
_DEFAULT_POWER_EXPONENT = 0.25
_ACQUISITION_MODES = {
    "spotlight": ICEYEAcquisitionMode.SPOTLIGHT,
    "stripmap": ICEYEAcquisitionMode.STRIPMAP,
    "scan": ICEYEAcquisitionMode.TOPSAR,
    "topsar": ICEYEAcquisitionMode.TOPSAR,
}
_STANDARD_ACQUISITION_MODES = {
    ICEYEAcquisitionMode.SPOTLIGHT: StandardSARAcquisitionMode.SPOTLIGHT,
    ICEYEAcquisitionMode.STRIPMAP: StandardSARAcquisitionMode.STRIPMAP,
    ICEYEAcquisitionMode.TOPSAR: StandardSARAcquisitionMode.TOPSAR,
}
_POLARIZATIONS = {
    "vv": SARPolarization.VV,
    "hh": SARPolarization.HH,
    "hv": SARPolarization.HV,
    "vh": SARPolarization.VH,
}
# relative tolerance to consider the acquisition range sampling rate consistent with the slant range pixel spacing
_RANGE_SAMPLING_RTOL = 1e-4
# number of range points used to fit the Doppler rate polynomial
_DOPPLER_RATE_FIT_POINTS = 16
_DOPPLER_RATE_FIT_DEGREE = 3


def read_cog_channel_metadata(metadata_file: Path | str, channel_id: str) -> ICEYEChannelMetadata:
    """Reading ICEYE SLC-COG/GRD-COG channel metadata from the STAC .json file.

    Parameters
    ----------
    metadata_file : Path | str
        path to the STAC .json metadata file
    channel_id : str
        selected channel id

    Returns
    -------
    ICEYEChannelMetadata
        ICEYEChannelMetadata metadata dataclass
    """
    metadata_file = Path(metadata_file)
    item = read_stac_item(metadata_file)
    if item is None:
        raise InvalidICEYEProduct(f"{metadata_file} is not an ICEYE SLC-COG/GRD-COG STAC metadata file")
    raster_file = cog_raster_file_from_stac_item(item, metadata_file)
    if raster_file is None:
        raise InvalidICEYEProduct(f"COG raster file not found for {metadata_file}")
    properties = item["properties"]

    raster_layout = cog_raster_layout(properties=properties, raster_file=raster_file)
    general_info = general_info_from_stac(properties=properties, channel_id=channel_id)
    raster_info = raster_info_from_stac(properties=properties, raster_layout=raster_layout)
    burst_info = burst_info_from_metadata(raster_info=raster_info)
    dataset_info = dataset_info_from_stac(properties=properties)
    swath_info = swath_info_from_stac(properties=properties)
    state_vectors = state_vectors_from_stac(properties=properties)
    sampling_constants = sampling_constants_from_stac(properties=properties, raster_info=raster_info)
    pulse = pulse_info_from_stac(properties=properties)
    coordinate_conversions = coordinates_conversions_from_stac(properties=properties, raster_info=raster_info)
    range_times = range_time_axis(raster_info=raster_info, coordinate_conversions=coordinate_conversions)
    doppler_centroid_poly = doppler_centroid_poly_from_stac(
        properties=properties, raster_info=raster_info, reference_range_time=mid_range_time(range_times)
    )
    doppler_rate_poly = doppler_rate_poly_from_orbit(
        state_vectors=state_vectors,
        raster_info=raster_info,
        range_times=range_times,
        carrier_frequency=dataset_info.fc_hz,
        look_side=dataset_info.side_looking,
        altitude=float(properties.get("iceye:average_scene_height", 0.0)),
    )
    incidence_angles_poly = incidence_angle_poly_from_stac(properties=properties, raster_info=raster_info)
    calibration_factor, radiometric_quantity = calibration_factor_and_radiometric_quantity_from_stac(
        properties=properties
    )

    return ICEYEChannelMetadata(
        general_info=general_info,
        image_calibration_factor=calibration_factor,
        image_radiometric_quantity=radiometric_quantity,
        orbit=state_vectors.orbit,
        state_vectors=state_vectors,
        raster_info=raster_info,
        burst_info=burst_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversions,
        incidence_angles_poly=incidence_angles_poly,
        sampling_constants=sampling_constants,
        pulse=pulse,
        product_format=ICEYEProductFormat.COG,
        raster_layout=raster_layout,
    )


def product_level_from_stac(properties: dict) -> ICEYEProductLevel:
    """Product level (SLC or GRD) from STAC properties.

    Parameters
    ----------
    properties : dict
        STAC item properties

    Returns
    -------
    ICEYEProductLevel
        product level
    """
    return ICEYEProductLevel(_COG_PRODUCT_TYPES[properties["sar:product_type"]])


def acquisition_mode_from_stac(properties: dict) -> ICEYEAcquisitionMode:
    """Acquisition mode from STAC properties. The instrument mode is used, not the ICEYE processing mode code.

    Parameters
    ----------
    properties : dict
        STAC item properties

    Returns
    -------
    ICEYEAcquisitionMode
        acquisition mode
    """
    instrument_mode = properties["sar:instrument_mode"].lower()
    if instrument_mode not in _ACQUISITION_MODES:
        raise InvalidICEYEProduct(f"unsupported sar:instrument_mode {instrument_mode!r}")
    return _ACQUISITION_MODES[instrument_mode]


def _parse_gdal_band_scaling(gdal_metadata: str | None, bands: int) -> tuple[list[float | None], list[float | None]]:
    """Parsing band scale and offset items from GDAL_METADATA tag content."""
    scales: list[float | None] = [None] * bands
    offsets: list[float | None] = [None] * bands
    if not gdal_metadata:
        return scales, offsets
    root = etree.fromstring(gdal_metadata.encode())
    for node in root.iter("Item"):
        sample = node.get("sample")
        if sample is None or int(sample) >= bands or node.text is None:
            continue
        role = (node.get("role") or node.get("name") or "").lower()
        if role == "scale":
            scales[int(sample)] = float(node.text)
        elif role == "offset":
            offsets[int(sample)] = float(node.text)
    return scales, offsets


def cog_raster_layout(properties: dict, raster_file: Path) -> ICEYERasterLayout:
    """Raster layout of an ICEYE COG product (shadows-down storage), from STAC properties and GeoTiff tags.

    Parameters
    ----------
    properties : dict
        STAC item properties
    raster_file : Path
        path to the COG raster file

    Returns
    -------
    ICEYERasterLayout
        COG raster layout
    """
    product_level = product_level_from_stac(properties)
    orientation = properties.get("iceye:orientation", "shadows-down")
    if orientation not in _SUPPORTED_ORIENTATIONS:
        raise InvalidICEYEProduct(f"unsupported COG raster orientation {orientation!r}")

    with tifffile.TiffFile(raster_file) as tif:
        page = tif.pages[0]
        rows, cols, bands = page.imagelength, page.imagewidth, page.samplesperpixel
        integer_data = np.issubdtype(page.dtype, np.integer)
        gdal_metadata = page.tags["GDAL_METADATA"].value if "GDAL_METADATA" in page.tags else None

    expected_bands = 2 if product_level == ICEYEProductLevel.SLC else 1
    if bands != expected_bands:
        raise InvalidICEYEProduct(f"{product_level.value}-COG raster must have {expected_bands} band(s), not {bands}")

    scales, offsets = _parse_gdal_band_scaling(gdal_metadata, bands)
    if product_level == ICEYEProductLevel.SLC and integer_data and scales[1] is None and offsets[1] is None:
        # phase stored as unsigned integer spanning [-pi, pi]
        scales[1], offsets[1] = 2 * np.pi / np.iinfo(np.uint16).max, -np.pi

    mapping = properties.get("iceye:amplitude_mapping") or {}
    mapping_function = mapping.get("function", "minmax")
    parameters = mapping.get("parameters") or {}
    if mapping_function == POWER_AMPLITUDE_MAPPING:
        exponent = float(parameters.get("exponent", _DEFAULT_POWER_EXPONENT))
    elif mapping_function in _IDENTITY_AMPLITUDE_MAPPINGS:
        exponent = float(parameters.get("exponent", 1.0))
    else:
        raise InvalidICEYEProduct(f"unsupported amplitude mapping function {mapping_function!r}")

    return ICEYERasterLayout(
        product_level=product_level,
        look_side=properties["sar:observation_direction"].upper(),
        azimuth_lines=cols,
        range_samples=rows,
        band_scales=tuple(1.0 if s is None else s for s in scales),
        band_offsets=tuple(0.0 if o is None else o for o in offsets),
        amplitude_mapping_function=mapping_function,
        amplitude_mapping_scale=float(parameters.get("scale", 1.0)),
        amplitude_mapping_exponent=exponent,
    )


def general_info_from_stac(properties: dict, channel_id: str) -> ICEYEGeneralChannelInfo:
    """Creating an ICEYEGeneralChannelInfo object from STAC properties.

    Parameters
    ----------
    properties : dict
        STAC item properties
    channel_id : str
        channel id

    Returns
    -------
    ICEYEGeneralChannelInfo
        general channel info dataclass
    """
    product_level = product_level_from_stac(properties)
    acquisition_mode = acquisition_mode_from_stac(properties)
    return ICEYEGeneralChannelInfo(
        product_name=Path(properties["iceye:filename"]).stem,
        channel_id=channel_id,
        swath="S1",
        product_level=product_level,
        polarization=_POLARIZATIONS[properties["sar:polarizations"][0].lower()],
        projection=(
            SARProjection.GROUND_RANGE if product_level == ICEYEProductLevel.GRD else SARProjection.SLANT_RANGE
        ),
        acquisition_mode=acquisition_mode,
        acquisition_mode_std=_STANDARD_ACQUISITION_MODES[acquisition_mode],
        orbit_direction=OrbitDirection(properties["sat:orbit_state"].lower()),
        signal_frequency=float(properties["sar:center_frequency"]),
        acq_start_time=PreciseDateTime.fromisoformat(properties["start_datetime"]),
        acq_stop_time=PreciseDateTime.fromisoformat(properties["end_datetime"]),
    )


def raster_info_from_stac(properties: dict, raster_layout: ICEYERasterLayout) -> RasterInfo:
    """Creating a RasterInfo object from STAC properties.

    Azimuth lines are spaced by (zero Doppler end - zero Doppler start) / (lines - 1), i.e. the zero Doppler times of
    first and last line. The processing PRF is not the azimuth line rate of COG products.

    Parameters
    ----------
    properties : dict
        STAC item properties
    raster_layout : ICEYERasterLayout
        COG raster layout

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """
    lines = raster_layout.azimuth_lines
    lines_start = PreciseDateTime.fromisoformat(properties["iceye:zero_doppler_start_datetime"])
    lines_stop = PreciseDateTime.fromisoformat(properties["iceye:zero_doppler_end_datetime"])
    lines_step = (lines_stop - lines_start) / (lines - 1)

    samples = raster_layout.range_samples
    if raster_layout.product_level == ICEYEProductLevel.SLC:
        samples_start = 2 * float(properties["iceye:range_near"]) / speed_of_light
        samples_step = 2 * float(properties["sar:pixel_spacing_range"]) / speed_of_light
        sampling_rate = float(properties.get("iceye:acquisition_range_sampling_rate", 0.0))
        if sampling_rate > 0 and abs(samples_step * sampling_rate - 1) < _RANGE_SAMPLING_RTOL:
            # pixel spacing is rounded in metadata, the sampling rate is more accurate when consistent
            samples_step = 1 / sampling_rate
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"
    else:
        samples_start = 0
        samples_step = float(properties["sar:pixel_spacing_range"])
        samples_step_unit = "m"
        celltype = "FLOAT32"

    return RasterInfo(
        lines=RasterInfoAxis(length=lines, start=lines_start, step=lines_step, step_unit="s"),
        samples=RasterInfoAxis(length=samples, start=samples_start, step=samples_step, step_unit=samples_step_unit),
        data_type=celltype,
        raster_name=properties["iceye:filename"],
    )


def dataset_info_from_stac(properties: dict) -> DatasetInfo:
    """Creating a DatasetInfo object from STAC properties.

    Parameters
    ----------
    properties : dict
        STAC item properties

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """
    slc = product_level_from_stac(properties) == ICEYEProductLevel.SLC
    return DatasetInfo(
        fc_hz=float(properties["sar:center_frequency"]),
        acquisition_mode=acquisition_mode_from_stac(properties).value,
        image_type="AZIMUTH FOCUSED RANGE COMPENSATED" if slc else "MULTILOOK",
        projection="SLANT RANGE" if slc else "GROUND RANGE",
        sensor_name=properties["platform"],
        side_looking=properties["sar:observation_direction"].upper(),
    )


def swath_info_from_stac(properties: dict) -> SwathInfo:
    """Creating a SwathInfo object from STAC properties.

    Parameters
    ----------
    properties : dict
        STAC item properties

    Returns
    -------
    SwathInfo
        SwathInfo metadata object
    """
    prf = float(properties["iceye:acquisition_prf"])
    return SwathInfo(
        swath="S1",
        rank=np.floor(float(properties["iceye:range_near"]) / (speed_of_light / 2) * prf),
        azimuth_steering_rate_poly=(0, 0, 0),  # not annotated, as for legacy products
        prf=prf,
    )


def state_vectors_from_stac(properties: dict) -> StateVectors:
    """Generating StateVectors object from STAC properties.

    Parameters
    ----------
    properties : dict
        STAC item properties

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """
    frame = properties.get("iceye:coordinate_frame", "ecef").lower()
    if frame != "ecef":
        raise InvalidICEYEProduct(f"unsupported orbit coordinate frame {frame!r}, only Earth-fixed is supported")
    states = properties["iceye:orbit_states"]
    time_axis = np.array([PreciseDateTime.fromisoformat(s["time"]) for s in states])
    return StateVectors(
        num=len(states),
        positions=np.array([s["position"] for s in states], dtype=float),
        velocities=np.array([s["velocity"] for s in states], dtype=float),
        time_axis=time_axis,
        time_step=time_axis[1] - time_axis[0],
        orbit_direction=OrbitDirection(properties["sat:orbit_state"].lower()),
        orbit_type=ICEYEOrbitProcessingLevel(properties["iceye:orbit_precision"].lower()),
        reference_frame=ICEYEStateVectorsReferenceSystem.WGS84,
    )


def sampling_constants_from_stac(properties: dict, raster_info: RasterInfo) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies object from STAC properties.

    Parameters
    ----------
    properties : dict
        STAC item properties
    raster_info : RasterInfo
        channel raster info

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    if raster_info.samples.step_unit == "s":
        range_freq_hz = 1 / raster_info.samples.step
    else:
        range_freq_hz = float(properties["iceye:acquisition_range_sampling_rate"])
    return SARSamplingFrequencies(
        azimuth_freq_hz=1 / raster_info.lines.step,
        azimuth_bandwidth_freq_hz=float(properties["iceye:processing_bandwidth_azimuth"]),
        range_freq_hz=range_freq_hz,
        range_bandwidth_freq_hz=float(
            properties.get("iceye:processing_bandwidth_range", properties["iceye:pulse_bandwidth"])
        ),
    )


def pulse_info_from_stac(properties: dict) -> PulseInfo:
    """Creating a PulseInfo object from STAC properties.

    Parameters
    ----------
    properties : dict
        STAC item properties

    Returns
    -------
    PulseInfo
        PulseInfo metadata object
    """
    chirp_bandwidth = float(properties["iceye:pulse_bandwidth"])
    return PulseInfo(
        length_s=float(properties["iceye:pulse_duration"]),
        bandwidth_hz=chirp_bandwidth,
        energy_j=1,
        sampling_rate_hz=float(properties["iceye:acquisition_range_sampling_rate"]),
        start_frequency_hz=-chirp_bandwidth / 2,
        start_phase=0,
        direction="UP",
    )


def coordinates_conversions_from_stac(properties: dict, raster_info: RasterInfo) -> CoordinatesConversions:
    """Generating CoordinatesConversions object from STAC properties.

    GRD-COG products annotate a single ground range to slant range polynomial: evaluated at the ground range in
    meters from the first range pixel, it gives the slant range in meters. As for legacy GRD products, coefficients are
    converted to give the two-way range time in seconds, and the slant to ground conversion is obtained by fitting the
    inverse function over the raster range extent.

    Parameters
    ----------
    properties : dict
        STAC item properties
    raster_info : RasterInfo
        channel raster info

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass, empty for SLC-COG
    """
    if product_level_from_stac(properties) == ICEYEProductLevel.SLC:
        return CoordinatesConversions()

    az_ref_time = raster_info.lines.start
    ground_to_slant_poly = Polynomial(
        np.asarray(properties["iceye:ground_to_slant_coeff"], dtype=float) / (speed_of_light / 2)
    )
    ground_axis = np.arange(raster_info.samples.length + 1) * raster_info.samples.step
    slant_to_ground_poly = Polynomial.fit(
        x=ground_to_slant_poly(ground_axis), y=ground_axis, deg=ground_to_slant_poly.degree()
    )
    return CoordinatesConversions(
        azimuth_reference_times=np.array([az_ref_time]),
        ground_to_slant=[
            ConversionFunction(azimuth_reference_time=az_ref_time, origin=0, function=ground_to_slant_poly)
        ],
        slant_to_ground=[
            ConversionFunction(azimuth_reference_time=az_ref_time, origin=0, function=slant_to_ground_poly)
        ],
    )


def range_time_axis(raster_info: RasterInfo, coordinate_conversions: CoordinatesConversions) -> np.ndarray:
    """Two-way slant range time of each range sample.

    Parameters
    ----------
    raster_info : RasterInfo
        channel raster info
    coordinate_conversions : CoordinatesConversions
        coordinate conversions, used for ground range products

    Returns
    -------
    np.ndarray
        range times in seconds
    """
    axis = raster_info.samples.start + np.arange(raster_info.samples.length) * raster_info.samples.step
    if raster_info.samples.step_unit == "s":
        return axis
    return coordinate_conversions.evaluate_ground_to_slant(azimuth_time=raster_info.lines.start, ground_range=axis)


def mid_range_time(range_times: np.ndarray) -> float:
    """Range time at half swath, as for legacy products (start + step * length / 2).

    Parameters
    ----------
    range_times : np.ndarray
        range times of the range samples

    Returns
    -------
    float
        mid range time in seconds
    """
    return float(np.interp(range_times.size / 2, np.arange(range_times.size), range_times))


def doppler_centroid_poly_from_stac(
    properties: dict, raster_info: RasterInfo, reference_range_time: float
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator Doppler centroid polynomial wrapper from STAC properties.

    Polynomials are functions of the range time relative to the mid range time. As for legacy products, spotlight and
    scan Doppler centroids are averaged to a single polynomial for the whole acquisition, because the annotated one
    follows the antenna steering.

    Parameters
    ----------
    properties : dict
        STAC item properties
    raster_info : RasterInfo
        channel raster info
    reference_range_time : float
        mid swath range time, origin of the polynomials

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    coeff_list = properties["iceye:doppler_centroid_coeffs"]
    degree = max(len(c) for c in coeff_list)
    coeff_raw = np.zeros((len(coeff_list), degree))
    for index, coeff in enumerate(coeff_list):
        coeff_raw[index, : len(coeff)] = coeff
    time_axis = np.array([PreciseDateTime.fromisoformat(t) for t in properties["iceye:doppler_centroid_datetimes"]])

    if acquisition_mode_from_stac(properties) != ICEYEAcquisitionMode.STRIPMAP:
        coeff_raw = np.atleast_2d(coeff_raw.mean(axis=0))
        time_axis = np.array([raster_info.lines.start])

    functions = [
        ConversionFunction(azimuth_reference_time=time, origin=reference_range_time, function=Polynomial(coeff))
        for coeff, time in zip(coeff_raw, time_axis, strict=True)
    ]
    return DopplerEvaluator(functions=functions, azimuth_reference_times=time_axis)


def doppler_rate_poly_from_orbit(
    state_vectors: StateVectors,
    raster_info: RasterInfo,
    range_times: np.ndarray,
    carrier_frequency: float,
    look_side: str,
    altitude: float = 0.0,
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator Doppler rate polynomial wrapper from the orbit.

    The iceye:doppler_rate_coeffs annotation is not an azimuth FM rate, so the Doppler rate is computed from the orbit
    as -2 / wavelength * d2R/dt2 for stationary ground points at zero Doppler, at the mid azimuth time and across the
    swath, and fitted with a polynomial in range time relative to the mid range time.

    Parameters
    ----------
    state_vectors : StateVectors
        orbit's state vectors
    raster_info : RasterInfo
        channel raster info
    range_times : np.ndarray
        two-way slant range times of the range samples
    carrier_frequency : float
        carrier frequency in Hz
    look_side : str
        "LEFT" or "RIGHT"
    altitude : float, optional
        altitude of ground points above the WGS84 ellipsoid, by default 0.0

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Rate polynomial
    """
    trajectory = state_vectors.orbit
    wavelength = speed_of_light / carrier_frequency
    mid_azimuth_time = raster_info.lines.start + raster_info.lines.step * (raster_info.lines.length - 1) / 2
    position = trajectory.position(mid_azimuth_time)
    velocity = trajectory.velocity(mid_azimuth_time)
    acceleration = trajectory.acceleration(mid_azimuth_time)

    fit_range_times = np.linspace(range_times[0], range_times[-1], _DOPPLER_RATE_FIT_POINTS)
    ground_points = direct_geocoding_monostatic(
        sensor_positions=position,
        sensor_velocities=velocity,
        range_times=fit_range_times,
        doppler_frequencies=0.0,
        wavelength=wavelength,
        look_direction=look_side,
        altitude=altitude,
    ).reshape(-1, 3)
    line_of_sight = position - ground_points
    distance = np.linalg.norm(line_of_sight, axis=1)
    doppler_rate = (
        -2
        / wavelength
        / distance
        * (np.dot(velocity, velocity) + line_of_sight @ acceleration - (line_of_sight @ velocity / distance) ** 2)
    )

    reference_range_time = mid_range_time(range_times)
    rate_poly = Polynomial.fit(fit_range_times - reference_range_time, doppler_rate, deg=_DOPPLER_RATE_FIT_DEGREE)
    functions = [
        ConversionFunction(
            azimuth_reference_time=raster_info.lines.start,
            origin=reference_range_time,
            function=rate_poly.convert(),
        )
    ]
    return DopplerEvaluator(functions=functions, azimuth_reference_times=np.array([raster_info.lines.start]))


def incidence_angle_poly_from_stac(properties: dict, raster_info: RasterInfo) -> ICEYEIncidenceAnglePolynomial:
    """Creating an ICEYEIncidenceAnglePolynomial from STAC properties.

    The annotated polynomial gives the ellipsoid incidence angle in degrees as a function of the range pixel index,
    starting from 0 at the first range pixel.

    Parameters
    ----------
    properties : dict
        STAC item properties
    raster_info : RasterInfo
        channel raster info

    Returns
    -------
    ICEYEIncidenceAnglePolynomial
        polynomial for incidence angle computation
    """
    coefficients = properties.get("iceye:incidence_angle_coeffs")
    if not coefficients:
        return ICEYEIncidenceAnglePolynomial()
    ref_time = raster_info.lines.start
    return ICEYEIncidenceAnglePolynomial(
        azimuth_reference_times=np.array([ref_time]),
        incidence_angles=[
            ConversionFunction(azimuth_reference_time=ref_time, origin=0, function=Polynomial(coefficients))
        ],
    )


def calibration_factor_and_radiometric_quantity_from_stac(properties: dict) -> tuple[float, SARRadiometricQuantity]:
    """Image calibration factor and radiometric quantity from STAC properties.

    SLC-COG: |amplitude|^2 * calibration factor = beta nought

    GRD-COG: amplitude^2 * calibration factor = beta nought

    The STAC calibration factor calibrates the amplitude to radar brightness (beta nought) for both product levels,
    unlike legacy GRD products, which are sigma nought. SLC-COG and GRD-COG of the same acquisition annotate the same
    calibration factor and give the same mean beta nought over the same area.

    Parameters
    ----------
    properties : dict
        STAC item properties

    Returns
    -------
    tuple[float, SARRadiometricQuantity]
        image calibration factor (to be applied to amplitude),
        radiometric quantity
    """
    return np.sqrt(float(properties["iceye:calibration_factor"])), SARRadiometricQuantity.BETA_NOUGHT
