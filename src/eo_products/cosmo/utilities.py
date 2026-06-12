# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""COSMO reader support module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import h5py
import numpy as np
from numpy.polynomial import Polynomial
from perseo_core.geometry.navigation import Trajectory
from perseo_core.timing import PreciseDateTime
from scipy.constants import speed_of_light

from eo_products.common.utilities import (
    BurstInfo,
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

_DATA_EXTENSION = ".h5"
METERS_TO_SECONDS_CONVERSION = 1 / (speed_of_light / 2)


class InvalidCOSMOProduct(RuntimeError):
    """Invalid COSMO product"""


class COSMOGeneration(Enum):
    """COSMO Generation"""

    FIRST = 1
    SECOND = 2


class COSMOProductType(Enum):
    """COSMO Product Types"""

    DGM = "DGM"  # GRD
    SCS_B = "SCS_B"  # SLC product with range compensation applied
    SCS_U = "SCS_U"  # SLC product without range compensation applied


class COSMOAcquisitionModes(Enum):
    """COSMO supported acquisition modes"""

    SPOTLIGHT = auto()  # (Enhanced Spotlight)
    STRIPMAP = auto()  # (Himage, PingPong)
    SCANSAR = auto()  # (WideRegion, HugeRegion)


def raster_info_from_metadata(root: h5py.File, channel_id: str) -> RasterInfo:
    """Creating a RasterInfo metadata object from metadata file.

    Parameters
    ----------
    root : h5py.File
        root object
    channel_id : str
        channel id

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """

    product_name = Path(root.filename).name.replace(_DATA_EXTENSION, "")
    product_type = detect_product_type(root.attrs["Product Type"].decode())
    # Raster can have third axis if data is complex, in that case is real + imaginary
    raster = get_raster(root, channel_id)
    lines, samples, *_ = raster.shape

    # lines
    raster_lines = RasterInfoAxis(
        length=lines,
        start=(
            PreciseDateTime.from_utc_string(root.attrs["Reference UTC"].decode())
            + raster.attrs["Zero Doppler Azimuth First Time"]
        ),
        step=raster.attrs["Line Time Interval"],
        step_unit="s",
    )

    # samples
    if product_type == COSMOProductType.DGM:
        samples_start = 0
        samples_step = raster.attrs["Column Spacing"]
        samples_step_unit = "m"
        celltype = "FLOAT32"
    else:
        samples_start = raster.attrs["Zero Doppler Range First Time"]
        samples_step = raster.attrs["Column Time Interval"]
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"

    raster_samples = RasterInfoAxis(length=samples, start=samples_start, step=samples_step, step_unit=samples_step_unit)

    return RasterInfo(lines=raster_lines, samples=raster_samples, data_type=celltype, raster_name=product_name)


def burst_info_from_metadata(raster_info: RasterInfo, range_ref_time: float) -> BurstInfo:
    """Generating BurstInfo object directly from metadata.

    Parameters
    ----------
    raster_info : RasterInfo
        product raster info
    range_ref_time : float
        reference range start time

    Returns
    -------
    BurstInfo
        burst info dataclass
    """
    return BurstInfo(
        num=1,
        lines_per_burst=raster_info.lines.length,
        samples_per_burst=raster_info.samples.length,
        azimuth_start_times=np.array([raster_info.lines.start]),
        range_start_times=np.array([range_ref_time]),
    )


def dataset_info_from_metadata(root_attributes: h5py.AttributeManager) -> DatasetInfo:
    """Creating a DatasetInfo metadata object from metadata file.

    Parameters
    ----------
    root_attributes : h5py.AttributeManager
        metadata root attributes

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """
    prod_type = detect_product_type(root_attributes["Product Type"].decode())

    projection = "SLANT RANGE"
    match prod_type:
        case COSMOProductType.DGM:
            image_type = "MULTILOOK"
            projection = "GROUND RANGE"
        case COSMOProductType.SCS_B:
            image_type = "AZIMUTH FOCUSED RANGE COMPENSATED"
        case COSMOProductType.SCS_U:
            image_type = "AZIMUTH FOCUSED"

    return DatasetInfo(
        fc_hz=root_attributes["Radar Frequency"],
        acquisition_mode=detect_acquisition_mode(root_attributes["Acquisition Mode"].decode()).name,
        image_type=image_type,
        sensor_name=root_attributes["Satellite ID"].decode(),
        projection=projection,
        side_looking=root_attributes["Look Side"].decode(),
    )


def swath_info_from_metadata(root: h5py.File, channel_id: str) -> SwathInfo:
    """Creating a SwathInfo metadata object from metadata file.

    Parameters
    ----------
    root : h5py.File
        root object
    channel_id : str
        channel id

    Returns
    -------
    SwathInfo
        SwathInfo metadata object
    """
    swath_group = root[channel_id.split("_")[0]]
    rank = swath_group.attrs["Rank"]
    acquisition_prf = swath_group.attrs["PRF"]
    gen = get_cosmo_generation(root)

    # Pick steering from first burst
    if gen == COSMOGeneration.FIRST:
        azimuth_steering_deg = swath_group["B001"].attrs["Azimuth Steering"]
        line_changes = swath_group["B001"].attrs["Azimuth Ramp Code Change Lines"]
    else:
        azimuth_steering_deg = swath_group["B0001"].attrs["Azimuth Steering"]
        line_changes = swath_group["B0001"].attrs["Azimuth Ramp Code Change Lines"]

    if len(azimuth_steering_deg) > 1:
        steering_rate_rad_sec = np.deg2rad(
            (azimuth_steering_deg[-1] - azimuth_steering_deg[0])
            / (line_changes[-1] - line_changes[0])
            * acquisition_prf
        )
        azimuth_steering_rate_poly = (steering_rate_rad_sec, 0, 0)
    else:
        azimuth_steering_rate_poly = (0, 0, 0)

    return SwathInfo(
        swath=channel_id.split("_")[0],
        rank=rank,
        azimuth_steering_rate_poly=azimuth_steering_rate_poly,
        prf=acquisition_prf,
    )


def state_vectors_from_metadata(root_attributes: h5py.AttributeManager) -> StateVectors:
    """Generating StateVectors object directly from product metadata.

    Parameters
    ----------
    root_attributes : h5py.AttributeManager
        metadata root attributes

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """
    numerosity = root_attributes["State Vectors Times"].size
    positions = root_attributes["ECEF Satellite Position"]
    velocities = root_attributes["ECEF Satellite Velocity"]
    time_axis = root_attributes["State Vectors Times"] + PreciseDateTime.from_utc_string(
        root_attributes["Reference UTC"].decode()
    )

    assert positions.shape[0] == velocities.shape[0] == time_axis.size == numerosity

    return StateVectors(
        num=numerosity,
        positions=positions,
        velocities=velocities,
        time_axis=time_axis,
        time_step=time_axis[1] - time_axis[0],
        orbit_direction=OrbitDirection[root_attributes["Orbit Direction"].decode()],
    )


def sampling_constants_from_metadata(
    swath_attributes: h5py.AttributeManager,
    raster_info: RasterInfo,
) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata object from metadata file.

    Parameters
    ----------
    swath_attributes : h5py.AttributeManager
        metadata swath attributes
    raster_info : RasterInfo
        raster info

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """

    range_freq_hz = 1 / raster_info.samples.step
    range_bandwidth_freq_hz = swath_attributes["Range Focusing Bandwidth"]

    azimuth_freq_hz = 1 / raster_info.lines.step
    azimuth_bandwidth_freq_hz = swath_attributes["Azimuth Focusing Transition Bandwidth"]

    return SARSamplingFrequencies(
        azimuth_freq_hz=azimuth_freq_hz,
        azimuth_bandwidth_freq_hz=azimuth_bandwidth_freq_hz,
        range_freq_hz=range_freq_hz,
        range_bandwidth_freq_hz=range_bandwidth_freq_hz,
    )


def pulse_info_from_metadata(swath_attributes: h5py.AttributeManager) -> PulseInfo:
    """Creating a PulseInfo metadata object from metadata.

    Parameters
    ----------
    swath_attributes : h5py.AttributeManager
        metadata swath attributes

    Returns
    -------
    PulseInfo
        PulseInfo metadata object
    """

    # TODO: check this
    chirp_bandwidth = 0
    return PulseInfo(
        length_s=swath_attributes["Range Chirp Length"],
        bandwidth_hz=chirp_bandwidth,
        energy_j=1,
        sampling_rate_hz=swath_attributes["Sampling Rate"],
        start_frequency_hz=-chirp_bandwidth / 2,
        start_phase=0,
        direction="UP",
    )


def doppler_centroid_poly_from_metadata(root: h5py.File, channel_id: str) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    root : h5py.File
        root object
    channel_id : str
        channel id

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    gen = get_cosmo_generation(root)
    if gen == COSMOGeneration.FIRST:
        attrs = root.attrs
        doppler_rate_coeffs = attrs["Centroid vs Range Time Polynomial"]
    else:
        attrs = root[channel_id.split("_")[0]].attrs
        doppler_rate_coeffs = attrs["Doppler Centroid vs Range Time Polynomial"]

    ref_az_time = attrs["Azimuth Polynomial Reference Time"] + PreciseDateTime.from_utc_string(
        root.attrs["Reference UTC"].decode()
    )
    ref_rng_time = attrs["Range Polynomial Reference Time"]
    # assembling list of polynomials
    doppler_poly_list = [
        ConversionFunction(
            azimuth_reference_time=ref_az_time,
            origin=ref_rng_time,
            function=Polynomial(doppler_rate_coeffs),
        )
    ]

    return DopplerEvaluator(functions=doppler_poly_list, azimuth_reference_times=np.array([ref_az_time]))


def doppler_rate_poly_from_metadata(root: h5py.File, channel_id: str) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler rate vector polynomial wrapper from metadata.

    Parameters
    ----------
    root : h5py.File
        root object
    channel_id : str
        channel id

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Rate polynomial
    """
    gen = get_cosmo_generation(root)
    if gen == COSMOGeneration.FIRST:
        attrs = root.attrs
    else:
        attrs = root[channel_id.split("_")[0]].attrs
    doppler_rate_coeffs = attrs["Doppler Rate vs Range Time Polynomial"]
    ref_az_time = attrs["Azimuth Polynomial Reference Time"] + PreciseDateTime.from_utc_string(
        root.attrs["Reference UTC"].decode()
    )
    ref_rng_time = attrs["Range Polynomial Reference Time"]

    # assembling list of polynomials
    doppler_poly_list = [
        ConversionFunction(
            azimuth_reference_time=ref_az_time,
            origin=ref_rng_time,
            function=Polynomial(doppler_rate_coeffs),
        )
    ]

    return DopplerEvaluator(functions=doppler_poly_list, azimuth_reference_times=np.array([ref_az_time]))


def coordinates_conversions_from_metadata(
    root_attributes: h5py.AttributeManager, azimuth_ref: PreciseDateTime, range_step_m: float, projection: SARProjection
) -> CoordinatesConversions:
    """Generating CoordinatesConversions object from metadata.

    About coefficients annotated in metadata:

    Coefficients of polynomial fit to the "Ground to Slant Range" transform applied. Fixed along all image slices.
    Defined with respect to (slant range in m)/(pixel no.)^n where first pixel in line is 0.

    Values output in order A0, A1, ...,  An, in order of increasing degree.
    Polynomial to be evaluated is:  A_0+ A_1 x + ... + A_n x^n
    where x is pixel number in the line, starting from 0.

    Evaluated polynomial gives Slant Range in meters.

    Parameters
    ----------
    root_attributes : h5py.AttributeManager
        metadata root attributes
    azimuth_ref : PreciseDateTime
        reference azimuth time
    range_step_m : float
        range step in meters
    projection : SARProjection
        product projection

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass
    """

    # TODO: add coordinate conversion section for slant range products
    if projection != SARProjection.GROUND_RANGE:
        return CoordinatesConversions()

    origin = root_attributes["Ground Projection Polynomial Reference Range"]
    slant_origin = origin * METERS_TO_SECONDS_CONVERSION
    # divide by samples step in meters
    # check with current translated product and its usage in protocol for PF
    slant_to_ground_coefficients = [
        c / (METERS_TO_SECONDS_CONVERSION**c_id) * range_step_m
        for c_id, c in enumerate(root_attributes["Slant to Ground Polynomial"])
    ]
    ground_to_slant_coefficients = [
        c / (range_step_m**c_id) * METERS_TO_SECONDS_CONVERSION
        for c_id, c in enumerate(root_attributes["Ground to Slant Polynomial"])
    ]
    ground_to_slant_coefficients[0] += slant_origin

    ground_to_slant_poly = Polynomial(ground_to_slant_coefficients)
    slant_to_ground_poly = Polynomial(slant_to_ground_coefficients)

    # assembling list of polynomials
    ground_to_slant_poly_list = [
        ConversionFunction(
            azimuth_reference_time=azimuth_ref,
            origin=0,
            function=ground_to_slant_poly,
        )
    ]
    slant_to_ground_poly_list = [
        ConversionFunction(
            azimuth_reference_time=azimuth_ref,
            origin=slant_origin,
            function=slant_to_ground_poly,
        )
    ]

    return CoordinatesConversions(
        azimuth_reference_times=np.array([azimuth_ref]),
        ground_to_slant=ground_to_slant_poly_list,
        slant_to_ground=slant_to_ground_poly_list,
    )


def compute_calibration_factor(root: h5py.File, channel_id: str) -> float:
    """Computing calibration factor to be applied to the image raster to get Sigma Nought radiometric quantity.

    Parameters
    ----------
    root : h5py.File
        root object
    channel_id : str
        channel id

    Returns
    -------
    float
        calibration factor
    """
    root_attributes = root.attrs
    raster_attributes = get_raster(root, channel_id).attrs
    swath_attributes = root[channel_id.split("_")[0]].attrs
    calibration_constant = swath_attributes["Calibration Constant"]
    gen = get_cosmo_generation(root)
    if gen == COSMOGeneration.FIRST:
        factor = 1 / (root_attributes["Rescaling Factor"] ** 2)
        if root_attributes["Range Spreading Loss Compensation Geometry"].decode() != "NONE":
            factor *= root_attributes["Reference Slant Range"] ** (
                2 * root_attributes["Reference Slant Range Exponent"]
            )
        if root_attributes["Incidence Angle Compensation Geometry"].decode() != "NONE":
            factor *= np.sin(np.deg2rad(root_attributes["Reference Incidence Angle"]))
        if root_attributes["Calibration Constant Compensation Flag"] == 0:
            factor *= 1 / calibration_constant
    else:
        # TODO: check if this is calibrated
        factor = 1 / (raster_attributes["Rescaling Factor"] ** 2)

    return np.sqrt(factor)


def detect_acquisition_mode(acq_mode_str: str) -> COSMOAcquisitionModes:
    """Detect Acquisition Mode from string.

    Parameters
    ----------
    acq_mode_str : str
        annotated acquisition mode

    Returns
    -------
    COSMOAcquisitionModes
        COSMO Acquisition Mode enum
    """
    if "region" in acq_mode_str.lower() or "scansar" in acq_mode_str.lower():  # Expected HUGEREGION or WIDEREGION
        return COSMOAcquisitionModes.SCANSAR

    if "spotlight" in acq_mode_str.lower():
        return COSMOAcquisitionModes.SPOTLIGHT

    return COSMOAcquisitionModes.STRIPMAP


def detect_product_type(product_type_str: str) -> COSMOProductType:
    """Detect Product Type from string.

    Parameters
    ----------
    product_type_str : str
        annotated product type

    Returns
    -------
    COSMOProductType
        COSMO Product Type enum
    """

    if "DGM" in product_type_str:
        return COSMOProductType.DGM

    if "SCS_B" in product_type_str:
        return COSMOProductType.SCS_B

    return COSMOProductType.SCS_U


def _get_channels_names(root: h5py.File) -> list[str]:
    """Combining swath names and polarizations to get the channel ids.

    Parameters
    ----------
    root : h5py.File
        root object

    Returns
    -------
    list[str]
        list of channel ids as "swath_polarization"
    """
    swaths = [s for s in root.keys() if "S0" in s]
    polarizations = _get_polarizations_str(root)

    if len(polarizations) == 1 and len(swaths) > 1:
        polarizations = polarizations * len(swaths)

    return ["_".join([s, p]) for s, p in list(zip(swaths, polarizations, strict=True))]


def _get_footprint(rasters: list[h5py.Dataset]) -> list[float, float, float, float]:
    """Get scene footprint from corner coordinates.

    Parameters
    ----------
    raster : h5py.Dataset
        raster dataset

    Returns
    -------
    list[float, float, float, float]
        min latitude, max latitude, min longitude, max longitude
    """
    footprint = np.stack(
        [
            [
                raster.attrs["Bottom Left Geodetic Coordinates"][:-1],
                raster.attrs["Bottom Right Geodetic Coordinates"][:-1],
                raster.attrs["Top Left Geodetic Coordinates"][:-1],
                raster.attrs["Top Right Geodetic Coordinates"][:-1],
            ]
            for raster in rasters
        ]
    )
    min_lat, min_lon = np.min(np.min(footprint, axis=1), axis=0)
    max_lat, max_lon = np.max(np.max(footprint, axis=1), axis=0)

    return [float(min_lat), float(max_lat), float(min_lon), float(max_lon)]


def _get_polarizations_str(root: h5py.Filer) -> list[str]:
    """Get polarizations values from product.

    Parameters
    ----------
    root : h5py.Filer
        root object

    Returns
    -------
    list[str]
        list of polarizations strings
    """
    gen = get_cosmo_generation(root)
    swaths = [s for s in root.keys() if "S0" in s]
    if gen == COSMOGeneration.FIRST:
        return [root[s].attrs["Polarisation"].decode() for s in swaths]
    return [root.attrs["Polarization"].decode()]


def get_raster(root: h5py.File, channel_id: str) -> h5py.Dataset:
    """Get the raster associated to a channel.

    Parameters
    ----------
    root : h5py.File
        root object
    channel_id : str
        selected channel id

    Returns
    -------
    h5py.Dataset
        Raster dataset
    """
    # Recover raster. If product is geocoded or geodetected and it's a ScanSAR, there's the single MBI raster, otherwise
    # pick the SBI associated to swath
    product_type = detect_product_type(root.attrs["Product Type"].decode())
    acquisition_mode = detect_acquisition_mode(root.attrs["Acquisition Mode"].decode())
    gen = get_cosmo_generation(root)

    if product_type == COSMOProductType.DGM and acquisition_mode == COSMOAcquisitionModes.SCANSAR:
        return root["MBI"]

    if gen == COSMOGeneration.FIRST:
        return root[channel_id.split("_")[0]]["SBI"]
    return root[channel_id.split("_")[0]]["IMG"]


def get_cosmo_generation(root: h5py.File) -> COSMOGeneration:
    """Get COSMO product generation.

    Parameters
    ----------
    root : h5py.File
        root object

    Returns
    -------
    COSMOGeneration
        COSMO product generation
    """
    if root.attrs["Mission ID"].decode() == "CSG":
        return COSMOGeneration.SECOND
    return COSMOGeneration.FIRST


@dataclass
class COSMOGeneralChannelInfo:
    """COSMO general channel info dataclass"""

    product_name: str
    channel_id: str
    swath: str
    product_level: COSMOProductType
    polarization: SARPolarization
    projection: SARProjection
    acquisition_mode: COSMOAcquisitionModes
    acquisition_mode_std: StandardSARAcquisitionMode
    orbit_direction: OrbitDirection
    signal_frequency: float
    acq_start_time: PreciseDateTime
    acq_stop_time: PreciseDateTime

    @classmethod
    def from_metadata(cls, root: h5py.File, channel_id: str) -> COSMOGeneralChannelInfo:
        """Generating COSMOGeneralChannelInfo object directly from metadata.

        Parameters
        ----------
        root : h5py.File
            root object
        channel_id : str
            channel id

        Returns
        -------
        COSMOGeneralChannelInfo
            general channel info dataclass
        """
        product_name = Path(root.filename).name.replace(_DATA_EXTENSION, "")
        product_level = detect_product_type(root.attrs["Product Type"].decode())
        acq_mode = detect_acquisition_mode(root.attrs["Acquisition Mode"].decode())
        if acq_mode == COSMOAcquisitionModes.SCANSAR:
            acq_mode_std = StandardSARAcquisitionMode.SCANSAR
        elif acq_mode == COSMOAcquisitionModes.STRIPMAP:
            acq_mode_std = StandardSARAcquisitionMode.STRIPMAP
        else:
            acq_mode_std = StandardSARAcquisitionMode.SPOTLIGHT
        return cls(
            product_name=product_name,
            channel_id=channel_id,
            swath=channel_id.split("_")[0],
            product_level=product_level,
            polarization=SARPolarization[channel_id.split("_")[-1].upper()],
            projection=(
                SARProjection.GROUND_RANGE if product_level == COSMOProductType.DGM else SARProjection.SLANT_RANGE
            ),
            acquisition_mode=acq_mode,
            acquisition_mode_std=acq_mode_std,
            orbit_direction=OrbitDirection[root.attrs["Orbit Direction"].decode()],
            signal_frequency=root.attrs["Radar Frequency"],
            acq_start_time=PreciseDateTime.from_utc_string(root.attrs["Scene Sensing Start UTC"].decode()),
            acq_stop_time=PreciseDateTime.from_utc_string(root.attrs["Scene Sensing Stop UTC"].decode()),
        )


@dataclass
class COSMOChannelMetadata:
    """COSMO channel metadata dataclass"""

    general_info: COSMOGeneralChannelInfo
    orbit: Trajectory
    image_calibration_factor: float
    image_radiometric_quantity: SARRadiometricQuantity
    burst_info: BurstInfo
    raster_info: RasterInfo
    dataset_info: DatasetInfo
    swath_info: SwathInfo
    sampling_constants: SARSamplingFrequencies
    doppler_centroid_poly: DopplerEvaluator
    doppler_rate_poly: DopplerEvaluator
    pulse: PulseInfo
    coordinate_conversions: CoordinatesConversions
    state_vectors: StateVectors


class COSMOProduct:
    """COSMO product object"""

    def __init__(self, path: str | Path) -> None:
        """COSMO Product init from directory path.

        Parameters
        ----------
        path : str | Path
            path to COSMO product
        """
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        root = h5py.File(path)
        self._generation = get_cosmo_generation(root)
        product_type = detect_product_type(root.attrs["Product Type"].decode())
        self._acq_time = PreciseDateTime.from_utc_string(root.attrs["Scene Sensing Start UTC"].decode())
        channels_list = _get_channels_names(root)
        if product_type == COSMOProductType.DGM:
            # swaths and burst are already merged, so taking only the first swath as reference because
            # swath attributes are not used in this case
            channels_list = [channels_list[0]]
        self._channels_list = channels_list
        self._channels_number = len(self._channels_list)
        # Get all the rasters to compute footprint
        rasters = [get_raster(root, channel_id) for channel_id in channels_list]
        self._footprint = _get_footprint(rasters)
        root.close()

    @property
    def generation(self) -> COSMOGeneration:
        """COSMO product generation"""
        return self._generation

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def channels_number(self) -> int:
        """Returning the number of channels of COSMO product"""
        return self._channels_number

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels"""
        return self._channels_list

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint


def is_cosmo_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid COSMO product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a .h5 file
        - open file and read acquisition mode

    Parameters
    ----------
    product : str | Path
        path to the product to be checked

    Returns
    -------
    bool
        True if it is a valid product, else False
    """
    product = Path(product)

    if not product.exists():
        return False

    if not str(product).endswith(_DATA_EXTENSION):
        return False

    # open product, read acquisition mode
    try:
        root = h5py.File(product)
        detect_product_type(root.attrs["Product Type"].decode())
    except Exception:
        return False
    finally:
        root.close()

    return True
