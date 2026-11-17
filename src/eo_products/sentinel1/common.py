# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Sentinel-1 product format reader common utilities and structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from perseo_core.geometry.coordinates import xyz2llh
from perseo_core.geometry.navigation import Trajectory
from perseo_core.timing import PreciseDateTime

from eo_products.common.utilities import (
    AcquisitionTimeline,
    BurstInfo,
    CoordinatesConversions,
    DatasetInfo,
    DopplerEvaluator,
    OrbitDirection,
    RasterInfo,
    SARPolarization,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
    StandardSARAcquisitionMode,
    StateVectors,
    SwathInfo,
)

S1_SIDE_LOOKING = "RIGHT"
_S1_TX_PULSE_LATCH_TIME = 1.439e-6  # [s]
ALPHA_ROLL = 0.0566
SENSOR_HEIGHT_REF = 711.7  # km
BORESIGHT_ANGLE_REF = 29.450  # deg


class InvalidTimeBoundaries(RuntimeError):
    """Invalid time boundaries with respect to validity period"""


class InvalidSentinel1Product(RuntimeError):
    """Invalid Sentinel-1 product."""


class S1AcquisitionMode(Enum):
    """Sentinel-1 acquisition modes"""

    IW = "IW"  # Interferometric Wide swath (IW)
    EW = "EW"  # Extra-Wide swath (EW)
    WV = "WV"  # Wave (WV): single polarization only (HH or VV)
    SM = "SM"  # Stripmap (SM)
    EN = "EN"  # Elevation Notch (instrument calibration mode)
    AN = "AN"  # Azimuth Notch (instrument calibration mode)
    S1 = "S1"  # Stripmap S1
    S2 = "S2"  # Stripmap S2
    S3 = "S3"  # Stripmap S3
    S4 = "S4"  # Stripmap S4
    S5 = "S5"  # Stripmap S5
    S6 = "S6"  # Stripmap S6


class S1L1ProductType(Enum):
    """Sentinel-1 L1 product types"""

    SLC = "SLC"  # Slant Range, Single Look Complex (SLC, lvl 1)
    GRD = "GRD"  # Ground Range Multi Look Detected (GRD, lvl 1, phase lost)
    NRB = "NRB"  # Analysis Ready Data (ARD), Normalized Radar Backscatter (NRB)


class S1DCEstimateMethod(Enum):
    """Doppler Centroid estimate method: data or geometry"""

    DATA = "DATA"
    GEOMETRY = "GEOMETRY"


class S1OrbitType(Enum):
    """EOrbitType class"""

    DOWNLINK = "DOWNLINK"
    PREDICTED = "PREDICTED"
    RESTITUTED = "RESTITUTED"
    PRECISE = "PRECISE"
    UNKNOWN = "UNKNOWN"


class S1ReferenceFrameType(Enum):
    """Sentinel-1 available reference frames"""

    UNDEFINED = "Undefined"
    GALACTIC = "Galactic"
    BM1950 = "BM1950"
    BM2000 = "BM2000"
    HM2000 = "HM2000"
    GM2000 = "GM2000"
    MEAN_OF_DATE = "Mean Of Date"
    TRUE_OF_DATE = "True Of Date"
    PSEUDO_EARTH_FIXED = "Pseudo Earth Fixed"
    EARTH_FIXED = "Earth Fixed"
    TOPOCENTRIC = "Topocentric"
    SATELLITE_ORBITAL = "Satellite Orbital"
    SATELLITE_NORMAL = "Satellite Nominal"
    SATELLITE_ATTITUDE = "Satellite Attitude"
    INSTRUMENT_ATTITUDE = "Instrument Attitude"


@dataclass
class S1ChannelMetadata:
    """Sentinel-1 channel metadata content."""

    general_info: S1GeneralChannelInfo
    image_radiometric_quantity: SARRadiometricQuantity
    raster_info: RasterInfo
    burst_info: BurstInfo
    dataset_info: DatasetInfo
    swath_info: SwathInfo
    sampling_constants: SARSamplingFrequencies
    state_vectors: StateVectors
    orbit: Trajectory
    doppler_centroid_poly: DopplerEvaluator
    doppler_rate_poly: DopplerEvaluator
    coordinate_conversions: CoordinatesConversions
    antenna_pattern: dict[str, S1AntennaPattern]  # dictionary key is the swath
    burst_sensing_times: np.ndarray | None  # burst sensing times array
    acquisition_timeline: AcquisitionTimeline
    pulse: S1Pulse
    chirp_replica: dict[str, S1ChirpReplica]  # dictionary key is the swath
    noise: dict[str, S1Noise]  # dictionary key is the swath
    attitude: S1Attitude | None
    calibration_factor: float | None


@dataclass
class S1GeneralChannelInfo:
    """Sentinel-1 general channel info representation dataclass"""

    mission_id: str
    swath: str
    product_type: S1L1ProductType
    polarization: SARPolarization
    projection: SARProjection
    mode: S1AcquisitionMode
    acquisition_mode_std: StandardSARAcquisitionMode
    orbit_direction: OrbitDirection
    signal_frequency: float
    start_time: PreciseDateTime
    stop_time: PreciseDateTime


@dataclass
class S1Attitude:
    """Sentinel-1 sensor's attitude"""

    num: int  # attitude data numerosity
    frame: S1ReferenceFrameType  # reference frame of the attitude data.
    quaternions: np.ndarray  # attitude quaternion as extracted from ancillary attitude data (N, 4)
    angular_velocities: np.ndarray  # angular velocity as extracted from ancillary attitude data [degrees/s] (N, 3)
    roll: np.ndarray  # platform roll
    pitch: np.ndarray  # platform pitch
    yaw: np.ndarray  # platform yaw
    time_axis: np.ndarray  # PreciseDateTime axis to which attitude data applies


@dataclass
class S1Pulse:
    """Sentinel-1 pulse info"""

    length: float
    bandwidth: float
    energy: float
    start_frequency: float
    start_phase: float
    direction: str
    tx_pulse_latch_time: float = _S1_TX_PULSE_LATCH_TIME


@dataclass
class S1AntennaPattern:
    """Sentinel-1 antenna pattern dataclass"""

    swath: str  # swath of to the current antenna pattern data
    time_axis: np.ndarray  # Zero Doppler azimuth time at which antenna pattern applies
    slant_range_time_array: np.ndarray  # two-way slant range time array for this antenna pattern [s]
    elevation_angle_array: np.ndarray  # corresponding elevation angle for this antenna pattern [degrees]
    elevation_pattern_array: np.ndarray  # corresponding two-way antenna elevation pattern value for this point
    incidence_angle: np.ndarray  # corresponding incidence angle value for this point [degrees]
    terrain_height: np.ndarray  # average terrain height in range for this antenna pattern [m]
    roll: np.ndarray  # estimated roll angle for this antenna pattern [degrees]


@dataclass
class S1Noise:
    """Sentinel-1 thermal noise parameters derived from noise packets dataclass"""

    swath: str  # swath of to the current noise data
    time_axis: np.ndarray  # Zero Doppler azimuth time of the noise measurement
    noise_power_correction_factor: np.ndarray  # noise power correction factor
    noise_lines_num: np.ndarray  # number of noise lines used to calculate noise correction factor


@dataclass
class S1ChirpReplica:
    """Sentinel-1 PG chirp replica parameters derived from the calibration pulses at 100 MHz bandwidth dataclass"""

    swath: str  # swath of to the current chirp replica
    time_axis: np.ndarray  # Zero Doppler azimuth time at which replica applies
    cross_correlation_bandwidth: (
        np.ndarray
    )  # 3-dB pulse width of cross-correlation function between the reconstructed replica and the nominal replica
    cross_correlation_pslr: (
        np.ndarray
    )  # pslr of cross-correlation function between the reconstructed replica and the nominal replica
    cross_correlation_islr: (
        np.ndarray
    )  # islr of cross-correlation function between the reconstructed replica and the nominal replica
    cross_correlation_peak_location: (
        np.ndarray
    )  # peak location of cross-correlation function between the reconstructed replica and the nominal replica [samples]
    pg_product_amplitude: np.ndarray  # amplitude of the PG product derived from this replica
    pg_product_phase: np.ndarray  # phase of the PG product derived from this replica [radians]
    model_pg_product_amplitude: np.ndarray  # PG product amplitude value from the input PG product model
    model_pg_product_phase: np.ndarray  # PG product phase value from the input PG product model [radians]
    internal_time_delay: (
        np.ndarray
    )  # calculated deviation of the location of this PG replica from the location of the transmitted pulse
    reconstructed_replica_validity: list[
        bool
    ]  # if the cross-correlation between the nominal and this extracted PG replica resulted in a valid peak location
    relative_validity: list[bool]  # if the amplitude and phase of the PG product passed relative validation
    absolute_validity: list[bool]  # if the amplitude and phase of the PG product passed the absolute validation


@dataclass
class S1AzimuthNoiseVector:
    """Sentinel-1 noise vector dataclass"""

    swath: str
    first_azimuth_line: int
    last_azimuth_line: int
    first_range_sample: int
    last_range_sample: int
    lines: np.ndarray
    lut: np.ndarray


@dataclass
class S1RangeNoiseVector:
    """Sentinel-1 noise vector dataclass"""

    azimuth_time: PreciseDateTime
    line: int
    pixels: np.ndarray
    lut: np.ndarray


def get_standard_acq_mode(mode: S1AcquisitionMode) -> StandardSARAcquisitionMode:
    """Conversion of annotated acquisition mode to standard acquisition mode.

    Parameters
    ----------
    mode : str
        annotated acquisition mode

    Returns
    -------
    StandardSARAcquisitionMode
        standard acquisition mode
    """
    if mode in (S1AcquisitionMode.EW, S1AcquisitionMode.IW):
        mode_std = StandardSARAcquisitionMode.TOPSAR
    elif mode == S1AcquisitionMode.WV:
        mode_std = StandardSARAcquisitionMode.WAVE
    elif mode in (
        S1AcquisitionMode.SM,
        S1AcquisitionMode.S1,
        S1AcquisitionMode.S2,
        S1AcquisitionMode.S3,
        S1AcquisitionMode.S4,
        S1AcquisitionMode.S5,
        S1AcquisitionMode.S6,
    ):
        mode_std = StandardSARAcquisitionMode.STRIPMAP
    elif mode == S1AcquisitionMode.EN:
        mode_std = StandardSARAcquisitionMode.ELEVATION_NOTCH
    else:
        mode_std = StandardSARAcquisitionMode.UNKNOWN
    return mode_std


def _get_noise_vector_at(
    swath: str,
    azimuth_index: int,
    azimuth_noise_vectors: list[S1AzimuthNoiseVector] | None,
    range_noise_vectors: list[S1RangeNoiseVector],
    is_grd: bool,
) -> np.ndarray:
    if azimuth_noise_vectors is None:
        azimuth_noise_factor = np.ones(range_noise_vectors[0].pixels[-1] + 1)
    else:
        if is_grd:
            swath_az_vectors = [v for v in azimuth_noise_vectors if swath in v.swath]
        else:
            swath_az_vectors = [v for v in azimuth_noise_vectors if v.swath == swath]
        selected = [s for s in swath_az_vectors if s.first_azimuth_line <= azimuth_index <= s.last_azimuth_line]
        if not selected:
            raise ValueError(f"No azimuth noise vectors found for given index {azimuth_index}")

        chunks = []
        for s in selected:
            val = np.interp(azimuth_index, s.lines, s.lut)
            length = s.last_range_sample - s.first_range_sample + 1
            chunks.append(np.full(length, val))

        azimuth_noise_factor = np.concatenate(chunks)

    lines = np.array([r.line for r in range_noise_vectors])
    idx = np.searchsorted(lines, azimuth_index, side="right") - 1
    noise_vector_idx = int(np.clip(idx, 0, len(lines) - 1))

    if noise_vector_idx < len(range_noise_vectors) - 1:
        range_noise_vector_0 = np.interp(
            np.arange(0, azimuth_noise_factor.size),
            range_noise_vectors[noise_vector_idx].pixels,
            range_noise_vectors[noise_vector_idx].lut,
        )
        range_noise_vector_1 = np.interp(
            np.arange(0, azimuth_noise_factor.size),
            range_noise_vectors[noise_vector_idx + 1].pixels,
            range_noise_vectors[noise_vector_idx + 1].lut,
        )
        interp_weight = (azimuth_index - range_noise_vectors[noise_vector_idx].line) / (
            range_noise_vectors[noise_vector_idx + 1].line - range_noise_vectors[noise_vector_idx].line
        )
        range_noise_vector = (1 - interp_weight) * range_noise_vector_0 + interp_weight * range_noise_vector_1
    else:
        range_noise_vector = np.interp(
            np.arange(0, azimuth_noise_factor.size),
            range_noise_vectors[noise_vector_idx].pixels,
            range_noise_vectors[noise_vector_idx].lut,
        )
    return range_noise_vector * azimuth_noise_factor


def get_noise_vectors(
    swath: str,
    azimuth_indexes: int | tuple[int, int],
    azimuth_noise_vectors: list[S1AzimuthNoiseVector] | None,
    range_noise_vectors: list[S1RangeNoiseVector],
    is_grd: bool,
) -> np.ndarray:
    """Get the noise vector(s) from Sentinel-1 annotation file for the given azimuth index(es).

    Parameters
    ----------
    swath : str
        current swath name
    azimuth_indexes : int | tuple[int, int]
        azimuth index (int) at which the noise vector is to be computed, or a tuple [start, stop) of azimuth indices to
        get multiple vectors in the inclusive range (start included, stop excluded).
    azimuth_noise_vectors : list[S1AzimuthNoiseVector] | None
        azimuth noise vectors, if any else None
    range_noise_vectors : list[S1RangeNoiseVector]
        range noise vectors
    is_grd : bool
        flag to indicate if the product is a GRD or not

    Returns
    -------
    np.ndarray
        noise vectors along range samples, with shape (N,) when azimuth_index is an int, (M, N) when azimuth_index
        is a tuple.

    Raises
    ------
    ValueError
        if no azimuth noise vectors are found for the given azimuth index
    """
    if isinstance(azimuth_indexes, tuple):
        start, stop = azimuth_indexes
        return np.array(
            [
                _get_noise_vector_at(swath, idx, azimuth_noise_vectors, range_noise_vectors, is_grd)
                for idx in range(start, stop)
            ]
        )
    return _get_noise_vector_at(swath, azimuth_indexes, azimuth_noise_vectors, range_noise_vectors, is_grd)


def compute_roll_angle_deg_from_sensor_position(sensor_pos: np.ndarray) -> float:
    """Computing roll angle from sensor position.

    Parameters
    ----------
    sensor_pos : np.ndarray
        sensor position

    Returns
    -------
    float
        roll angle [deg]
    """
    return BORESIGHT_ANGLE_REF - ALPHA_ROLL * (xyz2llh(sensor_pos)[2] / 1e3 - SENSOR_HEIGHT_REF)
