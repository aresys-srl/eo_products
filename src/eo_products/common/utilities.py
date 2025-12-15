# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Common Enum, dataclasses and other utilities
--------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
from arepytools.geometry.orbit import Orbit
from arepytools.timing.precisedatetime import PreciseDateTime
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

LookingDirection = Literal["LEFT", "RIGHT"]


class SARRadiometricQuantity(Enum):
    """Enum class for radiometric analysis input/output quantity types"""

    BETA_NOUGHT = auto()
    SIGMA_NOUGHT = auto()
    GAMMA_NOUGHT = auto()


class SARPolarization(Enum):
    """Polarization enum class"""

    HH = "H/H"
    VV = "V/V"
    HV = "H/V"
    VH = "V/H"


class SARProjection(Enum):
    """Enum class for managing swath projection of product folder"""

    SLANT_RANGE = "SLANT RANGE"
    GROUND_RANGE = "GROUND RANGE"


class OrbitDirection(Enum):
    """Orbit direction: ascending or descending"""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class StandardSARAcquisitionMode(Enum):
    """Standard cross-package SAR acquisition mode definition"""

    SCANSAR = auto()
    SPOTLIGHT = auto()
    STRIPMAP = auto()
    TOPSAR = auto()
    WAVE = auto()
    ELEVATION_NOTCH = auto()
    UNKNOWN = auto()


@dataclass
class SARSamplingFrequencies:
    """SAR signal sampling frequencies"""

    range_freq_hz: float
    range_bandwidth_freq_hz: float
    azimuth_freq_hz: float
    azimuth_bandwidth_freq_hz: float


@dataclass
class ConversionFunction:
    """Generic conversion function wrapper"""

    azimuth_reference_time: PreciseDateTime
    origin: float
    function: Polynomial | CubicSpline


@dataclass
class CoordinatesConversions:
    """Coordinates conversion: ground to slant and slant to ground polynomials"""

    ground_to_slant: list[ConversionFunction] | None = None
    slant_to_ground: list[ConversionFunction] | None = None
    azimuth_reference_times: np.ndarray | None = None

    def evaluate_ground_to_slant(self, azimuth_time: PreciseDateTime, ground_range: ArrayLike) -> ArrayLike:
        """Compute ground to slant conversion.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time to select the proper conversion function
        ground_range : ArrayLike
            ground range value(s) in meters

        Returns
        -------
        ArrayLike
            slant range values
        """
        poly_index = detect_right_polynomial_index(
            azimuth_time=azimuth_time, reference_azimuth_times=self.azimuth_reference_times
        )
        poly = self.ground_to_slant[poly_index]
        return poly.function(ground_range - poly.origin)

    def evaluate_slant_to_ground(self, azimuth_time: PreciseDateTime, slant_range: ArrayLike) -> ArrayLike:
        """Compute slant to ground conversion.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time to select the proper conversion function
        slant_range : ArrayLike
            slant range value(s) in meters

        Returns
        -------
        ArrayLike
            ground range values
        """
        poly_index = detect_right_polynomial_index(
            azimuth_time=azimuth_time, reference_azimuth_times=self.azimuth_reference_times
        )
        poly = self.slant_to_ground[poly_index]
        return poly.function(slant_range - poly.origin)


@dataclass
class DopplerEvaluator:
    """Doppler function (rate/centroid) evaluator"""

    functions: list[ConversionFunction] | None = None
    azimuth_reference_times: np.ndarray | None = None

    def evaluate(self, azimuth_time: PreciseDateTime, slant_range: ArrayLike) -> ArrayLike:
        """Evaluate function at given inputs.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time to select the proper function
        slant_range : ArrayLike
            slant range value(s) in meters

        Returns
        -------
        ArrayLike
            Doppler functions values at slant range
        """
        poly_index = detect_right_polynomial_index(
            azimuth_time=azimuth_time, reference_azimuth_times=self.azimuth_reference_times
        )
        poly = self.functions[poly_index]
        return poly.function(slant_range - poly.origin)


@dataclass
class RasterInfo:
    """Product Raster Info"""

    lines: RasterInfoAxis
    samples: RasterInfoAxis
    data_type: str | None = None
    raster_name: str | None = None


@dataclass
class RasterInfoAxis:
    """Axis representation for RasterInfo"""

    length: int
    step: float
    start: float | PreciseDateTime
    step_unit: str
    axis: np.ndarray = field(init=False)

    def __post_init__(self):
        # generating axis array from inputs
        self.axis = np.arange(0, self.length, 1) * self.step + self.start


@dataclass
class BurstInfo:
    """Swath burst info"""

    num: int  # number of bursts in this swath
    lines_per_burst: int | np.ndarray  # number of azimuth lines within each burst, int if constant, else array
    samples_per_burst: int | np.ndarray  # number of range samples within each burst, int if constant, else array
    azimuth_start_times: np.ndarray  # zero doppler azimuth time of the first line of this burst
    range_start_times: np.ndarray  # zero doppler range time of the first sample of this burst


@dataclass
class SwathInfo:
    """Swath Info"""

    swath: str | None
    rank: int
    azimuth_steering_rate_poly: tuple[float, float, float]
    prf: float


@dataclass
class DatasetInfo:
    """Dataset Info"""

    fc_hz: float
    acquisition_mode: str
    sensor_name: str
    image_type: str
    projection: str
    side_looking: LookingDirection


@dataclass
class StateVectors:
    """Orbit's state vectors"""

    num: int  # number of state vectors
    positions: np.ndarray  # platform position data with respect to the Earth-fixed reference frame (ECEF)
    velocities: np.ndarray  # platform velocity data with respect to the Earth-fixed reference frame (ECEF)
    time_axis: np.ndarray  # PreciseDateTime axis at which orbit state vectors apply
    time_step: float  # time axis step
    orbit_direction: OrbitDirection | None = None  # orbit direction: ascending or descending
    orbit_type: Any | None = None  # orbit level type
    reference_frame: Any | None = None  # reference frame
    orbit: Orbit = field(init=False)

    def __post_init__(self):
        """Generating an Orbit trajectory object from state vectors data"""
        assert self.time_axis.size == self.positions.shape[0] == self.velocities.shape[0] == self.num
        self.orbit = Orbit(times=self.time_axis, positions=self.positions, velocities=self.velocities)


@dataclass
class PulseInfo:
    """Chirp pulse info"""

    length_s: float
    bandwidth_hz: float
    sampling_rate_hz: float
    energy_j: float
    start_frequency_hz: float
    start_phase: float
    direction: str


@dataclass
class AcquisitionTimeline:
    """SAR Acquisition Timeline"""

    missing_lines_number: int
    missing_lines_azimuth_times: list[PreciseDateTime] | None
    swst_changes_number: int
    swst_changes_azimuth_times: list[PreciseDateTime]
    swst_changes_values: list[float]
    noise_packets_number: int
    noise_packets_azimuth_times: list[PreciseDateTime] | None
    internal_calibration_number: float
    internal_calibration_azimuth_times: list[PreciseDateTime] | None
    swl_changes_number: int
    swl_changes_azimuth_times: list[PreciseDateTime] | None
    swl_changes_values: list[float]


def detect_right_polynomial_index(azimuth_time: PreciseDateTime, reference_azimuth_times: np.ndarray) -> int:
    """Detecting the index of the right polynomial to be used given an input azimuth time.
    The polynomial to be used is the one with reference azimuth time closest to the input value but with
    reference_azimuth_time < input_azimuth_time.

    Parameters
    ----------
    azimuth_time : PreciseDateTime
        selected azimuth time
    reference_azimuth_times : np.ndarray
        array of reference azimuth times, in PreciseDateTime format

    Returns
    -------
    int
        index corresponding to the polynomial to be used
    """
    diff = np.array(azimuth_time - reference_azimuth_times).astype("float")
    return np.ma.masked_where(diff < 0, diff).argmin()
