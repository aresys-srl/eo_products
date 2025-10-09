# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Synspective StriX reader support module
---------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from arepytools.geometry.direct_geocoding import direct_geocoding_monostatic
from arepytools.geometry.orbit import Orbit
from arepytools.timing.precisedatetime import PreciseDateTime
from numpy.polynomial import Polynomial
from sarpy.io.complex.sicd import SICDReader, SICDType
from scipy.constants import speed_of_light

from eo_products.common.utilities import (
    BurstInfo,
    ConversionFunction,
    CoordinatesConversions,
    DatasetInfo,
    DopplerEvaluator,
    OrbitDirection,
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

_SLC_DATA_EXTENSION = ".nitf"
_GRD_METADATA_EXTENSION = ".xml"
_GRD_METADATA_PREFIX = "PAR"
_GRD_DATA_PREFIX = "IMG"
_GRD_DATA_EXTENSION = ".tif"

_SLC_STATE_VECTORS_NUM = 150

regex_collection = {
    "acquisition_mode": re.compile("(?<=<eop:operationalMode>).*(?=</eop:operationalMode>)"),
    "polarization": re.compile("(?<=<sar:polarisationChannels>).*(?=</sar:polarisationChannels>)"),
    "footprint": re.compile("(?<=<gml:posList>).*(?=</gml:posList>)"),
    "lines": re.compile("(?<=<eop:numberOfLine>).*(?=</eop:numberOfLine>)"),
    "samples": re.compile("(?<=<eop:numberOfPixel>).*(?=</eop:numberOfPixel>)"),
    "samples_step": re.compile("(?<=<sar:rangePixelSpacing>).*(?=</sar:rangePixelSpacing>)"),
    "azimuth_step": re.compile("(?<=<sar:azimuthPixelSpacing>).*(?=</sar:azimuthPixelSpacing>)"),
    "orbit_direction": re.compile("(?<=<eop:orbitDirection>).*(?=</eop:orbitDirection>)"),
    "signal_frequency": re.compile("(?<=<sar:carrierFrequency>).*(?=</sar:carrierFrequency>)"),
    "range_sampling_frequency": re.compile("(?<=<sar:rangeSamplingFrequency>).*(?=</sar:rangeSamplingFrequency>)"),
    "range_bandwidth": re.compile("(?<=<sar:chirpBandWidth>).*(?=</sar:chirpBandWidth>)"),
    "look_side": re.compile("(?<=<sar:antennaLookDirection>).*(?=</sar:antennaLookDirection>)"),
    "sensor": re.compile("(?<=<eop:shortName>).*(?=</eop:shortName>)"),
    "sensor_id": re.compile("(?<=<eop:serialIdentifier>).*(?=</eop:serialIdentifier>)"),
    "sv_num": re.compile("(?<=<numStateVectors>).*(?=</numStateVectors>)"),
    "sv_time_utc": re.compile("(?<=<timeUTC>).*(?=</timeUTC>)"),
    "sv_pos_x": re.compile("(?<=<posX>).*(?=</posX>)"),
    "sv_pos_y": re.compile("(?<=<posY>).*(?=</posY>)"),
    "sv_pos_z": re.compile("(?<=<posZ>).*(?=</posZ>)"),
    "sv_vel_x": re.compile("(?<=<velX>).*(?=</velX>)"),
    "sv_vel_y": re.compile("(?<=<velY>).*(?=</velY>)"),
    "sv_vel_z": re.compile("(?<=<velZ>).*(?=</velZ>)"),
    "local_value": re.compile("(?<=<eop:localValue>).*(?=</eop:localValue>)"),
    "calibration_factor": re.compile("<eop:localAttribute>calibrationFactor</eop:localAttribute>"),
    "acquisition_mid_center_time": re.compile("<eop:localAttribute>sceneCenterDateTime</eop:localAttribute>"),
}


class InvalidStriXProduct(RuntimeError):
    """Invalid StriX product"""


class StriXProcessingLevel(Enum):
    """StriX L1 processing level"""

    SLC = "SLC"  # Slant Range, Single Look Complex (SLC, lvl 1)
    GRD = "GRD"  # Ground Range Multi Look Detected (GRD, lvl 1, phase lost)


class StriXOperationalModes(Enum):
    """StriX Acquisition Modes"""

    STRIPMAP = "STRIPMAP"
    SLIDING_SPOTLIGHT = "DYNAMIC STRIPMAP"
    STARING_SPOTLIGHT = "SPOTLIGHT"


def _find_grd_metadata(product_path: Path) -> Path | None:
    """Find valid GRD StriX channel metadata for input GeoTiff product.

    Parameters
    ----------
    product_path : Path
        Path to the StriX GRD GeoTiff product

    Returns
    -------
    Path | None
        Path to the GRD StriX channel metadata file, if any else None
    """
    assert product_path.name.endswith(_GRD_DATA_EXTENSION)
    metadata_name = product_path.name.replace(_GRD_DATA_PREFIX, _GRD_METADATA_PREFIX).replace(
        _GRD_DATA_EXTENSION, _GRD_METADATA_EXTENSION
    )
    grd_metadata = product_path.parent.joinpath(metadata_name)
    if not grd_metadata.is_file():
        # GRD metadata .xml file not found
        return None
    return grd_metadata


def _compose_channels_names(polarizations: list[SARPolarization], beams: list[str]) -> list[str]:
    """Composing channel names from polarization and beam.

    Parameters
    ----------
    polarizations : list[SARPolarization]
        channel polarizations
    beams : list[str]
        channel beams

    Returns
    -------
    list[str]
        channel names as "beam_polarization"
    """
    # TODO: check this
    return [f"{b}_{p.name.lower()}" for p in polarizations for b in beams]


def _get_azimuth_time_axis_elements(
    root: str | SICDType, orbit: Orbit | None = None
) -> tuple[PreciseDateTime, float, int]:
    """Getting azimuth time axis elements from metadata file.

    Parameters
    ----------
    root : str | SICDType
        metadata root object, string file content or SICDType object
    orbit : Orbit | None, optional
        sensor orbit, needed for GRD only, by default None

    Returns
    -------
    PreciseDateTime
        azimuth axis start time
    float
        azimuth axis step
    int
        number of azimuth lines
    """
    _, lines_start, lines_mid, lines_stop, _, _ = get_basic_info_from_metadata(root=root)
    if isinstance(root, SICDType):
        lines = root.to_dict()["ImageData"]["NumCols"]
        lines_step = (lines_stop - lines_start) / (lines - 1)
    else:
        lines = int(regex_collection["lines"].findall(root)[0])
        lines_step_m = float(regex_collection["azimuth_step"].findall(root)[0])
        mid_swath_sat_velocity = np.linalg.norm(orbit.evaluate_first_derivatives(lines_mid))
        lines_step = lines_step_m / mid_swath_sat_velocity
        lines_start = lines_mid - lines / 2 * lines_step

    return lines_start, lines_step, lines


def _subsampling_array(array: np.ndarray, num_samples: int) -> np.ndarray:
    """Subsampling array always being sure to get an equally spaced array with first and last element included.

    Parameters
    ----------
    array : np.ndarray
        array to be subsampled
    num_samples : int
        number of final samples

    Returns
    -------
    np.ndarray
        subsampled array
    """
    original_indices = np.linspace(0, array.size - 1, len(array))
    target_indices = np.linspace(0, array.size - 1, num_samples)

    return np.interp(target_indices, original_indices, array)


def get_basic_info_from_metadata(
    root: str | SICDType,
) -> tuple[
    StriXOperationalModes,
    PreciseDateTime,
    PreciseDateTime,
    PreciseDateTime,
    SARPolarization,
    tuple[float, float, float, float],
]:
    """Get the product acquisition mode, acquisition start time, polarization and footprint from metadata file.

    Parameters
    ----------
    root : str | SICDType
        metadata root object, string file content or SICDType object

    Returns
    -------
    StriXOperationalModes
        Strix acquisition mode
    PreciseDateTime
        acquisition start time
    PreciseDateTime
        acquisition mid center time
    PreciseDateTime
        acquisition stop time
    SARPolarization
        product polarization
    tuple[float, float, float, float]
        product footprint
    """
    if isinstance(root, SICDType):
        mtd = root.to_dict()
        latitudes = [c["Lat"] for c in mtd["GeoData"]["ImageCorners"]]
        longitudes = [c["Lon"] for c in mtd["GeoData"]["ImageCorners"]]
        polarization = SARPolarization[mtd["ImageFormation"]["TxRcvPolarizationProc"].replace(":", "").upper()]
        acq_start_time = PreciseDateTime.fromisoformat(mtd["Timeline"]["CollectStart"])
        acq_stop_time = acq_start_time + mtd["Timeline"]["CollectDuration"]
        acq_mode = StriXOperationalModes(mtd["CollectionInfo"]["RadarMode"]["ModeType"])
        acq_mid_center_time = (acq_stop_time - acq_start_time) / 2 + acq_start_time
    else:
        mtd_lines = root.splitlines()
        acq_mode = StriXOperationalModes(regex_collection["acquisition_mode"].findall(root)[0].upper())
        for line_id, line in enumerate(mtd_lines):
            if regex_collection["acquisition_mid_center_time"].pattern in line:
                acq_mid_center_time = PreciseDateTime.fromisoformat(
                    regex_collection["local_value"].search(mtd_lines[line_id + 1]).group()
                )
                break
        acq_start_time, acq_stop_time = None, None
        polarization = SARPolarization[regex_collection["polarization"].findall(root)[0]]
        footprint = tuple(float(f) for f in regex_collection["footprint"].findall(root)[0].split())
        latitudes, longitudes = footprint[0::2], footprint[1::2]
    return (
        acq_mode,
        acq_start_time,
        acq_mid_center_time,
        acq_stop_time,
        polarization,
        (min(latitudes), max(latitudes), min(longitudes), max(longitudes)),
    )


def raster_info_from_metadata(root: str | SICDType, orbit: Orbit) -> RasterInfo:
    """Creating a RasterInfo metadata object from metadata file.

    Parameters
    ----------
    root : str | SICDType
        metadata root object, string file content or SICDType object
    orbit : Orbit
        sensor orbit

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """
    # lines
    lines_start, lines_step, lines = _get_azimuth_time_axis_elements(root=root, orbit=orbit)
    lines_step_unit = "s"

    if isinstance(root, SICDType):
        # assuming SLC only
        mtd = root.to_dict()
        # samples
        assert mtd["Grid"]["ImagePlane"] == "SLANT"
        samples = mtd["ImageData"]["NumRows"]
        samples_step = mtd["Grid"]["Row"]["SS"] / (speed_of_light / 2)
        samples_step_unit = "s"
        samples_start = (
            mtd["SCPCOA"]["SlantRange"] / (speed_of_light / 2) - mtd["ImageData"]["SCPPixel"]["Row"] * samples_step
        )
        celltype = "FLOAT_COMPLEX"

    else:
        # assuming GRD only
        # samples
        samples = int(regex_collection["samples"].findall(root)[0])
        samples_start = 0
        samples_step = float(regex_collection["samples_step"].findall(root)[0])
        samples_step_unit = "m"
        celltype = "FLOAT32"

    raster_lines = RasterInfoAxis(length=lines, start=lines_start, step=lines_step, step_unit=lines_step_unit)
    raster_samples = RasterInfoAxis(length=samples, start=samples_start, step=samples_step, step_unit=samples_step_unit)

    return RasterInfo(lines=raster_lines, samples=raster_samples, data_type=celltype)


def burst_info_from_metadata(raster_info: RasterInfo) -> BurstInfo:
    """Generating BurstInfo object directly from metadata.

    Parameters
    ----------
    raster_info : RasterInfo
        product raster info

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
        range_start_times=np.array([raster_info.samples.start]),
    )


def dataset_info_from_metadata(root: str | SICDType) -> DatasetInfo:
    """Creating a DatasetInfo metadata object from metadata file.

    Parameters
    ----------
    root : str | SICDType
        metadata root object, string file content or SICDType object

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """
    if isinstance(root, SICDType):
        # assuming SLC only
        mtd = root.to_dict()
        fc_hz = mtd["RMA"]["INCA"]["FreqZero"]
        sensor_name = mtd["CollectionInfo"]["CollectorName"]
        acquisition_mode = mtd["CollectionInfo"]["RadarMode"]["ModeType"].capitalize()
        look_side = "LEFT" if mtd["SCPCOA"]["SideOfTrack"] == "L" else "RIGHT"
        projection = "SLANT RANGE"
        image_type = "AZIMUTH FOCUSED RANGE COMPENSATED"
    else:
        # assuming GRD only
        fc_hz = float(regex_collection["signal_frequency"].findall(root)[0])
        sensor_name = f"{regex_collection['sensor'].findall(root)[0]}-{regex_collection['sensor_id'].findall(root)[0]}"
        acquisition_mode = regex_collection["acquisition_mode"].findall(root)[0].upper()
        look_side = regex_collection["look_side"].findall(root)[0].upper()
        projection = "GROUND RANGE"
        image_type = "MULTILOOK"

    return DatasetInfo(
        fc_hz=fc_hz,
        acquisition_mode=acquisition_mode,
        image_type=image_type,
        projection=projection,
        sensor_name=sensor_name,
        side_looking=look_side,
    )


def state_vectors_from_metadata(root: str | SICDType) -> StateVectors:
    """Generating StateVectors object directly from product metadata.

    Parameters
    ----------
    root : str | SICDType
        metadata root object, string file content or SICDType object

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """
    if isinstance(root, SICDType):
        lines_start, lines_step, lines = _get_azimuth_time_axis_elements(root=root)
        azimuth_relative_axis = np.arange(lines) * lines_step
        time_axis = _subsampling_array(array=azimuth_relative_axis, num_samples=_SLC_STATE_VECTORS_NUM)
        # assuming SLC only
        mtd = root.to_dict()
        # orbit is in ARP (Aperture Reference Position) polynomial
        arp_poly_x = Polynomial(mtd["Position"]["ARPPoly"]["X"]["Coefs"])
        arp_poly_y = Polynomial(mtd["Position"]["ARPPoly"]["Y"]["Coefs"])
        arp_poly_z = Polynomial(mtd["Position"]["ARPPoly"]["Z"]["Coefs"])
        positions = np.stack(
            [
                arp_poly_x(time_axis),
                arp_poly_y(time_axis),
                arp_poly_z(time_axis),
            ],
            axis=1,
        )
        velocities = np.stack(
            [
                arp_poly_x.deriv(1)(time_axis),
                arp_poly_y.deriv(1)(time_axis),
                arp_poly_z.deriv(1)(time_axis),
            ],
            axis=1,
        )
        numerosity = time_axis.size
        time_axis = time_axis + lines_start
        orbit_direction = OrbitDirection.ASCENDING if velocities[0, 2] > 0 else OrbitDirection.DESCENDING

    else:
        # assuming GRD only
        numerosity = int(regex_collection["sv_num"].findall(root)[0])
        orbit_direction = OrbitDirection(regex_collection["orbit_direction"].findall(root)[0].lower())
        time_axis = np.array([PreciseDateTime.fromisoformat(t) for t in regex_collection["sv_time_utc"].findall(root)])
        positions = np.stack(
            [
                [float(p) for p in regex_collection["sv_pos_x"].findall(root)],
                [float(p) for p in regex_collection["sv_pos_y"].findall(root)],
                [float(p) for p in regex_collection["sv_pos_z"].findall(root)],
            ],
            axis=1,
        )
        velocities = np.stack(
            [
                [float(p) for p in regex_collection["sv_vel_x"].findall(root)],
                [float(p) for p in regex_collection["sv_vel_y"].findall(root)],
                [float(p) for p in regex_collection["sv_vel_z"].findall(root)],
            ],
            axis=1,
        )

    assert positions.shape[0] == velocities.shape[0] == time_axis.size == numerosity
    return StateVectors(
        num=numerosity,
        positions=positions,
        velocities=velocities,
        time_axis=time_axis,
        time_step=time_axis[1] - time_axis[0],
        orbit_direction=orbit_direction,
    )


def sampling_constants_from_metadata(root: str | SICDType, lines_step: float) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata object from metadata file.

    Parameters
    ----------
    root : str | SICDType
        metadata root object, string file content or SICDType object
    lines_step : float
        azimuth lines step

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    if isinstance(root, SICDType):
        mtd = root.to_dict()
        range_freq_hz = mtd["RadarCollection"]["Waveform"][0]["ADCSampleRate"]
        range_bandwidth_freq_hz = mtd["RadarCollection"]["Waveform"][0]["TxRFBandwidth"]
    else:
        range_freq_hz = float(regex_collection["range_sampling_frequency"].findall(root)[0])
        range_bandwidth_freq_hz = float(regex_collection["range_bandwidth"].findall(root)[0])

    prf = 1 / lines_step
    azimuth_freq_hz = prf
    azimuth_bandwidth_freq_hz = prf

    return SARSamplingFrequencies(
        azimuth_freq_hz=azimuth_freq_hz,
        azimuth_bandwidth_freq_hz=azimuth_bandwidth_freq_hz,
        range_freq_hz=range_freq_hz,
        range_bandwidth_freq_hz=range_bandwidth_freq_hz,
    )


def doppler_centroid_poly_from_metadata_node(root: str | SICDType, raster_info: RasterInfo) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    root: str | SICDType
        metadata root object
    raster_info : RasterInfo
        channel raster info

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    # TODO: forcing zero here for now, but should be read from metadata
    doppler_poly_list = [
        ConversionFunction(
            azimuth_reference_time=raster_info.lines.start,
            origin=raster_info.samples.start,
            function=Polynomial([0.01]),
        )
    ]

    return DopplerEvaluator(functions=doppler_poly_list, azimuth_reference_times=np.array([raster_info.lines.start]))


def doppler_rate_poly_from_metadata(root: str | SICDType, raster_info: RasterInfo) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler rate vector polynomial wrapper from metadata.

    Parameters
    ----------
    root: str | SICDType
        metadata root object
    raster_info : RasterInfo
        channel raster info

    Returns
    -------
    DopplerEvaluator | None
        DopplerEvaluator dataclass for Doppler Rate polynomial, if any else None
    """
    if isinstance(root, SICDType):
        mtd = root.to_dict()
        coeff = mtd["RMA"]["INCA"]["DRateSFPoly"]["Coefs"][0]
        doppler_rate_poly = [
            ConversionFunction(
                azimuth_reference_time=raster_info.lines.start,
                origin=raster_info.samples.start,
                function=Polynomial(coeff),
            )
        ]
        return DopplerEvaluator(
            functions=doppler_rate_poly, azimuth_reference_times=np.array([raster_info.lines.start])
        )
    return None


def get_calibration_factor_and_quantity_from_metadata(root: str | SICDType) -> float:
    """Get the calibration factor and radiometric quantity from metadata file.

    Parameters
    ----------
    root : str | SICDType
        metadata root object, string file content or SICDType object

    Returns
    -------
    float
        calibration factor
    SARRadiometricQuantity
        radiometric quantity
    """
    if isinstance(root, SICDType):
        mtd = root.to_dict()
        assert mtd["Radiometric"]["BetaZeroSFPoly"]["Coefs"] == [[1.0]]
        cal_factor = 1
        rad_quantity = SARRadiometricQuantity.BETA_NOUGHT
    else:
        cal_factor = 1
        mtd_lines = root.splitlines()
        for line_id, line in enumerate(mtd_lines):
            if regex_collection["calibration_factor"].pattern in line:
                cal_factor = float(regex_collection["local_value"].search(mtd_lines[line_id + 1]).group())
                break
        cal_factor = float(1 / np.sqrt(cal_factor))
        rad_quantity = SARRadiometricQuantity.SIGMA_NOUGHT
    return cal_factor, rad_quantity


def coordinates_conversions_from_metadata(
    root: str | SICDType, raster_info: RasterInfo, orbit: Orbit
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
    root : str | SICDType
        metadata root object, string file content or SICDType object
    raster_info : RasterInfo
        product raster info
    orbit : Orbit
        sensor orbit

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass
    """
    # TODO: this is a temporary solution, should be read from metadata
    if isinstance(root, SICDType):
        look_side = "LEFT" if root.to_dict()["SCPCOA"]["SideOfTrack"] == "L" else "RIGHT"
    else:
        look_side = regex_collection["look_side"].findall(root)[0].upper()
    mid_azimuth = raster_info.lines.start + raster_info.lines.length * raster_info.lines.step / 2
    range_times = np.arange(0, raster_info.samples.length, 1) * raster_info.samples.step + raster_info.samples.start
    ground_points = direct_geocoding_monostatic(
        sensor_positions=orbit.evaluate(mid_azimuth),
        sensor_velocities=orbit.evaluate_first_derivatives(mid_azimuth),
        range_times=range_times,
        frequencies_doppler_centroid=0,
        wavelength=1,
        geocoding_side=look_side,
        geodetic_altitude=0,
    )
    ground_points_distances = np.linalg.norm(np.diff(ground_points, axis=0), axis=1)
    ground_range_axis = np.r_[[0], np.cumsum(ground_points_distances)]

    ground_to_slant_poly_list = [
        ConversionFunction(
            azimuth_reference_time=raster_info.lines.start,
            origin=0,
            function=Polynomial.fit(x=ground_range_axis, y=range_times, deg=8),
        )
    ]
    slant_to_ground_poly_list = [
        ConversionFunction(
            azimuth_reference_time=raster_info.lines.start,
            origin=0,
            function=Polynomial.fit(x=range_times, y=ground_range_axis, deg=8),
        )
    ]

    return CoordinatesConversions(
        azimuth_reference_times=raster_info.lines.start,
        ground_to_slant=ground_to_slant_poly_list,
        slant_to_ground=slant_to_ground_poly_list,
    )


@dataclass
class StriXGeneralChannelInfo:
    """StriX general channel info dataclass"""

    product_name: str
    channel_id: str
    swath: str
    processing_level: StriXProcessingLevel
    polarization: SARPolarization
    projection: SARProjection
    acquisition_mode: StriXOperationalModes
    acquisition_mode_std: StandardSARAcquisitionMode
    signal_frequency: float
    acq_start_time: PreciseDateTime
    acq_stop_time: PreciseDateTime
    orbit_direction: OrbitDirection | None = None

    @classmethod
    def from_metadata(
        cls, root: str | SICDType, orbit: Orbit, product_name: str, channel_id: str
    ) -> StriXGeneralChannelInfo:
        """Generating StriXGeneralChannelInfo object directly from metadata.

        Parameters
        ----------
        root : str | SICDType
            xml metadata file content as string or SICDType object
        orbit : Orbit
            sensor orbit
        product_name : str
            product name
        channel_id : str
            channel id

        Returns
        -------
        StriXGeneralChannelInfo
            general channel info dataclass
        """
        acq_mode, acq_start_time, _, acq_stop_time, polarization, _ = get_basic_info_from_metadata(root=root)
        if isinstance(root, SICDType):
            mtd = root.to_dict()
            carrier_frequency = mtd["RMA"]["INCA"]["FreqZero"]
        else:
            carrier_frequency = float(regex_collection["signal_frequency"].findall(root)[0])

        match acq_mode:
            case StriXOperationalModes.STRIPMAP:
                acquisition_mode_std = StandardSARAcquisitionMode.STRIPMAP
            case StriXOperationalModes.SLIDING_SPOTLIGHT:
                acquisition_mode_std = StandardSARAcquisitionMode.SPOTLIGHT
            case StriXOperationalModes.STARING_SPOTLIGHT:
                acquisition_mode_std = StandardSARAcquisitionMode.SPOTLIGHT
        return cls(
            product_name=product_name,
            channel_id=channel_id,
            swath=channel_id.split("_")[0],
            processing_level=StriXProcessingLevel.SLC if isinstance(root, SICDType) else StriXProcessingLevel.GRD,
            polarization=polarization,
            projection=SARProjection.SLANT_RANGE if isinstance(root, SICDType) else SARProjection.GROUND_RANGE,
            acquisition_mode=acq_mode,
            acquisition_mode_std=acquisition_mode_std,
            signal_frequency=carrier_frequency,
            acq_start_time=acq_start_time,
            acq_stop_time=acq_stop_time,
        )


@dataclass
class StriXChannelMetadata:
    """StriX channel metadata dataclass"""

    general_info: StriXGeneralChannelInfo
    orbit: Orbit
    image_calibration_factor: float
    image_radiometric_quantity: SARRadiometricQuantity
    burst_info: BurstInfo
    raster_info: RasterInfo
    dataset_info: DatasetInfo
    swath_info: SwathInfo
    sampling_constants: SARSamplingFrequencies
    doppler_centroid_poly: DopplerEvaluator | None
    doppler_rate_poly: DopplerEvaluator | None
    coordinate_conversions: CoordinatesConversions
    state_vectors: StateVectors


class StriXProduct:
    """StriX product object"""

    def __init__(self, path: str | Path) -> None:
        """StriX Product init from directory path.

        Parameters
        ----------
        path : str | Path
            path to StriX product
        """
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        self._is_grd = self._product_name.endswith(_GRD_DATA_EXTENSION)

        if self._is_grd:
            self._metadata_file = _find_grd_metadata(product_path=self._product_path)
            mtd = self._metadata_file.read_text(encoding="UTF-8")
        else:
            root = SICDReader(str(path))
            mtd = root.sicd_meta
            assert root.sicd_meta.to_dict()["ImageFormation"]["RcvChanProc"]["ChanIndices"] == [1]
            root.close()

        _, acq_start_time, _, _, polarization, footprint = get_basic_info_from_metadata(root=mtd)
        self._channels_list = _compose_channels_names(polarizations=[polarization], beams=["s"])

        self._acq_start_time = acq_start_time
        self._footprint = footprint

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_start_time

    @property
    def channels_number(self) -> int:
        """Returning the number of channels of StriX product"""
        return len(self._channels_list)

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels"""
        return self._channels_list

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_files_from_channel_name(self, channel_name: str) -> list[Path] | Path:
        """Get files associated to a given channel name.

        GRD channels will return a list of two paths: .xml metadata file, .tif GeoTiff raster file

        SLC channels will return a list of a single path: .nitf NITF file

        Parameters
        ----------
        channel_name : str
            channel id name

        Returns
        -------
        list[Path] | Path
            path to grd metadata and raster file, or path to slc file
        """
        if self._is_grd:
            return [self._metadata_file, self._product_path]
        return self._product_path


def is_strix_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid StriX product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a file
        - metadata can be found (GRD)
        - metadata can be read (GRD, SLC)

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

    if not product.exists() or not product.is_file():
        return False

    try:
        if product.name.endswith(_SLC_DATA_EXTENSION):
            # check SLC NITF
            root = SICDReader(str(product))
            mtd = root.sicd_meta
            root.close()
        elif product.name.endswith(_GRD_DATA_EXTENSION):
            # check GRD GeoTiff + .xml metadata
            mtd_file = _find_grd_metadata(product_path=product)
            if mtd_file is None:
                return False
            mtd = mtd_file.read_text(encoding="UTF-8")
        get_basic_info_from_metadata(root=mtd)
    except Exception:
        return False

    return True
