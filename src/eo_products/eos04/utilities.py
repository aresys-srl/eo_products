# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""EOS04 reader support module."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path

import numpy as np
from lxml import etree
from numpy.polynomial.polynomial import Polynomial
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

RASTER_EXTENSION = ".tif"
METADATA_EXTENSION = ".xml"


class InvalidEOS04Product(RuntimeError):
    """Invalid EOS04 Product"""


class EOS04TimeOrdering(Enum):
    """EOS04 available Time Ordering"""

    INCREASING = auto()
    DECREASING = auto()


class EOS04AcquisitionMode(Enum):
    """EOS04 Acquisition Modes"""

    SCANSAR = "SCANSAR"


class EOS04ProductType(Enum):
    """EOS04 Product Types"""

    SLC = "SLC"
    GRD = "GROUND RANGE"


def _parse_timestamp(timestamp: float) -> PreciseDateTime:
    """Parsing UTC timestamp referred to 01/01/1970.

    Parameters
    ----------
    timestamp : float
        float timestamp

    Returns
    -------
    PreciseDateTime
        PreciseDateTime date format
    """
    return PreciseDateTime.fromisoformat(
        datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None).isoformat()
    )


def _get_basic_info_from_metadata(
    metadata_path: Path,
) -> tuple[PreciseDateTime, list[int], list[SARPolarization], EOS04ProductType]:
    """Get the product acquisition time, polarizations and beams from metadata file.

    Parameters
    ----------
    metadata_path : Path
        path to the EOS04 metadata file

    Returns
    -------
    PreciseDateTime
        acquisition time
    list[int]
        list of beams
    list[SARPolarization]
        list of channels polarizations
    EOS04ProductType
        Product type
    """
    mtd = metadata_path.read_text(encoding="UTF-8")
    acq_time_re = re.compile("(?<=<StartTime>).*(?=</StartTime>)")
    beams_re = re.compile("(?<=<BeamID>).*(?=</BeamID>)")
    polarizations_re = re.compile("(?<=<Polarizations>).*(?=</Polarizations>)")
    product_type_re = re.compile("(?<=<ProductType>).*(?=</ProductType>)")
    acquisition_time = acq_time_re.findall(mtd)[0]
    product_type = EOS04ProductType(product_type_re.findall(mtd)[0])
    beams = list(range(0, len(beams_re.findall(mtd)[0].split())))
    polarizations = [SARPolarization[p.upper()] for p in polarizations_re.findall(mtd)[0].split()]
    return PreciseDateTime.fromisoformat(acquisition_time), beams, polarizations, product_type


def _retrieve_scene_footprint(metadata_path: Path) -> tuple[float, float, float, float]:
    """Product footprint as tuple of (min lat, max lat, min lon, max lon).

    Parameters
    ----------
    metadata_path : Path
        Path to the metadata .xml file

    Returns
    -------
    tuple[float, float, float, float]
        (min lat, max lat, min lon, max lon)
    """
    mtd = metadata_path.read_text(encoding="UTF-8")
    footprint_re = re.compile('(?<=<SourceDataGeometry type="WKT">).*(?=</SourceDataGeometry>)')
    footprint = footprint_re.findall(mtd)[0]
    footprint = [float(f) for f in footprint.replace("Polygon", "").strip("() ").replace(" ", ",").split(",")]
    longitudes, latitudes = footprint[0::2], footprint[1::2]
    return (min(latitudes), max(latitudes), min(longitudes), max(longitudes))


def compose_channel_name(polarization: SARPolarization, beam: int) -> str:
    """Composing channel name from polarization and beam.

    Parameters
    ----------
    polarization : SARPolarization
        channel polarization
    beam : int
        channel beam

    Returns
    -------
    str
        channel name, as B{beam}_POL
    """
    return "_".join([f"B{str(beam)}", polarization.name])


def unpack_channel_name(channel_name: str) -> tuple[int, SARPolarization]:
    """Recovering beam id and polarization value from channel name.

    Parameters
    ----------
    channel_name : str
        channel name string

    Returns
    -------
    int
        channel beam id
    SARPolarization
        channel polarization
    """
    beam_pol = channel_name.split("_")
    return int(beam_pol[0].strip("B")), SARPolarization[beam_pol[1]]


def raster_info_from_metadata_nodes(
    image_generation_parameters_node: etree._Element,
    image_attributes_node: etree._Element,
    beam_id: int,
    polarization: SARPolarization,
    product_type: EOS04ProductType,
) -> RasterInfo:
    """Creating a RasterInfo metadata element from xml node.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters metadata xml node
    image_attributes_node : etree._Element
        ImageAttributes metadata xml node
    beam_id : int
        swath beam id
    product_type : EOS04ProductType
        product type

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """

    # swath timing for this channel
    if product_type == EOS04ProductType.SLC:
        swath_timing_node = [
            s
            for s in image_generation_parameters_node.findall("swathTiming")
            if s.find("swath").get("pol") == polarization.name
        ][beam_id]
        bursts_num = int(swath_timing_node.find("burstList").get("count"))
        bursts = swath_timing_node.findall("burstList/burst")

        # azimuth
        lines = int(swath_timing_node.find("linesPerBurst").text) * bursts_num
        lines_step = 1 / float(swath_timing_node.find("swathPRF").text)
        lines_start = _parse_timestamp(timestamp=float(bursts[0].find("firstValidLineTime").text))
        lines_step_unit = "s"

        # samples
        samples = int(swath_timing_node.find("samplesPerBurst").text)
        samples_step = 2 / speed_of_light * float(swath_timing_node.find("swathRangeSampling").text)
        samples_start = 2 / speed_of_light * float(bursts[0].find("firstSampleRange").text)
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"
    else:
        # azimuth
        lines = int(image_attributes_node.find("RasterAttributes/NumberOfLines").text)
        lines_start = PreciseDateTime.fromisoformat(
            image_generation_parameters_node.find("SarProcessingInformation/ZeroDopplerTimeFirstLine").text
        )
        lines_stop = PreciseDateTime.fromisoformat(
            image_generation_parameters_node.find("SarProcessingInformation/ZeroDopplerTimeLastLine").text
        )
        lines_step = (lines_stop - lines_start) / (lines - 1)
        lines_step_unit = "s"

        # ground range
        samples = int(image_attributes_node.find("RasterAttributes/NumberOfSamplesPerLine").text)
        samples_start = 0
        samples_step = float(image_generation_parameters_node.find("SarProcessingInformation/RangePixelSpacing").text)
        samples_step_unit = "m"
        celltype = "FLOAT32"

    raster_lines = RasterInfoAxis(length=lines, start=lines_start, step=lines_step, step_unit=lines_step_unit)
    raster_samples = RasterInfoAxis(length=samples, start=samples_start, step=samples_step, step_unit=samples_step_unit)

    return RasterInfo(
        lines=raster_lines,
        samples=raster_samples,
        data_type=celltype,
    )


def burst_info_from_metadata(
    image_generation_parameters_node: etree._Element,
    polarization: SARPolarization,
    beam_id: int,
    raster_info: RasterInfo,
    product_type: EOS04ProductType,
) -> BurstInfo:
    """Generating BurstInfo object directly from metadata.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters xml node
    polarization : SARPolarization
        product acquisition mode
    beam_id : int
        channel beam id
    raster_info : RasterInfo
        product RasterInfo
    product_type : EOS04ProductType
        product type

    Returns
    -------
    BurstInfo
        burst info dataclass
    """
    if product_type == EOS04ProductType.SLC:
        swath_timing_node = [
            s
            for s in image_generation_parameters_node.findall("swathTiming")
            if s.find("swath").get("pol") == polarization.name
        ][beam_id]
        range_start_times = [
            2 / speed_of_light * float(s.text) for s in swath_timing_node.findall("burstList/burst/firstSampleRange")
        ]
        azimuth_start_times = [
            _parse_timestamp(float(s.text)) for s in swath_timing_node.findall("burstList/burst/firstValidLineTime")
        ]
        lines = int(swath_timing_node.find("linesPerBurst").text)
        samples = int(swath_timing_node.find("samplesPerBurst").text)
        return BurstInfo(
            num=len(range_start_times),
            lines_per_burst=lines,
            samples_per_burst=samples,
            azimuth_start_times=np.array(azimuth_start_times),
            range_start_times=np.array(range_start_times),
        )

    return BurstInfo(
        num=1,
        lines_per_burst=raster_info.lines.length,
        samples_per_burst=raster_info.samples.length,
        azimuth_start_times=np.array([raster_info.lines.start]),
        range_start_times=np.array([raster_info.samples.start]),
    )


def dataset_info_from_metadata_node(source_attributes_node: etree._Element, projection: SARProjection) -> DatasetInfo:
    """Creating a DatasetInfo metadata element from safe xml nodes.

    Parameters
    ----------
    source_attributes_node : etree._Element
        SourceAttributes metadata xml node
    projection : SARProjection
        product projection

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """

    return DatasetInfo(
        fc_hz=float(source_attributes_node.find("SourceDataAcquisitionParameters/RadarCenterFrequency").text),
        acquisition_mode=EOS04AcquisitionMode(
            source_attributes_node.find("SourceDataAcquisitionParameters/ObservationMode").text
        ).value,
        image_type="MULTILOOK" if projection == SARProjection.GROUND_RANGE else "AZIMUTH FOCUSED RANGE COMPENSATED",
        projection=projection.value,
        sensor_name=source_attributes_node.find("Satellite").text,
        side_looking=source_attributes_node.find("SourceDataAcquisitionParameters/AntennaPointing").text.upper(),
    )


def swath_info_from_metadata(
    image_generation_parameters_node: etree._Element,
    polarization: SARPolarization,
    beam: int,
    product_type: EOS04ProductType,
) -> SwathInfo:
    """Creating a SwathInfo metadata object from metadata file.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters xml node
    polarization : SARPolarization
        product acquisition mode
    beam : int
        channel beam id
    product_type : EOS04ProductType
        product type

    Returns
    -------
    SwathInfo
        SwathInfo metadata object
    """
    prf = 0  # TODO: check this for GRD
    if product_type == EOS04ProductType.SLC:
        swath_timing_node = [
            s
            for s in image_generation_parameters_node.findall("swathTiming")
            if s.find("swath").get("pol") == polarization.name
        ][beam]
        prf = float(swath_timing_node.find("swathPRF").text)

    return SwathInfo(
        swath=f"B{str(beam)}",
        rank=0,
        azimuth_steering_rate_poly=(0, 0, 0),
        prf=prf,
    )


def state_vectors_from_metadata(orbit_information_node: etree._Element) -> StateVectors:
    """Generating StateVectors object directly from product metadata.

    Parameters
    ----------
    orbit_information_node : etree._Element
        OrbitInformation xml node

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """
    pos_x = np.array([float(p.text) for p in orbit_information_node.findall("StateVectorECEF/xPosition")])
    pos_y = np.array([float(p.text) for p in orbit_information_node.findall("StateVectorECEF/yPosition")])
    pos_z = np.array([float(p.text) for p in orbit_information_node.findall("StateVectorECEF/zPosition")])
    positions = np.stack([pos_x, pos_y, pos_z], axis=1)

    vel_x = np.array([float(p.text) for p in orbit_information_node.findall("StateVectorECEF/xVelocity")])
    vel_y = np.array([float(p.text) for p in orbit_information_node.findall("StateVectorECEF/yVelocity")])
    vel_z = np.array([float(p.text) for p in orbit_information_node.findall("StateVectorECEF/zVelocity")])
    velocities = np.stack([vel_x, vel_y, vel_z], axis=1)

    time_axis = np.array(
        [PreciseDateTime.fromisoformat(p.text) for p in orbit_information_node.findall("StateVectorECEF/TimeStamp")]
    )

    numerosity = time_axis.size
    assert positions.shape == velocities.shape == (numerosity, 3)

    mean_delta_time = np.diff(time_axis).mean()

    return StateVectors(
        num=numerosity,
        positions=positions,
        velocities=velocities,
        time_axis=time_axis,
        time_step=mean_delta_time,
        orbit_direction=OrbitDirection[orbit_information_node.find("PassDirection").text],
    )


def sampling_constants_from_metadata(
    image_generation_parameters_node: etree._Element, raster_info: RasterInfo
) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata object from metadata file.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters xml node
    raster_info : RasterInfo
        product raster info

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    return SARSamplingFrequencies(
        azimuth_bandwidth_freq_hz=float(
            image_generation_parameters_node.find("SarProcessingInformation/TotalProcessedAzimuthBandwidth").text
        ),
        azimuth_freq_hz=1 / raster_info.lines.step,
        range_bandwidth_freq_hz=float(
            image_generation_parameters_node.find("SarProcessingInformation/TotalProcessedRangeBandwidth").text
        ),
        range_freq_hz=1 / raster_info.samples.step,
    )


def pulse_info_from_metadata_nodes(source_attributes_node: etree._Element, samples_step: float) -> PulseInfo:
    """Creating a PulseInfo dataclass from xml nodes.

    Parameters
    ----------
    source_attributes_node : etree._Element
        SourceAttributes metadata xml node
    samples_step : float
        raster info samples step

    Returns
    -------
    PulseInfo
        PulseInfo dataclass
    """
    pulse_bandwidth = float(source_attributes_node.find("SourceDataAcquisitionParameters/PulseBandwidth").text)

    return PulseInfo(
        length_s=float(source_attributes_node.find("SourceDataAcquisitionParameters/PulseLength").text),
        bandwidth_hz=pulse_bandwidth,
        energy_j=1,
        start_frequency_hz=-pulse_bandwidth / 2,
        start_phase=0,
        sampling_rate_hz=1 / samples_step,
        direction="UP",  # TODO: check this
    )


def doppler_centroid_poly_from_metadata_node(
    image_generation_parameters_node: etree._Element, raster_info: RasterInfo
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters metadata xml node
    raster_info : RasterInfo
        product raster info

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    coeff_raw = [
        [float(c) for c in cc.text.split()]
        for cc in image_generation_parameters_node.findall("DopplerCentroid/DopplerCentroidCoefficients")
    ]
    ref_times = [
        PreciseDateTime.fromisoformat(tt.text)
        for tt in image_generation_parameters_node.findall("DopplerCentroid/TimeOfDopplerCentroidEstimate")
    ]
    coefficients = [[cc / raster_info.samples.step**cc_id for cc_id, cc in enumerate(c)] for c in coeff_raw]

    doppler_poly_list = [
        ConversionFunction(
            azimuth_reference_time=ref_times[c],
            function=coefficients[c],
            origin=raster_info.samples.start,
        )
        for c in range(len(coefficients))
    ]

    return DopplerEvaluator(functions=doppler_poly_list, azimuth_reference_times=np.array([ref_times]))


def doppler_rate_poly_from_metadata_node(
    image_generation_parameters_node: etree._Element, raster_info: RasterInfo
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler rate vector polynomial wrapper from metadata.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters metadata xml node
    raster_info : RasterInfo
        product raster info

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Rate polynomial
    """
    coeff_raw = [
        [float(c) for c in cc.text.split()]
        for cc in image_generation_parameters_node.findall("DopplerRateValues/DopplerRateValuesCoefficients")
    ]
    ref_times = [
        [PreciseDateTime.fromisoformat(t) for t in tt.text.split()]
        for tt in image_generation_parameters_node.findall("DopplerRateValues/DopplerRateReferenceTime")
    ]
    coefficients = [[cc / raster_info.samples.step**cc_id for cc_id, cc in enumerate(c)] for c in coeff_raw]

    doppler_poly_list = [
        ConversionFunction(
            azimuth_reference_time=ref_times[c][0],
            function=coefficients[c],
            origin=raster_info.samples.start,
        )
        for c in range(len(coefficients))
    ]

    return DopplerEvaluator(
        functions=doppler_poly_list,
        azimuth_reference_times=np.array([c.azimuth_reference_time for c in doppler_poly_list]),
    )


def coordinates_conversions_from_metadata(
    image_generation_parameters_node: etree._Element, raster_info: RasterInfo
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
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters metadata xml node
    raster_info : RasterInfo
        product raster info

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass
    """

    if image_generation_parameters_node.find("SlantRangeToGroundRange") is None:
        return CoordinatesConversions()

    node = image_generation_parameters_node.find("SlantRangeToGroundRange")
    # recovering coefficients and applying conversion factor meters to seconds
    az_ref_time = PreciseDateTime.fromisoformat(node.find("ZeroDopplerAzimuthTime").text)
    m2s_conversion_factor = 1 / (speed_of_light / 2)
    coeff_raw = [m2s_conversion_factor * float(c) for c in node.find("SlantToGroundRangeCoefficients").text.split()]
    ground_to_slant_coeff = [c / raster_info.samples.step**idx for idx, c in enumerate(coeff_raw)]
    ground_to_slant_poly = Polynomial(ground_to_slant_coeff)

    # slant to ground poly is not given, so it must be evaluated by inverting the ground to slant poly
    rng_axis = np.arange(0, (raster_info.samples.length + 1) * raster_info.samples.step, raster_info.samples.step)
    ground_to_slant_poly_evaluated = ground_to_slant_poly(rng_axis)
    slant_to_ground_poly = Polynomial.fit(
        x=ground_to_slant_poly_evaluated, y=rng_axis, deg=ground_to_slant_poly.degree()
    )

    ground_to_slant_poly_list = [
        ConversionFunction(
            azimuth_reference_time=raster_info.lines.start,
            origin=0,
            function=ground_to_slant_poly,
        )
    ]
    slant_to_ground_poly_list = [
        ConversionFunction(
            azimuth_reference_time=raster_info.lines.start,
            origin=0,
            function=slant_to_ground_poly,
        )
    ]

    return CoordinatesConversions(
        azimuth_reference_times=az_ref_time,
        ground_to_slant=ground_to_slant_poly_list,
        slant_to_ground=slant_to_ground_poly_list,
    )


@dataclass
class EOS04Attitude:
    """EOS04 sensor's attitude"""

    num: int  # attitude data numerosity
    yaw: np.ndarray  # platform yaw
    pitch: np.ndarray  # platform pitch
    roll: np.ndarray  # platform roll
    time_axis: np.ndarray  # PreciseDateTime axis to which attitude data applies
    time_step: float  # time axis step

    @staticmethod
    def from_metadata_node(attitude_information_node: etree._Element) -> EOS04Attitude:
        """Generating EOS04Attitude object directly from metadata xml node.

        Parameters
        ----------
        attitude_information_node : etree._Element
            AttitudeInformation xml node

        Returns
        -------
        EOS04Attitude
            sensor's attitude dataclass
        """

        time_axis = np.array(
            [
                PreciseDateTime.fromisoformat(t.text)
                for t in attitude_information_node.findall("AttitudeAngles/TimeStamp")
            ]
        )
        yaw = np.array([float(t.text) for t in attitude_information_node.findall("AttitudeAngles/yaw")])
        roll = np.array([float(t.text) for t in attitude_information_node.findall("AttitudeAngles/roll")])
        pitch = np.array([float(t.text) for t in attitude_information_node.findall("AttitudeAngles/pitch")])

        assert time_axis.size == yaw.size == roll.size == pitch.size

        return EOS04Attitude(
            num=time_axis.size,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            time_axis=time_axis,
            time_step=time_axis[1] - time_axis[0],
        )


@dataclass
class EOS04GeneralChannelInfo:
    """EOS04 general channel info representation dataclass"""

    channel_id: str
    product_name: str
    satellite: str
    swath: str
    acq_start_time: PreciseDateTime
    product_type: EOS04ProductType
    acquisition_mode: EOS04AcquisitionMode
    acquisition_mode_std: StandardSARAcquisitionMode
    polarization: SARPolarization
    projection: SARProjection
    orbit_direction: OrbitDirection

    @staticmethod
    def from_metadata_node(
        source_attributes_node: etree._Element, product_type: str, channel_id: str
    ) -> EOS04GeneralChannelInfo:
        """Generating EOS04GeneralChannelInfo object directly from metadata xml nodes.

        Parameters
        ----------
        source_attributes_node : etree._Element
            SourceAttributes metadata xml node
        product_type : str
            product type
        channel_id : str
            channel id

        Returns
        -------
        EOS04GeneralChannelInfo
            general channel info dataclass
        """

        start_time = PreciseDateTime.fromisoformat(
            source_attributes_node.find("SourceDataAcquisitionTime/StartTime").text
        )
        product_type = EOS04ProductType(product_type)
        acq_mode = EOS04AcquisitionMode(
            source_attributes_node.find("SourceDataAcquisitionParameters/ObservationMode").text
        )
        projection = SARProjection.SLANT_RANGE if product_type == EOS04ProductType.SLC else SARProjection.GROUND_RANGE
        orbit_direction = OrbitDirection[
            source_attributes_node.find("OrbitAndAttitude/OrbitInformation/PassDirection").text
        ]
        beam, polarization = unpack_channel_name(channel_name=channel_id)

        if acq_mode == EOS04AcquisitionMode.SCANSAR:
            acq_mode_std = StandardSARAcquisitionMode.SCANSAR
        else:
            acq_mode_std = StandardSARAcquisitionMode.UNKNOWN

        return EOS04GeneralChannelInfo(
            channel_id=channel_id,
            swath=f"B{str(beam)}",
            product_name=source_attributes_node.find("ProductID").text,
            satellite=source_attributes_node.find("Satellite").text,
            acq_start_time=start_time,
            product_type=product_type,
            acquisition_mode=acq_mode,
            acquisition_mode_std=acq_mode_std,
            projection=projection,
            polarization=polarization,
            orbit_direction=orbit_direction,
        )


@dataclass
class EOS04ChannelMetadata:
    """EOS04 channel metadata xml file wrapper"""

    channel_id: str
    general_info: EOS04GeneralChannelInfo
    orbit: Trajectory
    attitude: EOS04Attitude
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


class EOS04FolderLayout:
    """EOS04 file main directory architecture"""

    def __init__(self, path: Path) -> None:
        """Definition of internal architecture of a EOS04 product folder.

        Parameters
        ----------
        path : Path
            path to the EOS04 product base folder
        """
        self._product_path = path
        self._product_name = path.name
        self._band_meta_file = path.joinpath("BAND_META.txt")
        self._metadata_file = path.joinpath("product" + METADATA_EXTENSION)

    @property
    def band_meta_file(self) -> Path:
        """Path to the BAND_META.txt file"""
        return self._band_meta_file

    @property
    def metadata_file(self) -> Path:
        """Path to the product.xml metadata file"""
        return self._metadata_file

    def get_slant_range_grid_file(self, polarization: str | SARPolarization) -> Path:
        """Retrieving the _L1_SlantRange_grid.txt file for the input polarization.

        Parameters
        ----------
        polarization : str | SARPolarization
            polarization value

        Returns
        -------
        Path
            Path to the _L1_SlantRange_grid.txt for the selected polarization
        """
        pol = SARPolarization(polarization).name.upper()
        return self._product_path.joinpath("_".join([self._product_name, pol, "L1_Slant_Range_grid.txt"]))

    def get_beam_raster_file(
        self, polarization: str | SARPolarization, beam: int, product_type: EOS04ProductType
    ) -> Path:
        """Retrieving the raster file for the selected polarization and beam number.

        Parameters
        ----------
        polarization : str | SARPolarization
            polarization value
        beam : int
            selected beam number
        product_type : EOS04ProductType
            product type

        Returns
        -------
        Path
            Path to the beam raster file
        """
        pol = SARPolarization(polarization).name.upper()
        scene_folder = self._product_path.joinpath(f"scene_{pol}")
        if product_type == EOS04ProductType.GRD:
            return scene_folder.joinpath(f"imagery_{pol}" + RASTER_EXTENSION)
        return scene_folder.joinpath(f"imagery_{pol}_b{beam}" + RASTER_EXTENSION)


class EOS04Product:
    """EOS04 Product"""

    def __init__(self, path: str | Path) -> None:
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        self._layout = EOS04FolderLayout(self._product_path)

        # acquisition time, beams and polarizations
        self._acq_time, self._beams, self._pol_list, self._product_type = _get_basic_info_from_metadata(
            self._layout.metadata_file
        )

        if self._product_type == EOS04ProductType.GRD:
            # GRD products still presents beams in metadata but actually there is no dependency on beams! taking the
            # first one only to keep the name for the only available raster
            self._beams = [self._beams[0]]

        self._footprint = _retrieve_scene_footprint(self._layout.metadata_file)

        self._channels = [compose_channel_name(p, b) for p in self._pol_list for b in self._beams]

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def metadata_file(self) -> Path:
        """Returning the Path to the product metadata file"""
        return self._layout.metadata_file

    @property
    def channels_number(self) -> int:
        """Returning the number of channels for this product"""
        return len(self._channels)

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels in terms of SwathID (beam-polarization)"""
        return self._channels

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_raster_file_from_channel_name(self, channel_name: str) -> Path:
        """Get raster file path associated to input channel name.

        Parameters
        ----------
        channel_name : str
            selected channel name

        Returns
        -------
        Path
            raster file path
        """
        beam, pol = unpack_channel_name(channel_name)
        return self._layout.get_beam_raster_file(polarization=pol, beam=beam, product_type=self._product_type)


def is_eos04_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid EOS04 product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - metadata file exists
        - metadata basic info extraction works

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

    if not product.exists() or not product.is_dir():
        return False

    try:
        layout = EOS04FolderLayout(path=product)
    except Exception:
        return False

    if not layout.metadata_file.is_file():
        return False

    try:
        _, _, _, _ = _get_basic_info_from_metadata(layout.metadata_file)
    except Exception:
        return False

    return True
