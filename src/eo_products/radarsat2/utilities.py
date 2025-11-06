# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
RADARSAT-2 reader support module
--------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np
from arepytools.geometry.orbit import Orbit
from arepytools.timing.precisedatetime import PreciseDateTime
from lxml import etree
from numpy.polynomial import Polynomial
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

_METADATA_FILE = "product.xml"
_IMAGE_FORMAT = ".tif"
_DATA_SUFFIX = "imagery"
_LUT_BETA_FILE = "lutBeta.xml"


class RADARSATTimeOrdering(Enum):
    """RADARSAT-2 available Time Ordering"""

    INCREASING = "Increasing"
    DECREASING = "Decreasing"


class RADARSATProductType(Enum):
    """RADARSAT-2 L1 product types"""

    SLC = "SLC"  # stripmap, single look, complex, slant range
    SGF = "SGF"  # SAR Georeferenced Fine product
    SGX = "SGX"  # SAR Georeferenced Extra-Fine product
    SGC = "SGC"  # SAR Georeferenced Coarse product
    SSG = "SSG"  # SAR Systematic Geocorrected product
    SPG = "SPG"  # SAR Precision Geocorrected product
    SCF = "SCF"  # ScanSAR Sampled product type
    SCS = "SCS"  # ScanSAR Sampled product type


class RADARSATAcquisitionModes(Enum):
    """RADARSAT-2 L1 acquisition modes"""

    STRIPMAP = auto()
    SCANSAR = auto()
    SPOTLIGHT = auto()


def get_acquisition_mode(beam_mode: str) -> RADARSATAcquisitionModes:
    """
    Parameters
    ----------
    beam_mode : str
        RS2 beam mode mnemomic

    Returns
    -------
    RADARSATAcquisitionModes
        acquisition mode among STRIPMAP, SCANSAR, SPOTLIGHT
    """
    if "SCN" in beam_mode or "SCW" in beam_mode:
        return RADARSATAcquisitionModes.SCANSAR
    if "SLA" in beam_mode:
        return RADARSATAcquisitionModes.SPOTLIGHT
    return RADARSATAcquisitionModes.STRIPMAP


class InvalidRADARSATProduct(RuntimeError):
    """Invalid RADARSAT-2 product"""


def read_lut(path: str | Path) -> float:
    """Reading Beta LUT calibration file to extract the calibration factor.

    Parameters
    ----------
    path : str | Path
        Path to the Beta LUT calibration .xml file

    Returns
    -------
    float
        calibration factor
    """
    root = etree.parse(str(path)).getroot()
    gain = root.find("gains")
    calibration_factor = {float(f) for f in gain.text.split()}
    assert len(calibration_factor) == 1
    return calibration_factor.pop()


def raster_info_from_metadata(
    image_generation_parameters_node: etree._Element,
    image_attributes_node: etree._Element,
    namespace: dict[str, str],
    projection: SARProjection,
) -> RasterInfo:
    """Creating a RasterInfo metadata element from xml node.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        imageGenerationParameters metadata xml node
    image_attributes_node : etree._Element
        imageAttributes metadata xml node
    namespace : dict[str, str]
        xml namespace
    projection : SARProjection
        SAR data projection

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """
    lines = int(
        image_attributes_node.xpath(".//base:rasterAttributes/base:numberOfLines", namespaces=namespace)[0].text
    )
    samples = int(
        image_attributes_node.xpath(".//base:rasterAttributes/base:numberOfSamplesPerLine", namespaces=namespace)[
            0
        ].text
    )
    first_doppler_time = PreciseDateTime.fromisoformat(
        image_generation_parameters_node.xpath(
            ".//base:sarProcessingInformation/base:zeroDopplerTimeFirstLine", namespaces=namespace
        )[0].text
    )
    last_doppler_time = PreciseDateTime.fromisoformat(
        image_generation_parameters_node.xpath(
            ".//base:sarProcessingInformation/base:zeroDopplerTimeLastLine", namespaces=namespace
        )[0].text
    )
    if last_doppler_time < first_doppler_time:
        lines_start = last_doppler_time
    else:
        lines_start = first_doppler_time
    lines_step = abs(first_doppler_time - last_doppler_time) / (lines - 1)
    lines_step_unit = "s"

    # samples
    if projection == SARProjection.SLANT_RANGE:
        # slant range
        samples_start = float(
            image_generation_parameters_node.xpath(
                ".//base:sarProcessingInformation/base:slantRangeNearEdge", namespaces=namespace
            )[0].text
        ) / (speed_of_light / 2)
        samples_step = float(
            image_attributes_node.xpath(".//base:sampledPixelSpacing", namespaces=namespace)[0].text
        ) / (speed_of_light / 2)
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"
    else:
        # ground range
        samples_start = 0
        samples_step = float(image_attributes_node.xpath(".//base:sampledPixelSpacing", namespaces=namespace)[0].text)
        samples_step_unit = "m"
        celltype = "FLOAT32"

    raster_lines = RasterInfoAxis(length=lines, start=lines_start, step=lines_step, step_unit=lines_step_unit)
    raster_samples = RasterInfoAxis(length=samples, start=samples_start, step=samples_step, step_unit=samples_step_unit)

    return RasterInfo(
        lines=raster_lines,
        samples=raster_samples,
        data_type=celltype,
    )


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


def dataset_info_from_metadata(
    source_attributes_node: etree._Element,
    namespace: dict[str, str],
    projection: SARProjection,
    acq_mode: RADARSATAcquisitionModes,
) -> DatasetInfo:
    """Creating a DatasetInfo metadata element from xml nodes.

    Parameters
    ----------
    source_attributes_node : etree._Element
        sourceAttributes metadata xml node
    namespace : dict[str, str]
        xml namespace
    projection : SARProjection
        product projection
    acq_mode : RADARSATAcquisitionModes
        acquisition mode

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """

    return DatasetInfo(
        fc_hz=float(
            source_attributes_node.xpath(".//base:radarParameters/base:radarCenterFrequency", namespaces=namespace)[
                0
            ].text
        ),
        acquisition_mode=acq_mode.value,
        sensor_name=source_attributes_node.xpath(".//base:satellite", namespaces=namespace)[0].text,
        image_type="AZIMUTH FOCUSED RANGE COMPENSATED" if projection == SARProjection.SLANT_RANGE else "MULTILOOK",
        projection=projection.value,
        side_looking=source_attributes_node.xpath(".//base:radarParameters/base:antennaPointing", namespaces=namespace)[
            0
        ].text.upper(),
    )


def swath_info_from_metadata(
    source_attributes_node: etree._Element, namespace: dict[str, str], prod_type: RADARSATProductType
) -> SwathInfo:
    """Creating a SwathInfo metadata object from metadata file.

    Parameters
    ----------
    source_attributes_node : etree._Element
        sourceAttributes xml node
    namespace : dict[str, str]
        xml namespace
    prod_type : RADARSATProductType
        product type

    Returns
    -------
    SwathInfo
        SwathInfo metadata object
    """
    rank = 0
    acquisition_prf = 0
    if prod_type == RADARSATProductType.SLC:
        rank = int(source_attributes_node.xpath(".//base:radarParameters/base:rank", namespaces=namespace)[0].text)
        acquisition_prf = float(
            source_attributes_node.xpath(".//base:radarParameters/base:pulseRepetitionFrequency", namespaces=namespace)[
                0
            ].text
        )
    return SwathInfo(
        swath="S1",
        rank=rank,
        azimuth_steering_rate_poly=(0, 0, 0),
        prf=acquisition_prf,
    )


def state_vectors_from_metadata(orbit_data_node: etree._Element, namespace: dict[str, str]) -> StateVectors:
    """Generating StateVectors object directly from product metadata.

    Parameters
    ----------
    orbit_data_node : etree._Element
        OrbitInformation xml node

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """
    pos_x = np.array(
        [float(x.text) for x in orbit_data_node.xpath(".//base:stateVector/base:xPosition", namespaces=namespace)]
    )
    pos_y = np.array(
        [float(y.text) for y in orbit_data_node.xpath(".//base:stateVector/base:yPosition", namespaces=namespace)]
    )
    pos_z = np.array(
        [float(z.text) for z in orbit_data_node.xpath(".//base:stateVector/base:zPosition", namespaces=namespace)]
    )
    positions = np.stack([pos_x, pos_y, pos_z], axis=1)

    vel_x = np.array(
        [float(x.text) for x in orbit_data_node.xpath(".//base:stateVector/base:xVelocity", namespaces=namespace)]
    )
    vel_y = np.array(
        [float(y.text) for y in orbit_data_node.xpath(".//base:stateVector/base:yVelocity", namespaces=namespace)]
    )
    vel_z = np.array(
        [float(z.text) for z in orbit_data_node.xpath(".//base:stateVector/base:zVelocity", namespaces=namespace)]
    )
    velocities = np.stack([vel_x, vel_y, vel_z], axis=1)

    time_axis = np.array(
        [
            PreciseDateTime.fromisoformat(t.text)
            for t in orbit_data_node.xpath(".//base:stateVector/base:timeStamp", namespaces=namespace)
        ]
    )

    assert positions.shape[0] == velocities.shape[0] == time_axis.size
    return StateVectors(
        num=time_axis.size,
        positions=positions,
        velocities=velocities,
        time_axis=time_axis,
        time_step=np.mean(np.diff(time_axis)),
        orbit_direction=OrbitDirection(
            orbit_data_node.xpath(".//base:passDirection", namespaces=namespace)[0].text.lower()
        ),
    )


def sampling_constants_from_metadata(
    image_generation_parameters_node: etree._Element,
    namespace: dict[str, str],
    raster_info: RasterInfo,
    projection: SARProjection,
) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata element from xml nodes.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        imageGenerationParameters metadata xml node
    namespace : dict[str, str]
        xml namespace
    raster_info : RasterInfo
        raster info
    projection : SARProjection
        product projection

    Returns
    -------
    SARSamplingFrequencies
        sampling frequencies
    """

    if projection == SARProjection.SLANT_RANGE:
        sampling_constants = SARSamplingFrequencies(
            azimuth_freq_hz=1 / raster_info.lines.step,
            azimuth_bandwidth_freq_hz=float(
                image_generation_parameters_node.xpath(
                    ".//base:sarProcessingInformation/base:totalProcessedAzimuthBandwidth", namespaces=namespace
                )[0].text
            ),
            range_freq_hz=1 / raster_info.samples.step,
            range_bandwidth_freq_hz=float(
                image_generation_parameters_node.xpath(
                    ".//base:sarProcessingInformation/base:totalProcessedRangeBandwidth", namespaces=namespace
                )[0].text
            ),
        )
    else:
        sampling_constants = SARSamplingFrequencies(
            azimuth_freq_hz=0,
            azimuth_bandwidth_freq_hz=0,
            range_freq_hz=0,
            range_bandwidth_freq_hz=0,
        )

    return sampling_constants


def calibration_factor_from_metadata(file_path: str | Path, product_type: RADARSATProductType) -> float:
    """Retrieving the correct calibration factor to convert digital values to Beta Nought.

    Parameters
    ----------
    file_path : str | Path
        Path to the calibration LUT .xml file

    Returns
    -------
    float
        beta nought calibration factor
    """
    cal_factor = read_lut(path=file_path)
    if product_type == RADARSATProductType.SLC:
        return 1 / cal_factor
    return 1 / np.sqrt(cal_factor)


def doppler_centroid_poly_from_metadata(
    doppler_centroid_node: etree._Element, namespace: dict[str, str]
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    doppler_centroid_node : etree._Element
        dopplerCentroid metadata xml node
    namespace : dict[str, str]
        xml namespace

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    azimuth_time_ref = PreciseDateTime.fromisoformat(
        doppler_centroid_node.xpath(".//base:timeOfDopplerCentroidEstimate", namespaces=namespace)[0].text
    )
    range_time_ref = float(
        doppler_centroid_node.xpath(".//base:dopplerCentroidReferenceTime", namespaces=namespace)[0].text
    )
    coeff_raw = [
        float(c)
        for c in doppler_centroid_node.xpath(".//base:dopplerCentroidCoefficients", namespaces=namespace)[
            0
        ].text.split()
    ]

    doppler_poly_list = [
        ConversionFunction(
            azimuth_reference_time=azimuth_time_ref,
            origin=range_time_ref,
            function=Polynomial(coeff_raw),
        )
    ]

    return DopplerEvaluator(functions=doppler_poly_list, azimuth_reference_times=np.array([azimuth_time_ref]))


def doppler_rate_poly_from_metadata(doppler_rate_node: etree._Element, namespace: dict[str, str]) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler rate vector polynomial wrapper from metadata.

    Parameters
    ----------
    doppler_rate_node : etree._Element
        dopplerRateValues metadata xml node
    namespace : dict[str, str]
        xml namespace

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Rate polynomial
    """
    azimuth_ref = PreciseDateTime.fromisoformat(
        doppler_rate_node.xpath("..//base:dopplerCentroid/base:timeOfDopplerCentroidEstimate", namespaces=namespace)[
            0
        ].text
    )
    range_time_ref = float(doppler_rate_node.xpath(".//base:dopplerRateReferenceTime", namespaces=namespace)[0].text)
    coeff_raw = [
        float(c)
        for c in doppler_rate_node.xpath(".//base:dopplerRateValuesCoefficients", namespaces=namespace)[0].text.split()
    ]

    doppler_poly_list = [
        ConversionFunction(
            azimuth_reference_time=azimuth_ref,
            origin=range_time_ref,
            function=Polynomial(coeff_raw),
        )
    ]

    return DopplerEvaluator(functions=doppler_poly_list, azimuth_reference_times=np.array([azimuth_ref]))


def coordinates_conversions_from_metadata(
    image_generation_parameters_node: etree._Element, namespace: dict[str, str], raster_info: RasterInfo
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
        imageGenerationParameters metadata xml node
    namespace : dict[str, str]
        xml namespace
    raster_info : RasterInfo
        product raster info

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass
    """
    conversion_nodes = image_generation_parameters_node.xpath(".//base:slantRangeToGroundRange", namespaces=namespace)
    ground_to_slant_poly_list = []
    slant_to_ground_poly_list = []
    azimuth_times = []
    for node in conversion_nodes:
        azimuth_ref_time = PreciseDateTime.fromisoformat(
            node.xpath(".//base:zeroDopplerAzimuthTime", namespaces=namespace)[0].text
        )
        origin = float(node.xpath(".//base:groundRangeOrigin", namespaces=namespace)[0].text)
        coefficients = [
            float(c) / (speed_of_light / 2)
            for c in node.xpath(".//base:groundToSlantRangeCoefficients", namespaces=namespace)[0].text.split()
        ]
        ground_to_slant_poly = Polynomial(coef=coefficients)
        ground_to_slant_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=azimuth_ref_time,
                origin=origin,
                function=ground_to_slant_poly,
            )
        )
        # slant to ground poly is not given, so it must be evaluated by inverting the ground to slant poly
        rng_axis = np.arange(
            0, (raster_info.samples.length + 1) * raster_info.samples.step, raster_info.samples.step
        ) * (speed_of_light / 2)
        ground_to_slant_poly_evaluated = ground_to_slant_poly(rng_axis - origin)
        slant_to_ground_poly = Polynomial.fit(
            x=ground_to_slant_poly_evaluated, y=rng_axis, deg=ground_to_slant_poly.degree()
        )
        slant_to_ground_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=azimuth_ref_time,
                origin=0,
                function=slant_to_ground_poly,
            )
        )
        azimuth_times.append(azimuth_ref_time)

    return CoordinatesConversions(
        azimuth_reference_times=np.array(azimuth_times),
        ground_to_slant=ground_to_slant_poly_list,
        slant_to_ground=slant_to_ground_poly_list,
    )


@dataclass
class RADARSATAttitude:
    """RADARSAT-2 sensor's attitude"""

    num: int  # attitude data numerosity (same as interpolated orbit)
    yaw_deg: np.ndarray  # platform yaw
    pitch_deg: np.ndarray  # platform pitch
    roll_deg: np.ndarray  # platform roll
    time_axis: np.ndarray  # PreciseDateTime axis to which attitude data applies
    time_step: float  # time axis step

    @staticmethod
    def _unpack_attitude_data(
        node: etree._Element, namespace: dict[str, str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracting yaw, pitch, roll and times vector from metadata node.

        Parameters
        ----------
        node : etree._Element
            attitudeInformation node from .xml metadata
        namespace : dict[str, str]
            xml namespace

        Returns
        -------
        np.ndarray
            yaw in deg, with shape (N,)
        ndarray
            pitch in deg, with shape (N,)
        ndarray
            roll in deg, with shape (N,)
        ndarray
            time axis, with shape (N,)
        """
        yaw = np.array([float(x.text) for x in node.xpath(".//base:attitudeAngles/base:yaw", namespaces=namespace)])
        pitch = np.array([float(y.text) for y in node.xpath(".//base:attitudeAngles/base:pitch", namespaces=namespace)])
        roll = np.array([float(z.text) for z in node.xpath(".//base:attitudeAngles/base:roll", namespaces=namespace)])
        time_axis = np.array(
            [
                PreciseDateTime.fromisoformat(t.text)
                for t in node.xpath(".//base:attitudeAngles/base:timeStamp", namespaces=namespace)
            ]
        )

        return yaw, pitch, roll, time_axis

    # TODO: this should be a class method (cls, ...)
    @staticmethod
    def from_metadata(attitude_data_node: etree._Element, namespace: dict[str, str]) -> RADARSATAttitude:
        """Generating RADARSATAttitude object directly from metadata xml node.

        Parameters
        ----------
        attitude_data_node : etree._Element
            attitudeInformation xml node
        namespace : dict[str, str]
            xml namespace

        Returns
        -------
        RADARSATAttitude
            sensor's attitude dataclass
        """
        yaw, pitch, roll, times = RADARSATAttitude._unpack_attitude_data(node=attitude_data_node, namespace=namespace)
        assert yaw.size == pitch.size == roll.size == times.size
        return RADARSATAttitude(
            num=times.size,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            time_axis=times,
            time_step=np.mean(np.diff(times)),
        )


@dataclass
class RADARSATGeneralChannelInfo:
    """RADARSAT-2 general channel info representation dataclass"""

    product_name: str
    product_id: int
    channel_id: str
    swath: str
    satellite: str
    acq_start_time: PreciseDateTime
    product_type: RADARSATProductType
    acquisition_mode: RADARSATAcquisitionModes
    acquisition_mode_std: StandardSARAcquisitionMode
    polarization: SARPolarization
    projection: SARProjection
    orbit_direction: OrbitDirection

    @staticmethod
    def from_metadata_node(
        root: etree._Element, namespace: dict[str, str], name: str, channel_id: str
    ) -> RADARSATGeneralChannelInfo:
        """Generating S1GeneralChannelInfo object directly from metadata xml nodes.

        Parameters
        ----------
        root : etree._Element
            root metadata xml node
        namespace : dict[str, str]
            xml namespace
        name : str
            product name
        channel_id : str
            current channel ID

        Returns
        -------
        RADARSATGeneralChannelInfo
            general channel info dataclass
        """
        prod_type = RADARSATProductType(
            root.xpath(
                ".//base:imageGenerationParameters/base:generalProcessingInformation/base:productType",
                namespaces=namespace,
            )[0].text.upper()
        )
        beam_mode = root.xpath(
            ".//base:sourceAttributes/base:beamModeMnemonic",
            namespaces=namespace,
        )[0].text.upper()
        mode = get_acquisition_mode(beam_mode)
        if mode == RADARSATAcquisitionModes.SCANSAR:
            mode_std = StandardSARAcquisitionMode.SCANSAR
        elif mode == RADARSATAcquisitionModes.STRIPMAP:
            mode_std = StandardSARAcquisitionMode.STRIPMAP
        else:
            mode_std = StandardSARAcquisitionMode.SPOTLIGHT
        return RADARSATGeneralChannelInfo(
            product_name=name,
            product_id=root.xpath(".//base:productId", namespaces=namespace)[0].text,
            channel_id=channel_id,
            swath="S1",
            satellite=root.xpath(".//base:sourceAttributes/base:satellite", namespaces=namespace)[0].text,
            acq_start_time=PreciseDateTime.fromisoformat(
                root.xpath(".//base:sourceAttributes/base:rawDataStartTime", namespaces=namespace)[0].text
            ),
            product_type=prod_type,
            acquisition_mode=mode,
            acquisition_mode_std=mode_std,
            projection=get_projection_from_product_type(prod_type=prod_type),
            polarization=SARPolarization[channel_id.split("_")[-1].upper()],
            orbit_direction=OrbitDirection(
                root.xpath(
                    ".//base:sourceAttributes/base:orbitAndAttitude/base:orbitInformation/base:passDirection",
                    namespaces=namespace,
                )[0].text.lower()
            ),
        )


@dataclass
class RADARSATChannelMetadata:
    """RADARSAT-2 channel metadata dataclass"""

    general_info: RADARSATGeneralChannelInfo
    attitude: RADARSATAttitude
    image_calibration_factor: float
    image_radiometric_quantity: SARRadiometricQuantity
    burst_info: BurstInfo
    raster_info: RasterInfo
    dataset_info: DatasetInfo
    swath_info: SwathInfo
    sampling_constants: SARSamplingFrequencies
    state_vectors: StateVectors
    orbit: Orbit
    samples_ordering: RADARSATTimeOrdering
    lines_ordering: RADARSATTimeOrdering
    doppler_centroid_poly: DopplerEvaluator
    doppler_rate_poly: DopplerEvaluator
    coordinate_conversions: CoordinatesConversions


def get_basic_info_from_metadata(
    metadata_path: str | Path,
) -> tuple[PreciseDateTime, RADARSATProductType, list[str], tuple[float, float, float, float], str]:
    """Recovering acquisition time and list of channels.

    Parameters
    ----------
    metadata_path : str | Path
        Path to RADARSAT-2 metadata file

    Returns
    -------
    PreciseDateTime
        acquisition time in PreciseDateTime format
    RADARSATProductType
        product type
    list[str]
        list of channels ids
    tuple[float, float, float, float]
        scene footprint [min lat, max lat, min lon, max lon]
    str
        beam mode
    """
    metadata_path = Path(metadata_path)
    mtd = metadata_path.read_text(encoding="UTF-8")

    # regex init
    acq_time_re = re.compile("(?<=<rawDataStartTime>).*(?=</rawDataStartTime>)")
    type_re = re.compile("(?<=<productType>).*(?=</productType>)")
    pols_re = re.compile("(?<=<polarizations>).*(?=</polarizations>)")
    footprint_lat_re = re.compile('(?<=<latitude units="deg">).*(?=</latitude>)')
    footprint_lon_re = re.compile('(?<=<longitude units="deg">).*(?=</longitude>)')
    beam_mode_re = re.compile("(?<=<beamModeMnemonic>).*(?=</beamModeMnemonic>)")

    # info extraction
    acq_time = acq_time_re.findall(mtd)[0]
    acq_type = type_re.findall(mtd)[0].lower()
    acq_pols = pols_re.findall(mtd)[0].lower()
    beam_mode = beam_mode_re.findall(mtd)[0].lower()

    # generating channels names
    pol_list = acq_pols.split()
    channels_list = [acq_type + "_" + pol for pol in pol_list]

    # recovering scene footprint
    footprint_lat = [float(f) for f in footprint_lat_re.findall(mtd)]
    footprint_lon = [float(f) for f in footprint_lon_re.findall(mtd)]
    footprint = (min(footprint_lat), max(footprint_lat), min(footprint_lon), max(footprint_lon))

    return (
        PreciseDateTime.fromisoformat(acq_time),
        RADARSATProductType(acq_type.upper()),
        channels_list,
        footprint,
        beam_mode,
    )


def get_projection_from_product_type(prod_type: RADARSATProductType) -> SARProjection:
    """Get product projection from product type.

    Parameters
    ----------
    prod_type : RADARSATProductType
        product type

    Returns
    -------
    SARProjection
        product projection
    """
    if prod_type is RADARSATProductType.SLC:
        return SARProjection.SLANT_RANGE

    return SARProjection.GROUND_RANGE


class RADARSATProduct:
    """RADARSAT-2 product object"""

    def __init__(self, path: str | Path) -> None:
        """RADARSAT-2 Product init from directory path.

        Parameters
        ----------
        path : str | Path
            path to RADARSAT-2 product
        """
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        self._metadata_path = self._product_path.joinpath(_METADATA_FILE)
        self._acq_time, self._product_type, self._channel_list_by_swath_id, self._footprint, self._beam_mode = (
            get_basic_info_from_metadata(metadata_path=self._metadata_path)
        )
        self._channels_number = len(self._channel_list_by_swath_id)
        self._data_paths = [
            f
            for f in self._product_path.iterdir()
            if f.name.startswith(_DATA_SUFFIX) and f.name.endswith(_IMAGE_FORMAT)
        ]
        self._beta_calibration_file = self._product_path.joinpath(_LUT_BETA_FILE)

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
        """Returning the list of channels in terms of SwathID (swath-polarization)"""
        return self._channel_list_by_swath_id

    @property
    def data_list(self) -> list[Path]:
        """Returning the list of raster data files of RADARSAT-2 product"""
        return self._data_paths

    @property
    def product_type(self) -> RADARSATProductType:
        """Returning the product type"""
        return self._product_type

    @property
    def metadata_file(self) -> Path:
        """Returning the product metadata file path of RADARSAT-2 product"""
        return self._metadata_path

    @property
    def beta_calibration_lut_file(self) -> Path:
        """Returning the beta LUT calibration file path"""
        return self._beta_calibration_file

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
        return [r for r in self.data_list if channel_name.split("_")[-1] in r.name.lower()][0]


def is_radarsat_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid RADARSAT-2 product, basic version.

    Conditions to be met for basic validity:
        - path is dir
        - metadata file exist
        - read basic info from metadata file

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

    if not product.is_dir():
        return False

    if not product.joinpath(_METADATA_FILE).exists():
        return False

    # read basic info from metadata file
    try:
        get_basic_info_from_metadata(metadata_path=product.joinpath(_METADATA_FILE))
    except Exception:
        return False

    return True
