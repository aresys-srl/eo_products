# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
SAOCOM reader support module
----------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from arepytools.geometry.orbit import Orbit
from arepytools.io.metadata import DopplerCentroid, DopplerCentroidVector, DopplerRate, DopplerRateVector
from arepytools.math.genericpoly import SortedPolyList, create_sorted_poly_list
from arepytools.timing.precisedatetime import PreciseDateTime
from lxml import etree
from numpy.polynomial import Polynomial

from eo_products.common.utilities import (
    BurstInfo,
    ConversionFunction,
    CoordinatesConversions,
    DatasetInfo,
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

_MANIFEST_EXTENSION = ".xemt"
_RASTER_METADATA_EXTENSION = ".xml"


polarization_dict = {
    "vv": SARPolarization.VV,
    "hh": SARPolarization.HH,
    "hv": SARPolarization.HV,
    "vh": SARPolarization.VH,
}

data_type_dict = {
    "FLOAT32": "f4",
    "FLOAT_COMPLEX": "c8",
    "INT16": "i2",
    "INT32": "i4",
    "UINT16": "u2",
    "UINT32": "u4",
    "INT8": "b",
    "UINT8": "B",
    "INT16_COMPLEX": "i2, i2",
    "INT_COMPLEX": "i4, i4",
    "DOUBLE_COMPLEX": "c16",
    "FLOAT64": "f8",
    "INT8_COMPLEX": "i1, i1",
}

byte_order_dict = {"BIGENDIAN": ">", "LITTLEENDIAN": "<"}


class InvalidSAOCOMProduct(RuntimeError):
    """Invalid SAOCOM product"""


class InvalidHeaderOffset(ValueError):
    """Invalid header offset"""


class InvalidRowPrefix(ValueError):
    """Invalid prefix value"""


class InvalidDataType(ValueError):
    """Invalid data type"""


class BlockExceedsRasterLimits(RuntimeError):
    """Block to be read exceeds raster limits"""


class SAOCOMProductType(Enum):
    """SAOCOM L1 product level"""

    SLC = "SLC"  # Slant Range, Single Look Complex (SLC, lvl 1)
    GRD = "GRD"  # Ground Range Multi Look Detected (GRD, lvl 1, phase lost)


class SAOCOMAcquisitionMode(Enum):
    """SAOCOM L1 acquisition modes"""

    STRIPMAP = "STRIPMAP"
    TOPSAR = "TOPSAR"


def _get_conversion_polynomial(node: etree._Element, tag: str) -> list[ConversionFunction]:
    """Composing a list of ConversionPolynomial object from main metadata node and selected poly tag.

    Parameters
    ----------
    node : etree._Element
        Channel metadata node
    tag : str
        selected conversion polynomial, "GroundToSlant" and "SlantToGround"

    Returns
    -------
    list[ConversionPolynomial]
        list of ConversionPolynomial, one for each occurrence in main metadata node
    """
    azimuth_ref_times, range_ref_times, coefficients = _extract_poly_info_from_node(node, tag)

    # removing cross azimuth terms from coefficients
    coefficients = [[c[0], c[1], c[4], c[5], c[6]] for c in coefficients]

    return [
        ConversionFunction(
            azimuth_reference_time=az_time, origin=range_ref_times[idx], function=Polynomial(coefficients[idx])
        )
        for idx, az_time in enumerate(azimuth_ref_times)
    ]


def raster_info_from_metadata(node: etree._Element) -> RasterInfo:
    """Creating a RasterInfo metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        RasterInfo metadata node

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """

    raster_lines = RasterInfoAxis(
        length=int(node.find("Lines").text),
        start=PreciseDateTime.from_utc_string(node.find("LinesStart").text),
        step=float(node.find("LinesStep").text),
        step_unit=node.find("LinesStep").get("unit"),
    )
    raster_samples = RasterInfoAxis(
        length=int(node.find("Samples").text),
        start=float(node.find("SamplesStart").text),
        step=float(node.find("SamplesStep").text),
        step_unit=node.find("SamplesStep").get("unit"),
    )

    return RasterInfo(
        lines=raster_lines,
        samples=raster_samples,
        data_type=node.find("CellType").text,
        raster_name=node.find("FileName").text,
    )


def burst_info_from_metadata(node: etree._Element, raster_info: RasterInfo) -> BurstInfo:
    """Generating BurstInfo object directly from metadata.

    Parameters
    ----------
    node : etree._Element
        BurstInfo metadata node
    raster_info : RasterInfo
        product raster info

    Returns
    -------
    BurstInfo
        burst info dataclass
    """
    if node is None:
        # GRD does not have burst
        return BurstInfo(
            num=1,
            lines_per_burst=raster_info.lines.length,
            samples_per_burst=raster_info.samples.length,
            azimuth_start_times=np.array([raster_info.lines.start]),
            range_start_times=np.array([raster_info.samples.start]),
        )

    return BurstInfo(
        num=int(node.find("NumberOfBursts").text),
        lines_per_burst=int(node.find("LinesPerBurst").text),
        samples_per_burst=raster_info.samples.length,
        azimuth_start_times=np.array(
            [PreciseDateTime.from_utc_string(t.text) for t in node.findall("Burst/AzimuthStartTime")]
        ),
        range_start_times=np.array([float(t.text) for t in node.findall("Burst/RangeStartTime")]),
    )


def dataset_info_from_metadata(node: etree._Element) -> DatasetInfo:
    """Creating a DatasetInfo metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        DatasetInfo metadata node

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """
    return DatasetInfo(
        fc_hz=float(node.find("fc_hz").text),
        acquisition_mode=node.find("AcquisitionMode").text,
        image_type=node.find("ImageType").text,
        projection=node.find("Projection").text,
        sensor_name=node.find("SensorName").text,
        side_looking=node.find("SideLooking").text,
    )


def swath_info_from_metadata(node: etree._Element) -> SwathInfo:
    """Creating a SwathInfo metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        SwathInfo metadata node

    Returns
    -------
    SwathInfo
        SwathInfo metadata object
    """

    rank = 0
    azimuth_steering_rate_poly = (0, 0, 0)
    try:
        rank = int(node.find("Rank").text)
        azimuth_steering_rate_poly = tuple([float(p.text) for p in node.findall("AzimuthSteeringRatePol/val")])
    except Exception:
        # some of these info are optional in the metadata but all of them are not used so much
        pass

    return SwathInfo(
        swath=node.find("Swath").text,
        rank=rank,
        azimuth_steering_rate_poly=azimuth_steering_rate_poly,
        prf=float(node.find("AcquisitionPRF").text),
    )


def state_vectors_from_metadata(node: etree._Element) -> StateVectors:
    """Generating StateVectors object directly from product metadata.

    Parameters
    ----------
    node : etree._Element
        StateVectorData xml node

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """
    numerosity = int(node.find("nSV_n").text)
    time_step = float(node.find("dtSV_s").text)
    time_start = PreciseDateTime.from_utc_string(node.find("t_ref_Utc").text)

    return StateVectors(
        num=numerosity,
        positions=np.array([float(p.text) for p in node.findall("pSV_m/val")]).reshape(-1, 3),
        velocities=np.array([float(v.text) for v in node.findall("vSV_mOs/val")]).reshape(-1, 3),
        time_axis=np.arange(0, numerosity * time_step, time_step) + time_start,
        time_step=time_step,
        orbit_direction=OrbitDirection(node.find("OrbitDirection").text.lower()),
    )


def sampling_constants_from_metadata(node: etree._Element) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        SamplingConstants metadata node

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    return SARSamplingFrequencies(
        azimuth_freq_hz=float(node.find("faz_hz").text),
        azimuth_bandwidth_freq_hz=float(node.find("Baz_hz").text),
        range_freq_hz=float(node.find("frg_hz").text),
        range_bandwidth_freq_hz=float(node.find("Brg_hz").text),
    )


def pulse_from_metadata(node: etree._Element) -> PulseInfo | None:
    """Creating a PulseInfo metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        Pulse metadata node

    Returns
    -------
    PulseInfo | None
        PulseInfo metadata object or None if node not found
    """
    if node is None:
        return None

    return PulseInfo(
        length_s=float(node.find("PulseLength").text),
        bandwidth_hz=float(node.find("Bandwidth").text),
        energy_j=float(node.find("PulseEnergy").text),
        sampling_rate_hz=float(node.find("PulseSamplingRate").text),
        start_frequency_hz=float(node.find("PulseStartFrequency").text),
        start_phase=float(node.find("PulseStartPhase").text),
        direction=node.find("Direction").text,
    )


def doppler_poly_from_metadata(node: etree._Element, doppler_node_tag: str) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools object for Doppler Polynomial from metadata file.

    Parameters
    ----------
    node : etree._Element
        Channel metadata node
    doppler_node_tag : str
        doppler polynomial node tag, it could be "DopplerCentroid" or "DopplerRate"

    Returns
    -------
    SortedPolyList
        Doppler polynomial SortedPolyList object
    """

    azimuth_ref_times, range_ref_times, coefficients = _extract_poly_info_from_node(node, doppler_node_tag)
    if not azimuth_ref_times and not range_ref_times and not coefficients:
        # GRD does not have these polynomials, so a set of 0-valued coefficients is created
        coefficients = [[0] * 7]
        azimuth_ref_times = [PreciseDateTime.from_utc_string(node.find("RasterInfo/LinesStart").text)]
        range_ref_times = [float(node.find("RasterInfo/SamplesStart").text)]

    doppler_poly_list = []
    for az_t_ref, rng_t_ref, coeffs in zip(azimuth_ref_times, range_ref_times, coefficients, strict=False):
        if doppler_node_tag == "DopplerCentroid":
            doppler_poly_list.append(
                DopplerCentroid(
                    i_ref_az=az_t_ref,
                    i_ref_rg=rng_t_ref,
                    i_coefficients=coeffs,
                )
            )
        else:
            doppler_poly_list.append(
                DopplerRate(
                    i_ref_az=az_t_ref,
                    i_ref_rg=rng_t_ref,
                    i_coefficients=coeffs,
                )
            )

    doppler_vector = (
        DopplerCentroidVector(doppler_poly_list)
        if doppler_node_tag == "DopplerCentroid"
        else DopplerRateVector(doppler_poly_list)
    )

    return create_sorted_poly_list(poly2d_vector=doppler_vector)


def coordinates_conversions_from_metadata(node: etree._Element) -> CoordinatesConversions:
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
    node : etree._Element
        Channel metadata node

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass
    """

    ground_to_slant_poly_list = _get_conversion_polynomial(node=node, tag="GroundToSlant")
    slant_to_ground_poly_list = _get_conversion_polynomial(node=node, tag="SlantToGround")

    azimuth_ref_times = [PreciseDateTime.from_utc_string(t.text) for t in node.findall("GroundToSlant/taz0_Utc")]

    return CoordinatesConversions(
        azimuth_reference_times=np.array(azimuth_ref_times),
        ground_to_slant=ground_to_slant_poly_list,
        slant_to_ground=slant_to_ground_poly_list,
    )


def read_raster(
    raster_file_name: str | Path,
    num_of_samples: int,
    num_of_lines: int,
    data_type: str = "FLOAT32",
    binary_ordering_mode: str = "LITTLEENDIAN",
    block_to_read: list[int] | None = None,
    header_offset: int = 0,
    row_prefix: int = 0,
) -> np.ndarray:
    """Read raster file data.

    Parameters
    ----------
    raster_file_name : str | Path
        path to the raster file to be read
    num_of_samples : int
        number of samples to be read
    num_of_lines : int
        number of lines to be read
    data_type : str, optional
        data type corresponding to the raster itself, by default FLOAT32
    binary_ordering_mode : str, optional
        binary ordering mode corresponding to the raster itself, by default LITTLEENDIAN
    block_to_read : list[int] | None, optional
        data block to be read, to be specified as a list of 4 integers, in the form:
            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None
    header_offset : int, optional
        header offset of the raster file, by default 0
    row_prefix : int, optional
        row prefix of the raster file, by default 0

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)

    Raises
    ------
    InvalidHeaderOffset
        if header offset is negative
    InvalidRowPrefix
        if row prefix is negative
    UnsupportedDataType
        invalid data type
    BlockExceedsRasterLimits
        data type not yet supported
    BlockExceedsRasterLimits
        if first line to be read is negative
    BlockExceedsRasterLimits
        if first sample to be read is negative
    BlockExceedsRasterLimits
        if block to be read exceeds raster number of lines
    BlockExceedsRasterLimits
        if block to be read exceeds raster number of samples
    """

    if header_offset < 0:
        raise InvalidHeaderOffset("header_offset should be non-negative")

    if row_prefix < 0:
        raise InvalidRowPrefix("row_prefix should be non-negative")

    if data_type in data_type_dict:
        data_type_numpy_value = data_type_dict[data_type]
        data_type = np.dtype(data_type_numpy_value)
    else:
        raise InvalidDataType(f"Unknown data type id: {data_type.value}")

    file_data_type = np.dtype(byte_order_dict[binary_ordering_mode.value] + data_type_numpy_value)

    # Compute the items to read
    if block_to_read is None:
        lines_to_read = num_of_lines
        samples_to_read = num_of_samples
        first_line = 0
        first_sample = 0
    else:
        lines_to_read = block_to_read[2]
        samples_to_read = block_to_read[3]
        first_line = block_to_read[0]
        first_sample = block_to_read[1]

    # convert to int
    lines_to_read = int(lines_to_read)
    samples_to_read = int(samples_to_read)
    first_line = int(first_line)
    first_sample = int(first_sample)

    if first_line < 0:
        raise BlockExceedsRasterLimits("First line to read should be non-negative")

    if first_sample < 0:
        raise BlockExceedsRasterLimits("First sample to read should be non-negative")

    if first_line + lines_to_read > num_of_lines:
        raise BlockExceedsRasterLimits("Block to read exceeds max num lines")

    if first_sample + samples_to_read > num_of_samples:
        raise BlockExceedsRasterLimits("Block to read exceeds max num samples")

    # Read data from file
    with open(raster_file_name, "rb") as fdesc:
        if samples_to_read == num_of_samples and row_prefix == 0:
            offset_byte = header_offset + first_line * num_of_samples * data_type.itemsize
            data = np.fromfile(
                fdesc,
                dtype=file_data_type,
                count=lines_to_read * samples_to_read,
                offset=offset_byte,
            )

            return data.reshape((lines_to_read, samples_to_read))

        data = np.empty((lines_to_read, samples_to_read), dtype=data_type)

        offset_byte = (
            (first_line * num_of_samples + first_sample) * data_type.itemsize
            + row_prefix * (first_line + 1)
            + header_offset
        )
        fdesc.seek(offset_byte, 0)

        offset_line_byte = (num_of_samples - samples_to_read) * data_type.itemsize + row_prefix

        for line in range(lines_to_read):
            offset_byte = offset_line_byte if line > 0 else 0
            data[line, :] = np.fromfile(fdesc, dtype=file_data_type, count=samples_to_read, offset=offset_byte)

        return data


@dataclass
class SAOCOMGeneralChannelInfo:
    """SAOCOM general channel info dataclass"""

    channel_id: str
    swath: str
    product_type: SAOCOMProductType
    polarization: SARPolarization
    projection: SARProjection
    acquisition_mode: SAOCOMAcquisitionMode
    acquisition_mode_std: StandardSARAcquisitionMode
    orbit_direction: OrbitDirection
    signal_frequency: float
    acq_start_time: PreciseDateTime

    @staticmethod
    def from_metadata(node: etree._Element, channel_id: str) -> SAOCOMGeneralChannelInfo:
        """Generating SAOCOMGeneralChannelInfo object directly from metadata.

        Parameters
        ----------
        root : etree._Element
            Channel metadata node
        channel_id : str
            channel id

        Returns
        -------
        SAOCOMGeneralChannelInfo
            general channel info dataclass
        """
        projection = SARProjection(node.find("DataSetInfo/Projection").text)
        mode = SAOCOMAcquisitionMode(node.find("DataSetInfo/AcquisitionMode").text)
        mode_std = (
            StandardSARAcquisitionMode.STRIPMAP
            if mode == SAOCOMAcquisitionMode.STRIPMAP
            else StandardSARAcquisitionMode.TOPSAR
        )

        return SAOCOMGeneralChannelInfo(
            channel_id=channel_id,
            swath=node.find("SwathInfo/Swath").text,
            product_type=SAOCOMProductType.SLC if projection == SARProjection.SLANT_RANGE else SAOCOMProductType.GRD,
            polarization=SARPolarization(node.find("SwathInfo/Polarization").text),
            projection=projection,
            acquisition_mode=mode,
            acquisition_mode_std=mode_std,
            orbit_direction=OrbitDirection(node.find("StateVectorData/OrbitDirection").text.lower()),
            signal_frequency=float(node.find("DataSetInfo/fc_hz").text),
            acq_start_time=PreciseDateTime.from_utc_string(node.find("SwathInfo/AcquisitionStartTime").text),
        )


@dataclass
class SAOCOMChannelMetadata:
    """SAOCOM channel metadata dataclass"""

    general_info: SAOCOMGeneralChannelInfo
    orbit: Orbit
    image_radiometric_quantity: SARRadiometricQuantity
    burst_info: BurstInfo
    raster_info: RasterInfo
    dataset_info: DatasetInfo
    swath_info: SwathInfo
    sampling_constants: SARSamplingFrequencies
    doppler_centroid_poly: SortedPolyList
    doppler_rate_poly: SortedPolyList
    pulse: PulseInfo | None
    coordinate_conversions: CoordinatesConversions
    state_vectors: StateVectors


class SAOCOMProduct:
    """SAOCOM product object"""

    def __init__(self, path: str | Path) -> None:
        """SAOCOM Product init from directory path.

        Parameters
        ----------
        path : str | Path
            path to SAOCOM product
        """
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        self._manifest_path = list(self._product_path.glob("*" + _MANIFEST_EXTENSION))
        assert len(self._manifest_path) == 1
        self._manifest = SAOCOMManifest.from_file(self._manifest_path[0])
        self._footprint = self._manifest.footprint

    @property
    def manifest_path(self) -> Path:
        """Manifest .xemt file path"""
        return self._manifest_path

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._manifest.acquisition_start_time

    @property
    def channels_number(self) -> int:
        """Returning the number of channels of SAOCOM product"""
        return len(self._manifest.channels)

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels"""
        return self._manifest.channels

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_files_from_channel_name(self, channel_name: str) -> list[Path]:
        """Get files associated to a given channel name.

        Parameters
        ----------
        channel_name : str
            channel id name

        Returns
        -------
        list[Path]
            path to .xml metadata file and binary file
        """
        return self._manifest.raster_files[channel_name]


@dataclass
class SAOCOMManifest:
    """SAOCOM .xemt manifest class"""

    manifest_path: Path
    channels: list[str]
    polarizations: list[SARPolarization]
    acquisition_start_time: PreciseDateTime
    acquisition_end_time: PreciseDateTime
    raster_files: dict[str, list[Path]]  # for each channel, a list of raster binary file and .xml metadata
    footprint: tuple[float, float, float, float]  # min lat, max lat, min lon, max lon of the scene

    @staticmethod
    def from_file(path: str | Path) -> SAOCOMManifest:
        """Generating a SAOCOMManifest object representing the content of the .xemt file.

        Parameters
        ----------
        path : str | Path
            path to the .xemt file

        Returns
        -------
        SAOCOMManifest
            SAOCOM manifest dataclass
        """
        path = Path(path)
        assert str(path.name).endswith(_MANIFEST_EXTENSION)

        # loading file
        root = etree.parse(path).getroot()
        image_attribute_node = root.find("product/features/imageAttributes")
        components_node = root.find("product/dataFile/components")

        channels = _generate_channels_from_manifest(image_attribute_node=image_attribute_node)
        raster_files_relative = _get_raster_file_paths_from_manifest(components_node=components_node, channels=channels)
        raster_files_full_path = {
            k: [path.parent.joinpath(path.parent.name, vv) for vv in v] for k, v in raster_files_relative.items()
        }

        # recovering scene footprint
        scene_vertices = root.find("product/features/scene/frame").findall("vertex")
        latitudes = [float(f.find("lat").text) for f in scene_vertices]
        longitudes = [float(f.find("lon").text) for f in scene_vertices]
        footprint = (min(latitudes), max(latitudes), min(longitudes), max(longitudes))

        return SAOCOMManifest(
            manifest_path=path,
            acquisition_start_time=PreciseDateTime.fromisoformat(
                root.find("product/features/acquisition/acquisitionTime/startTime").text
            ),
            acquisition_end_time=PreciseDateTime.fromisoformat(
                root.find("product/features/acquisition/acquisitionTime/endTime").text
            ),
            channels=channels,
            polarizations=[
                polarization_dict[p.lower()]
                for p in root.find("product/features/acquisition/parameters/acquiredPols").text.split("-")
            ],
            raster_files=raster_files_full_path,
            footprint=footprint,
        )


def _extract_poly_info_from_node(
    node: etree._Element, doppler_node_tag: str
) -> tuple[list[PreciseDateTime], list[float], list[list[float]]]:
    """Extracting main info from a polynomial metadata node.

    Parameters
    ----------
    node : etree._Element
        Channel metadata node
    doppler_node_tag : str
        polynomial tag to be searched for, it could be:
        "DopplerCentroid", "DopplerRate", "SlantToGround", "GroundToSlant"

    Returns
    -------
    tuple[list[PreciseDateTime], list[float], list[list[float]]]
        reference azimuth times (a list with a time for each polynomial node found),
        reference range times (same),
        polynomial coefficients (same)
    """
    azimuth_ref_times = [PreciseDateTime.from_utc_string(t.text) for t in node.findall(doppler_node_tag + "/taz0_Utc")]
    range_ref_times = [float(t.text) for t in node.findall(doppler_node_tag + "/trg0_s")]
    coefficients = [[float(c.text) for c in poly.findall("val")] for poly in node.findall(doppler_node_tag + "/pol")]

    assert len(azimuth_ref_times) == len(range_ref_times) == len(coefficients)

    return azimuth_ref_times, range_ref_times, coefficients


def _generate_channels_from_manifest(image_attribute_node: etree._Element) -> list[str]:
    """Composing channel names from manifest metadata.

    Channel names convention: swath + '_' + pol

    Parameters
    ----------
    image_attribute_node : etree._Element
        imageAttributes metadata node

    Returns
    -------
    list[str]
        list of channel names
    """
    swath_infos = image_attribute_node.findall("SwathInfo")
    return [s.find("Swath").text.lower() + "_" + s.find("Polarization").text.lower() for s in swath_infos]


def _get_raster_file_paths_from_manifest(components_node: etree._Element, channels: list[str]) -> dict[str, Path]:
    """Retrieve path to .xml and binary files from metadata.

    Parameters
    ----------
    components_node : etree._Element
        components metadata node
    channels : list[str]
        list of channels names

    Returns
    -------
    dict[str, list[Path]]
        each key is a channel name, each value a list of .xml and binary file
    """
    components = components_node.findall("component")
    channel_files_association_dict = dict.fromkeys(channels)

    tag = "Science samples"
    # get metadata files from manifest related to input tag
    metadata_files = [c.find("componentPath").text for c in components if c.find("componentTitle").text == tag]
    metadata_content = [
        c.find("componentContent").text.lower() for c in components if c.find("componentTitle").text == tag
    ]
    # associate each file with the corresponding channel
    for channel in channel_files_association_dict:
        channel_parts = channel.split("_")
        channel_files_association_dict[channel] = np.where(
            [all([c in r for c in channel_parts]) for r in metadata_content]
        )[0][0]
    # creating list of linked binary-metadata file for each metadata file found in manifest
    linked_files = list(
        map(
            list, zip(metadata_files, [b.replace(_RASTER_METADATA_EXTENSION, "") for b in metadata_files], strict=False)
        )
    )

    return {k: linked_files[v] for k, v in channel_files_association_dict.items()}


def is_saocom_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid SAOCOM product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - metadata file exist (.xemt)
        - folder with same name of metadata file exists
        - subfolder Data in the previous folder

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

    # check for metadata file existence
    manifest_file = list(product.glob("*" + _MANIFEST_EXTENSION))
    if len(manifest_file) != 1:
        return False
    prod_name = manifest_file[0].name.strip(_MANIFEST_EXTENSION)
    prod_data_folder = product.joinpath(prod_name)

    if not prod_data_folder.exists() or not prod_data_folder.is_dir():
        return False

    try:
        # loading manifest
        SAOCOMManifest.from_file(manifest_file[0])
    except Exception:
        return False

    return True
