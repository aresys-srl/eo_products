# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
TERRASAR-X reader support module
--------------------------------
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from arepytools.geometry.direct_geocoding import direct_geocoding_monostatic
from arepytools.geometry.orbit import Orbit
from arepytools.timing.precisedatetime import PreciseDateTime
from lxml import etree
from scipy.constants import speed_of_light

import eo_products.terrasarx.raster_reader as raster_reader
from eo_products.common.utilities import (
    BurstInfo,
    ConversionFunction,
    CoordinatesConversions,
    DatasetInfo,
    DopplerEvaluator,
    OrbitDirection,
    Polynomial,
    RasterInfo,
    RasterInfoAxis,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
    StateVectors,
    SwathInfo,
)

_GEOREF_FILE = "GEOREF.xml"
_RASTER_FOLDER = "IMAGEDATA"


class InvalidTERRASARXProduct(RuntimeError):
    """Invalid TERRASAR-X product"""


class InvalidTERRASARXProjection(RuntimeError):
    """Invalid TERRASAR-X projection"""


class TERRASARXProductVariant(Enum):
    """TERRASAR-X product type"""

    SSC = "SSC"
    MGD = "MGD"
    GEC = "GEC"
    EEC = "EEC"


class TERRASARXAcquisitionModes(Enum):
    """TERRASAR-X L1 acquisition modes"""

    STRIPMAP = "STRIPMAP"
    SCANSAR = "SCANSAR"
    SPOTLIGHT = "SPOTLIGHT"


class TERRASARXProjections(Enum):
    """TERRASAR-X L1 projection types"""

    SLANTRANGE = "SLANTRANGE"
    UNDEFINED = "UNDEFINED"
    GROUNDRANGE = "GROUNDRANGE"
    MAP = "MAP"


def get_basic_info_from_metadata(
    metadata_path: str | Path,
) -> tuple[PreciseDateTime, TERRASARXProductVariant, tuple[float, float, float, float], TERRASARXAcquisitionModes]:
    """Recovering basic product information
    Parameters
    ----------
    metadata_path : str | Path
        path to TERRASAR-X basic product specification xml file

    Returns
    -------
    PreciseDateTime
        acquisition time in PreciseDateTime format
    TERRASARXProductVariant
        product variant
    tuple[float, float, float, float]
        scene footprint in [min lat, max lat, min lon, max lon]
    TERRASARXAcquisitionModes
        acquisition mode
    """
    metadata_path = Path(metadata_path)
    mtd = metadata_path.read_text(encoding="UTF-8")

    # regex init
    imaging_mode_re = re.compile("(?<=<imagingMode>).*(?=</imagingMode>)")
    variant_re = re.compile("(?<=<productVariant>).*(?=</productVariant>)")
    acq_time_re = re.compile("(?<=<startTimeUTC>).*(?=</startTimeUTC>)")
    lat_re = re.compile("(?<=<lat>).*(?=</lat>)")
    lon_re = re.compile("(?<=<lon>).*(?=</lon>)")

    # info extraction
    imaging_mode = imaging_mode_re.findall(mtd)[0].lower()
    acq_variant = variant_re.findall(mtd)[0].lower()
    acq_time = acq_time_re.findall(mtd)[0]

    # recovering scene footprint
    footprint_lat = [float(f) for f in lat_re.findall(mtd)]
    footprint_lon = [float(f) for f in lon_re.findall(mtd)]
    footprint = (min(footprint_lat), max(footprint_lat), min(footprint_lon), max(footprint_lon))

    return (
        PreciseDateTime.fromisoformat(acq_time),
        TERRASARXProductVariant(acq_variant.upper()),
        footprint,
        get_acquisition_mode(imaging_mode),
    )


def raster_info_from_metadata(
    root: etree._Element,
    projection: SARProjection,
    beam_id: str,
) -> RasterInfo:
    """Creating a RasterInfo metadata element from xml node.

    Parameters
    ----------
    root : etree._Element
        root metadata xml node
    projection : SARProjection
        SAR product projection
    beam_id : str
        beam identifier string

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """

    product_info_node = root.xpath(".//productInfo")[0]
    image_data_info_node = product_info_node.xpath(".//imageDataInfo")[0]

    image_raster_nodes = image_data_info_node.findall("imageRaster")
    if len(image_raster_nodes) == 1:
        image_raster_node = image_raster_nodes[0]
    else:
        image_raster_node = [r for r in image_raster_nodes if r.values()[0] == beam_id]
        assert len(image_raster_node) == 1, "none or more than one nodes found"
        image_raster_node = image_raster_node[0]

    # beam dependent info
    if image_raster_node.attrib:
        image_raster_node = image_data_info_node.xpath(f".//imageRaster[@beamID='{beam_id}']")[0]
    n_rows = int(image_raster_node.xpath(".//numberOfRows")[0].text)
    n_cols = int(image_raster_node.xpath(".//numberOfColumns")[0].text)
    col_unit = image_raster_node.xpath(".//columnSpacing")[0].attrib.get("units", None)
    row_spacing = float(image_raster_node.xpath(".//rowSpacing")[0].text)

    file_name = root.xpath(".//productComponents/imageData/file/location/filename")[0].text
    zero_doppler_start_time = PreciseDateTime.fromisoformat(
        product_info_node.xpath(".//sceneInfo/start/timeUTC")[0].text
    )
    zero_doppler_stop_time = PreciseDateTime.fromisoformat(product_info_node.xpath(".//sceneInfo/stop/timeUTC")[0].text)
    row_unit = "s"
    col_spacing = abs(zero_doppler_stop_time - zero_doppler_start_time) / (n_rows - 1)
    raster_rows = RasterInfoAxis(n_rows, col_spacing, zero_doppler_start_time, row_unit)

    if projection == SARProjection.SLANT_RANGE:
        celltype = "FLOAT_COMPLEX"
        range_start = float(product_info_node.xpath(".//sceneInfo/rangeTime/firstPixel")[0].text)
    else:
        celltype = "FLOAT32"
        range_start = 0.0
    raster_cols = RasterInfoAxis(n_cols, row_spacing, range_start, col_unit)

    return RasterInfo(
        lines=raster_rows,
        samples=raster_cols,
        data_type=celltype,
        raster_name=file_name,
    )


def burst_info_from_raster(raster_info: RasterInfo, raster_file: str | Path) -> BurstInfo:
    """Generate burst info from raster file in the case of scansar ssc products"""
    azimuth_start_times = []
    range_start_times = []
    lines_per_burst = []
    _read_burst_info_core(raster_info, raster_file, azimuth_start_times, range_start_times, lines_per_burst)

    return BurstInfo(
        num=len(lines_per_burst),
        samples_per_burst=raster_info.samples.length,
        lines_per_burst=np.array(lines_per_burst),
        azimuth_start_times=np.array(azimuth_start_times),
        range_start_times=np.array(range_start_times),
    )


def _read_burst_info_core(
    raster_info: RasterInfo,
    raster_file: str | Path,
    azimuth_start_times: list[PreciseDateTime],
    range_start_times: list[float],
    lines_per_burst: list[int],
):
    burst_id = 0
    while True:
        try:
            _, burst_annotation = raster_reader.read_binary_cos_file(raster_file, burst_id)
            assert np.unique(burst_annotation.azimuth_sample_relative_index).size == 1, (
                "lines start changes within burst"
            )
            azimuth_start_times.append(raster_info.lines.axis[burst_annotation.azimuth_sample_relative_index[0]])
            range_start_times.append(raster_info.samples.axis[0])
            lines_per_burst.append(burst_annotation.azimuth_samples)
            burst_id = burst_id + 1
        except ValueError:  # exceeded number of bursts
            break


def burst_info_from_metadata(
    raster_info: RasterInfo,
) -> BurstInfo:
    """Get BurstInfo object from metadata file for non scansar ssc products

    Parameters
    ----------
    raster_info : RasterInfo
        product raster info

    Returns
    -------
    BurstInfo
        burst info dataclass
    """
    # single burst
    num = 1
    lines_per_burst = raster_info.lines.length
    samples_per_burst = raster_info.samples.length
    azimuth_start_times = np.array([raster_info.lines.start])
    range_start_times = np.array([raster_info.samples.start])

    return BurstInfo(
        num=num,
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
        azimuth_start_times=azimuth_start_times,
        range_start_times=range_start_times,
    )


def dataset_info_from_metadata(
    root: etree._Element,
    projection: SARProjection,
    acq_mode: TERRASARXAcquisitionModes,
    prod_variant: TERRASARXProductVariant,
) -> DatasetInfo:
    """Creating a DatasetInfo metadata element from xml nodes.

    Parameters
    ----------
    root : etree._Element
        root metadata xml node
    projection : SARProjection
        SAR product projection
    acq_mode : TERRASARXAcquisitionModes
        TERRASAR-X product acquisition mode
    prod_variant : TERRASARXProductVariant
        TERRASAR-X product variant

    Returns
    -------
    DatasetInfo
        DatasetInfo metadata object
    """

    fc_hz = float(root.xpath(".//instrument/radarParameters/centerFrequency")[0].text)
    mission_name = root.xpath(".//productInfo/generationInfo/copyrightInfo")[0].text.split()[0]
    look_direction = root.xpath(".//productInfo/acquisitionInfo/lookDirection")[0].text.lower()
    return DatasetInfo(
        fc_hz=fc_hz,
        acquisition_mode=acq_mode,
        sensor_name=mission_name,
        image_type=prod_variant.value,
        projection=projection.value,
        side_looking=look_direction,
    )


def sampling_constants_from_metadata(
    root: etree._Element,
    raster_info: RasterInfo,
    beam_id: str,
    projection: SARProjection,
    acq_mode: TERRASARXAcquisitionModes,
) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata element from xml nodes

    Parameters
    ----------
    root : etree._Element
        root metadata xml node
    raster_info : RasterInfo
        rasterInfo
    beam_id : str
        beam identifier string
    projection : SARProjection
        SAR product projection
    acq_mode : TERRASARXAcquisitionModes
        TERRASAR-X acquisition mode

    Returns
    -------
    SARSamplingFrequencies
        sampling frequencies
    """
    if projection == SARProjection.SLANT_RANGE:
        processing_node = root.xpath(".//processing")[0]
        processing_parameter_nodes = processing_node.findall("processingParameter")
        if len(processing_parameter_nodes) == 1:
            parameter_node = processing_parameter_nodes[0]
        else:
            parameter_node = [r for r in processing_parameter_nodes if r.xpath("beamID")[0].text == beam_id]
            assert len(parameter_node) == 1, "none or more than one nodes found"
            parameter_node = parameter_node[0]

        if acq_mode == TERRASARXAcquisitionModes.SPOTLIGHT:
            azimuth_bandwidth_freq_hz = float(parameter_node.xpath(".//azimuthLookBandwidth")[0].text)
        else:
            azimuth_bandwidth_freq_hz = float(parameter_node.xpath(".//totalProcessedAzimuthBandwidth")[0].text)
        range_bandwidth_freq_hz = float(parameter_node.xpath(".//totalProcessedRangeBandwidth")[0].text)
        azimuth_freq_hz = 1.0 / raster_info.lines.step
        range_freq_hz = 1.0 / raster_info.samples.step
    else:
        azimuth_bandwidth_freq_hz = 0
        range_bandwidth_freq_hz = 0
        azimuth_freq_hz = 0
        range_freq_hz = 0

    sampling_constants = SARSamplingFrequencies(
        azimuth_bandwidth_freq_hz=azimuth_bandwidth_freq_hz,
        range_bandwidth_freq_hz=range_bandwidth_freq_hz,
        azimuth_freq_hz=azimuth_freq_hz,
        range_freq_hz=range_freq_hz,
    )
    return sampling_constants


def get_acquisition_mode(imaging_mode: str) -> TERRASARXAcquisitionModes:
    """Convert TERRASAR-X imaging mode to acquisition mode

    Parameters
    ----------
    imaging_mode : str
        TERRASAR-X imaging mode

    Returns
    -------
    TERRASARXAcquisitionModes
        acquisition mode among STRIPMAP, SCANSAR, SPOTLIGHT
    """
    imaging_mode = imaging_mode.upper()
    if imaging_mode == "SM":
        return TERRASARXAcquisitionModes.STRIPMAP
    if imaging_mode == "SC":
        return TERRASARXAcquisitionModes.SCANSAR
    return TERRASARXAcquisitionModes.SPOTLIGHT


def get_SARProjection_from_TERRASARXProjection(projection: TERRASARXProjections) -> SARProjection:
    """Convert TERRASAR-X projection to generic SAR projection

    Parameters
    ----------
    projection : TERRASARXProjections
        TERRASAR-X product projection

    Returns
    -------
    SARProjection
        SAR product projection

    Raises
    ------
    InvalidTERRASARXProjection
        Invalid TERRASAR-X product projection
    """
    if projection == TERRASARXProjections.GROUNDRANGE:
        return SARProjection.GROUND_RANGE
    elif projection == TERRASARXProjections.SLANTRANGE:
        return SARProjection.SLANT_RANGE
    else:
        raise InvalidTERRASARXProjection(projection)


def polarization_list_from_metadata(product_info_node: etree._Element) -> list[str]:
    """get polarization list from productInfo metadata xml node

    Parameters
    ----------
    product_info_node : etree._Element
        productInfo metadata xml node

    Returns
    -------
    list[str]
        list of polarization layers
    """
    polarization_list_node = product_info_node.xpath(".//acquisitionInfo/polarisationList")[0]
    return [pol.text for pol in polarization_list_node.findall(".//polLayer")]


def get_radiometric_quantity_from_metadata(root: etree._Element) -> SARRadiometricQuantity | None:
    """Get SARRadiometricQuantity object from metadata

    Parameters
    ----------
    root : etree._Element
        root metadata xml node

    Returns
    -------
    SARRadiometricQuantity | None
        SAR radiometric quantity object
    """
    pixel_value_id = root.xpath(".//imageDataInfo/pixelValueID")[0].text.lower()
    if pixel_value_id == "radar brightness" or pixel_value_id == "beta nought":
        return SARRadiometricQuantity.BETA_NOUGHT
    elif pixel_value_id == "sigma nought":
        return SARRadiometricQuantity.SIGMA_NOUGHT
    elif pixel_value_id == "gamma nought":
        return SARRadiometricQuantity.GAMMA_NOUGHT
    else:
        return None


def generate_channels_names(metadata: str | Path) -> list[str]:
    """Generate list of channels names from metadata xml file

    Parameters
    ----------
    metadata : str | Path
        path to metadata xml file

    Returns
    -------
    list[str]
        list of channels names
    """
    root = etree.parse(metadata).getroot()
    polarizations = polarization_list_from_metadata(root.xpath(".//productInfo")[0])
    beams = get_beams_list_from_metadata(root.xpath(".//productInfo")[0])
    # TODO: checking that only one polarization is available
    assert len(polarizations) == 1, "more than one polarization found"
    return ["_".join([b, polarizations[0].lower()]) for b in beams]


def generate_raster_paths(xml_path: str | Path) -> list[Path]:
    """Generate raster paths from metadata xml file path

    Parameters
    ----------
    xml_path : str | Path
        path to xml metadata file

    Returns
    -------
    list[Path]
        paths of raster files
    """
    xml_path = Path(xml_path)
    root = etree.parse(str(xml_path)).getroot()
    image_data_nodes = root.xpath(".//productComponents/imageData")
    paths = []
    for image_data_node in image_data_nodes:
        filename = image_data_node.xpath(".//file/location/filename")[0].text
        folder = image_data_node.xpath(".//file/location/path")[0].text
        paths.append(Path(xml_path.parent / folder / filename))

    return paths


def get_beams_list_from_metadata(product_info_node: etree._Element) -> list[str]:
    """Generate list of beams identifiers from metadata

    Parameters
    ----------
    product_info_node : etree._Element
        productInfo metadata xml node

    Returns
    -------
    list[str]
        list of beams identifiers
    """
    product_variant = TERRASARXProductVariant(
        product_info_node.xpath(".//productVariantInfo/productVariant")[0].text.upper()
    )
    acq_mode = get_acquisition_mode(product_info_node.xpath(".//acquisitionInfo/imagingMode")[0].text.upper())
    if product_variant == TERRASARXProductVariant.SSC and acq_mode == TERRASARXAcquisitionModes.SCANSAR:
        beam_list_node = product_info_node.xpath(".//acquisitionInfo/imagingModeSpecificInfo/scanSAR/beamList")[0]
        beam_IDs = [beam.text for beam in beam_list_node.findall(".//beamID")]
    else:
        beam_IDs = ["s"]
    return beam_IDs


def calibration_factor_from_metadata(root: etree._Element, beam_id: str) -> float:
    """get calibration factor from metadata xml file

    Parameters
    ----------
    root : etree._Element
        root metadata xml node
    beam_id: str
        beam identifier string

    Returns
    -------
    float
        calibration factor
    """
    cal_node = root.xpath(".//calibration")[0]
    cal_constant_nodes = cal_node.findall(".//calibrationConstant")
    if len(cal_constant_nodes) == 1:
        cal_constant_node = cal_constant_nodes[0]
    else:
        cal_constant_node = [r for r in cal_constant_nodes if r.xpath(".//beamID")[0].text == beam_id]
        assert len(cal_constant_node) == 1, "none or more than one nodes found"
        cal_constant_node = cal_constant_node[0]

    return np.sqrt(float(cal_constant_node.xpath(".//calFactor")[0].text))


def swath_info_from_metadata(prod_variant: TERRASARXProductVariant, root: etree._Element, beam_id: str) -> SwathInfo:
    """Creating a SwathInfo metadata object from metadata file.

    Parameters
    ----------
    prod_variant : TERRASARXProductVariant
        TERRASAR-X product variant
    root : etree._Element
        root metadata xml node
    beam_id : str
        beam identifier string

    Returns
    -------
    SwathInfo
        SwathInfo metadata object
    """
    rank = 0
    acquisition_prf = 0
    swath = "s"

    if prod_variant == TERRASARXProductVariant.SSC:
        instrument_node = root.xpath(".//instrument")[0]
        settings_nodes = instrument_node.findall(".//settings")
        if len(settings_nodes) == 1:
            settings_node = settings_nodes[0]
        else:
            settings_node = [r for r in settings_nodes if r.xpath(".//beamID")[0].text == beam_id]
            assert len(settings_node) == 1, "none or more than one nodes found"
            settings_node = settings_node[0]
            swath = beam_id

        acquisition_prf = float(settings_node.xpath(".//settingRecord/PRF")[0].text)
        rank = int(settings_node.xpath(".//settingRecord/echoIndex")[0].text)

    return SwathInfo(
        swath=swath,
        rank=rank,
        azimuth_steering_rate_poly=(0, 0, 0),
        prf=acquisition_prf,
    )


def state_vectors_from_metadata(root: etree._Element) -> StateVectors:
    """Generating StateVectors object directly from product metadata.

    Parameters
    ----------
    root : etree._Element
        root metadata xml node

    Returns
    -------
    StateVectors
        orbit's state vectors dataclass
    """

    orbit_data_node = root.xpath(".//platform//orbit")[0]
    product_info_node = root.xpath(".//productInfo")[0]

    state_vecs = orbit_data_node.findall(".//stateVec")

    pos_x = np.array([float(x.find("posX").text) for x in state_vecs])
    pos_y = np.array([float(y.find("posY").text) for y in state_vecs])
    pos_z = np.array([float(z.find("posZ").text) for z in state_vecs])
    positions = np.stack([pos_x, pos_y, pos_z], axis=1)

    vel_x = np.array([float(x.find("velX").text) for x in state_vecs])
    vel_y = np.array([float(y.find("velY").text) for y in state_vecs])
    vel_z = np.array([float(z.find("velZ").text) for z in state_vecs])
    velocities = np.stack([vel_x, vel_y, vel_z], axis=1)

    time_axis = np.array([PreciseDateTime.fromisoformat(str(t.find("timeUTC").text)) for t in state_vecs])
    orbit_dir = product_info_node.xpath(".//missionInfo/orbitDirection")[0].text.lower()
    assert positions.shape[0] == velocities.shape[0] == time_axis.size
    return StateVectors(
        num=time_axis.size,
        positions=positions,
        velocities=velocities,
        time_axis=time_axis,
        time_step=np.mean(np.diff(time_axis)),
        orbit_direction=OrbitDirection(orbit_dir),
    )


def doppler_centroid_poly_from_metadata(root: etree._Element, beam_id: str) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    root : etree._Element
        root metadata xml node
    beam_id : str
        beam identifier string

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Centroid polynomial
    """
    acq_mode = get_acquisition_mode(root.xpath(".//productInfo/acquisitionInfo/imagingMode")[0].text)
    prod_variant = get_acquisition_mode(root.xpath(".//productInfo/productVariantInfo/productVariant")[0].text)
    doppler_poly_list = []
    doppler_node = root.xpath(".//processing/doppler")[0]
    doppler_centroid_nodes = doppler_node.findall(".//dopplerCentroid")
    if len(doppler_centroid_nodes) == 1:
        doppler_centroid_node = doppler_centroid_nodes[0]
    # TODO: manage multiple strips! Here we just take the first.
    elif acq_mode == TERRASARXAcquisitionModes.SCANSAR and prod_variant != TERRASARXProductVariant.SSC:
        doppler_centroid_node = doppler_centroid_nodes[0]
    else:
        doppler_centroid_node = [r for r in doppler_centroid_nodes if r.xpath(".//beamID")[0].text == beam_id]
        assert len(doppler_centroid_node) == 1, "none or more than one nodes found"
        doppler_centroid_node = doppler_centroid_node[0]

    doppler_estimate_nodes = doppler_centroid_node.findall(".//dopplerEstimate")
    for item in doppler_estimate_nodes:
        ref_point = float(item.xpath(".//combinedDoppler/referencePoint")[0].text)
        coefficients = [float(c.text) for c in item.xpath(".//combinedDoppler/coefficient")]
        azimuth_reference_time = PreciseDateTime.fromisoformat(item.xpath(".//timeUTC")[0].text)
        doppler_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=azimuth_reference_time,
                origin=ref_point,
                function=Polynomial(coefficients),
            )
        )
    return DopplerEvaluator(
        functions=doppler_poly_list,
        azimuth_reference_times=np.array([c.azimuth_reference_time for c in doppler_poly_list]),
    )


def doppler_rate_poly_from_metadata(
    root: etree._Element, beam_id: str, prod_variant: TERRASARXProductVariant
) -> DopplerEvaluator:
    """Creating a DopplerEvaluator doppler rate vector polynomial wrapper from metadata.

    Parameters
    ----------
    root : etree._Element
        root metadata xml node
    beam_id : str
        beam identifier string
    prod_variant: TERRASARXProductVariant
        TERRASAR-X product variant

    Returns
    -------
    DopplerEvaluator
        DopplerEvaluator dataclass for Doppler Rate polynomial
    """
    doppler_rate_nodes = root.xpath(".//processing/geometry")[0].findall(".//dopplerRate")
    valid_doppler_rate_nodes = []
    for node in doppler_rate_nodes:
        dr = node.find(".//beamID")
        if prod_variant == TERRASARXProductVariant.SSC:
            if dr is not None:
                if dr.text == beam_id:
                    valid_doppler_rate_nodes.append(node)
            else:
                valid_doppler_rate_nodes.append(node)
        else:
            valid_doppler_rate_nodes.append(node)

    doppler_poly_list = []
    for item in valid_doppler_rate_nodes:
        azimuth_reference_time = PreciseDateTime.fromisoformat(item.xpath(".//timeUTC")[0].text)
        ref_point = float(item.xpath(".//dopplerRatePolynomial/referencePoint")[0].text)
        coefficients = [float(c.text) for c in item.xpath(".//dopplerRatePolynomial/coefficient")]
        doppler_poly_list.append(
            ConversionFunction(
                azimuth_reference_time=azimuth_reference_time,
                origin=ref_point,
                function=Polynomial(coefficients),
            )
        )
    return DopplerEvaluator(
        functions=doppler_poly_list,
        azimuth_reference_times=np.array([c.azimuth_reference_time for c in doppler_poly_list]),
    )


def coordinates_conversions_from_metadata(
    prod_variant: TERRASARXProductVariant,
    root: etree._Element,
    raster_info: RasterInfo,
    state_vectors: StateVectors,
    doppler_centroid_poly: DopplerEvaluator,
) -> CoordinatesConversions:
    """Generating CoordinateConversions object from metadata

    Parameters
    ----------
    prod_variant : TERRASARXProductVariant
        TERRASAR-X product variant
    root : etree._Element
        root metadata xml node
    raster_info : RasterInfo
        product raster info
    state_vectors: StateVectors
        product state vectors
    doppler_centroid_poly: DopplerEvaluator
        product doppler centroid evaluator

    Returns
    -------
    CoordinatesConversions
        polynomial for coordinate conversion dataclass
    """
    slant_to_ground_conversion = (None,)
    azimuth_ref_time = (None,)
    if prod_variant == TERRASARXProductVariant.MGD:  # slant to ground poly provided
        projected_image_info_node = root.xpath(".//productSpecific/projectedImageInfo")[0]
        coefficients = [
            float(c.text) for c in projected_image_info_node.xpath(".//slantToGroundRangeProjection/coefficient")
        ]
        slant_to_ground_poly = Polynomial(coef=coefficients)
        ref_point = float(projected_image_info_node.xpath(".//referencePoint")[0].text)
        azimuth_ref_time = PreciseDateTime.fromisoformat(
            projected_image_info_node.xpath(".//mappingGridInfo/gridReferenceTime/tReferenceTimeUTC")[0].text
        )
        slant_to_ground_conversion = ConversionFunction(
            azimuth_reference_time=azimuth_ref_time, origin=ref_point, function=slant_to_ground_poly
        )
        # ground to slant is not given, so it must be evaluated by inverting the ground to slant poly
        first_pixel_time = float(root.xpath(".//productInfo/sceneInfo/rangeTime/firstPixel")[0].text)
        last_pixel_time = float(root.xpath(".//productInfo/sceneInfo/rangeTime/lastPixel")[0].text)
        range_axis = np.linspace(first_pixel_time, last_pixel_time, raster_info.samples.length)
        slant_to_ground_poly_evaluated = slant_to_ground_poly(range_axis - ref_point)
        ground_to_slant_poly = Polynomial.fit(
            x=slant_to_ground_poly_evaluated, y=range_axis, deg=slant_to_ground_poly.degree()
        )
        ground_to_slant_conversion = ConversionFunction(
            azimuth_reference_time=azimuth_ref_time, origin=0, function=ground_to_slant_poly
        )

        return CoordinatesConversions(
            slant_to_ground=[slant_to_ground_conversion],
            ground_to_slant=[ground_to_slant_conversion],
            azimuth_reference_times=np.array([azimuth_ref_time]),
        )
    else:  # no coefficient is given
        mid_azimuth = raster_info.lines.start + raster_info.lines.length * raster_info.lines.step / 2
        range_times = np.arange(0, raster_info.samples.length, 1) * raster_info.samples.step + raster_info.samples.start
        orbit = Orbit(state_vectors.time_axis, state_vectors.positions, state_vectors.velocities)
        fc_hz = float(root.xpath(".//instrument/radarParameters/centerFrequency")[0].text)
        ground_points = direct_geocoding_monostatic(
            sensor_positions=orbit.evaluate(mid_azimuth),
            sensor_velocities=orbit.evaluate_first_derivatives(mid_azimuth),
            range_times=range_times,
            frequencies_doppler_centroid=doppler_centroid_poly.evaluate(raster_info.lines.start, range_times),
            wavelength=speed_of_light / fc_hz,
            geocoding_side=root.xpath(".//productInfo/acquisitionInfo/lookDirection")[0].text.upper(),
            geodetic_altitude=0,
        )
        ground_points_distances = np.linalg.norm(np.diff(ground_points, axis=0), axis=1)
        ground_range_axis = np.r_[[0], np.cumsum(ground_points_distances)]
        slant_to_ground_poly = Polynomial.fit(x=range_times, y=ground_range_axis, deg=8)
        ground_to_slant_poly = Polynomial.fit(x=ground_range_axis, y=range_times, deg=8)

        ground_to_slant_conversion = ConversionFunction(
            azimuth_reference_time=raster_info.lines.start, origin=0, function=ground_to_slant_poly
        )
        slant_to_ground_conversion = ConversionFunction(
            azimuth_reference_time=raster_info.lines.start, origin=0, function=slant_to_ground_poly
        )

        return CoordinatesConversions(
            slant_to_ground=[slant_to_ground_conversion],
            ground_to_slant=[ground_to_slant_conversion],
            azimuth_reference_times=np.array([raster_info.lines.start]),
        )


@dataclass
class TERRASARXAttitude:
    """TERRASAR-X sensor's attitude"""

    num: int  # attitude data numerosity (same as interpolated orbit)
    q0: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    q3: np.ndarray
    time_axis: np.ndarray  # PreciseDateTime axis to which attitude data applies
    time_step: float  # time axis step


@dataclass
class TERRASARXChannelMetadata:
    image_calibration_factor: float
    image_radiometric_quantity: SARRadiometricQuantity
    polarization: str
    burst_info: BurstInfo
    raster_info: RasterInfo
    dataset_info: DatasetInfo
    swath_info: SwathInfo
    sampling_constants: SARSamplingFrequencies
    doppler_centroid_poly: DopplerEvaluator
    doppler_rate_poly: DopplerEvaluator
    coordinate_conversions: CoordinatesConversions
    state_vectors: StateVectors
    attitude: TERRASARXAttitude


class TERRASARXProduct:
    """TERRASAR-X product object"""

    def __init__(self, path: str | Path) -> None:
        """TERRASAR-X product init from directory path"""
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        self._metadata_path = self._product_path.joinpath(f"{self._product_path.name}.xml")
        self._acq_time, self._product_variant, self._footprint, self._imaging_mode = get_basic_info_from_metadata(
            self._metadata_path
        )
        self._channels_names = generate_channels_names(self._metadata_path)
        self._data_list = list(self._product_path.joinpath(_RASTER_FOLDER).iterdir())

    def get_raster_files_from_channel_name(self, channel_name: str) -> Path:
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
        if len(self._data_list) > 1:
            return [r for r in self._data_list if "_".join(channel_name.split("_")[:-1]).lower() in r.name.lower()][0]
        return self._data_list[0]

    @property
    def channels_number(self) -> int:
        """Returning the number of channels of TERRASAR-X product"""
        return len(self._channels_names)

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels in terms of SwathID (swath-polarization)"""
        return self._channels_names

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def acquisition_mode(self) -> TERRASARXAcquisitionModes:
        """Acquisition mode for this product"""
        return get_acquisition_mode(self._imaging_mode)

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    @property
    def metadata_file(self) -> Path:
        """Returning the product metadata file path of TERRASAR-X product"""
        return self._metadata_path


def is_terrasarx_product(product: str | Path) -> bool:
    """Check if input path corresponds to a valid TERRASAR-X product, basic version.

    Conditions to be met for basic validity:
        - path is dir
        - GEOREF.xml file exists
        - product.html file exists
        - read basic info from product.html file

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

    if not product.joinpath("ANNOTATION", _GEOREF_FILE):
        return False

    metadata = product.joinpath(f"{product.name}.xml")
    if not metadata.exists():
        return False

    try:
        get_basic_info_from_metadata(metadata_path=metadata)
    except Exception:
        return False

    return True


def attitude_data_from_metadata(root: etree._Element) -> TERRASARXAttitude:
    """Creating a TERRASARXAttitude metadata element from xml node

    Parameters
    ----------
    root : etree._Element
        root metadata xml node

    Returns
    -------
    TERRASARXAttitude
        TERRASAR-X attitude using quaternions
    """

    attitude_data_node = root.xpath(".//platform//attitude")[0]
    time_step = float(attitude_data_node.xpath(".//attitudeHeader/attitudeDataTimeSpacing")[0].text)
    attitude_data = attitude_data_node.findall(".//attitudeData")
    q0 = np.array([float(q.find("q0").text) for q in attitude_data])
    q1 = np.array([float(q.find("q1").text) for q in attitude_data])
    q2 = np.array([float(q.find("q2").text) for q in attitude_data])
    q3 = np.array([float(q.find("q3").text) for q in attitude_data])
    time_axis = np.array([PreciseDateTime.fromisoformat(str(t.find("timeUTC").text)) for t in attitude_data])
    assert len(q0) == len(q1) == len(q2) == len(q3) == len(time_axis)
    num = len(q0)
    return TERRASARXAttitude(num=num, q0=q0, q1=q1, q2=q2, q3=q3, time_axis=time_axis, time_step=time_step)
