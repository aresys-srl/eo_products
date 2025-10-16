# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Sentinel-1 SAFE product format reader
-------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from arepytools.timing.precisedatetime import PreciseDateTime
from lxml import etree
from tifffile import imread

import eo_products.sentinel1.utilities as support
from eo_products.common.utilities import SARRadiometricQuantity, StateVectors


def read_external_orbit(
    xml_path: str | Path, time_boundaries: tuple[PreciseDateTime, PreciseDateTime] | None = None
) -> StateVectors:
    """Reading SAFE product external orbit file. External orbits can be very large and span over several hours so time
    boundaries can be provided to limit the orbit extent just around the acquisition time interval.

    Parameters
    ----------
    xml_path : str | Path
        path to the external orbit file
    time_boundaries : tuple[PreciseDateTime, PreciseDateTime] | None, optional
        time boundaries as start time [0] and stop time [1], by default None

    Returns
    -------
    StateVectors
        StateVectors dataclass from orbit
    """

    xml_path = Path(xml_path)

    # loading the xml file
    root = etree.parse(xml_path).getroot()

    # finding nodes
    fixed_header = root.find("Earth_Explorer_Header/Fixed_Header")
    variable_header = root.find("Earth_Explorer_Header/Variable_Header")
    validity_period = fixed_header.find("Validity_Period")
    state_vectors_list = root.find("Data_Block/List_of_OSVs")

    # assessing reference frame
    reference_frame = variable_header.find("Ref_Frame").text
    if reference_frame == "EARTH_FIXED":
        reference_frame = support.S1ReferenceFrameType.EARTH_FIXED

    # retrieving datetimes information
    validity_start = PreciseDateTime.fromisoformat(validity_period.find("Validity_Start").text.split("=")[-1])
    validity_stop = PreciseDateTime.fromisoformat(validity_period.find("Validity_Stop").text.split("=")[-1])

    if time_boundaries is not None:
        if time_boundaries[0] < validity_start or time_boundaries[1] > validity_stop:
            raise support.InvalidTimeBoundaries(
                f"Time boundaries {time_boundaries} are out of validity period {(validity_start, validity_stop)}"
            )

    # retrieving state vectors data
    state_vectors_num = int(state_vectors_list.get("count"))
    positions = np.zeros((state_vectors_num, 3))
    velocities = np.zeros((state_vectors_num, 3))
    times_for_assessing_delta = []
    for idx, item in enumerate(state_vectors_list):
        positions[idx] = [float(item.find("X").text), float(item.find("Y").text), float(item.find("Z").text)]
        velocities[idx] = [float(item.find("VX").text), float(item.find("VY").text), float(item.find("VZ").text)]
        if idx < 2:
            # taking the first two state vectors reference times to compute the time delta for the time axis
            times_for_assessing_delta.append(PreciseDateTime.from_utc_string(item.find("UTC").text.replace("UTC=", "")))

    # computing time axis
    time_delta = times_for_assessing_delta[1] - times_for_assessing_delta[0]
    # using the first state vector reference time as start time for the time axis
    time_axis = np.arange(state_vectors_num) * time_delta + times_for_assessing_delta[0]

    # orbit type
    orbit_type = fixed_header.find("File_Type").text
    if orbit_type == "AUX_PREORB":
        orbit_type = support.S1OrbitType.PREDICTED
    elif orbit_type == "AUX_RESORB":
        orbit_type = support.S1OrbitType.RESTITUTED
    elif orbit_type == "AUX_POEORB":
        orbit_type = support.S1OrbitType.PRECISE

    if time_boundaries is None:
        return StateVectors(
            num=state_vectors_num,
            time_axis=time_axis,
            positions=positions,
            velocities=velocities,
            frame=reference_frame,
            time_step=time_delta,
            orbit_type=orbit_type,
        )

    # cropping orbit only between provided time boundaries
    start_index = np.where(time_axis > time_boundaries[0])[0][0]
    stop_index = np.where(time_axis > time_boundaries[1])[0][0]
    time_axis = time_axis[start_index:stop_index]
    positions = positions[start_index:stop_index]
    velocities = velocities[start_index:stop_index]
    state_vectors_num = stop_index - start_index

    return StateVectors(
        num=state_vectors_num,
        time_axis=time_axis,
        positions=positions,
        velocities=velocities,
        reference_frame=reference_frame,
        time_step=time_delta,
        orbit_type=orbit_type,
    )


def read_channel_metadata(
    xml_path: str | Path, external_orbit_path: str | Path | None = None
) -> support.S1ChannelMetadata:
    """Reading SAFE product channel metadata.

    Parameters
    ----------
    xml_path : str | Path
        path to the channel metadata xml file
    external_orbit_path : str | Path | None, optional
        path to the external orbit xml file, by default None

    Returns
    -------
    S1ChannelMetadata
        channel metadata dataclass
    """

    xml_path = Path(xml_path)
    external_orbit_path = Path(external_orbit_path) if external_orbit_path is not None else None

    # loading the xml file
    root = etree.parse(xml_path).getroot()

    # gathering nodes from metadata
    header = root.find("adsHeader")
    general_annotation = root.find("generalAnnotation")
    product_info = general_annotation.find("productInformation")
    general_info = support.S1GeneralChannelInfo.from_metadata_node(header_node=header, product_info_node=product_info)
    image_annotation = root.find("imageAnnotation")
    image_information = image_annotation.find("imageInformation")
    antenna_pattern = root.find("antennaPattern")
    downlink_info = general_annotation.find("downlinkInformationList/downlinkInformation")
    dc_estimate = root.find("dopplerCentroid/dcEstimateList")
    swath_processing = root.find("imageAnnotation/processingInformation/swathProcParamsList/swathProcParams")
    swath_timing = root.find("swathTiming")
    orbit_list = general_annotation.find("orbitList")
    attitude_list = general_annotation.find("attitudeList")
    azimuth_fm_rate = general_annotation.find("azimuthFmRateList")
    coord_conversion_node = root.find("coordinateConversion/coordinateConversionList")
    replica_list_info = general_annotation.find("replicaInformationList")
    noise_list = general_annotation.find("noiseList")
    antenna_pattern_list = antenna_pattern.find("antennaPatternList")

    # orbit
    if external_orbit_path is None:
        orbit_type = None
        if image_annotation.find("processingInformation/orbitSource").text == "Extracted":
            orbit_type = support.S1OrbitType.DOWNLINK
        elif image_annotation.find("processingInformation/orbitSource").text == "Auxiliary":
            orbit_type = support.S1OrbitType.UNKNOWN
        state_vectors = support.state_vectors_from_metadata(orbit_node=orbit_list, orbit_type=orbit_type)
    else:
        # reducing orbit span around start and end time of acquisition
        # 5 mins before start time and after stop time
        state_vectors = read_external_orbit(
            xml_path=external_orbit_path, time_boundaries=[general_info.start_time - 300, general_info.stop_time + 300]
        )

    # updating orbit direction and type
    current_ind = int((general_info.start_time - state_vectors.time_axis[0]) / state_vectors.time_step)
    state_vectors.orbit_direction = (
        support.OrbitDirection.ASCENDING
        if state_vectors.velocities[current_ind][2] > 0
        else support.OrbitDirection.DESCENDING
    )

    # attitude
    attitude = support.S1Attitude.from_metadata_node(attitude_node=attitude_list)

    # raster info
    if general_info.product_type == support.S1L1ProductType.GRD:
        # GRD
        raster_info = support.raster_info_from_metadata_node(image_information_node=image_information)
    else:
        # SLC
        raster_info = support.raster_info_from_metadata_node(
            image_information_node=image_information, samples_step=1 / general_info.range_sampling_rate
        )

    # burst info
    burst_info = support.burst_info_from_metadata(burst_node=swath_timing, samples_start=raster_info.samples.start)

    # dataset info
    dataset_info = support.dataset_info_from_metadata_nodes(header_node=header, product_info_node=product_info)

    # swath info
    swath_info = support.swath_info_from_metadata(
        product_info=product_info, downlink_info_node=downlink_info, swath=general_info.swath
    )

    # sampling constants
    sampling_constants = support.sampling_constants_from_metadata_nodes(
        swath_processing_node=swath_processing, product_info_node=product_info, image_info_node=image_information
    )

    # acquisition timeline
    acquisition_timeline = support.acquisition_timeline_from_metadata_nodes(downlink_info_node=downlink_info)

    # doppler centroid vector
    # TODO: check if forcing estimate method to be DATA is correct
    doppler_centroid_poly = support.doppler_centroid_evaluator_from_metadata_nodes(
        dc_estimate_node=dc_estimate, estimate_method=support.S1DCEstimateMethod.DATA
    )

    # doppler rate vector
    doppler_rate_poly = support.doppler_rate_evaluator_from_metadata_nodes(azimuth_fm_rate_node=azimuth_fm_rate)

    # pulse
    pulse = support.S1Pulse.from_metadata_nodes(
        swath_processing_node=swath_processing, downlink_info_node=downlink_info
    )

    # coordinates conversion
    coords_conversion = support.coordinates_conversions_from_metadata(coord_conversion_node=coord_conversion_node)

    # chirp replica
    chirp_replicas = []
    for item in replica_list_info:
        chirp_replicas.append(support.S1ChirpReplica.from_metadata_node(replica_info_node=item))
    chirp_replicas = {c.swath: c for c in chirp_replicas}

    # noise
    noise = support.noise_from_metadata_nodes(noise_list_node=noise_list)

    # antenna pattern
    antenna = support.antenna_pattern_from_metadata_nodes(antenna_pattern_list_node=antenna_pattern_list)

    # composing output channel dataclass
    return support.S1ChannelMetadata(
        general_info=general_info,
        orbit=state_vectors.orbit,
        attitude=attitude,
        burst_info=burst_info,
        raster_info=raster_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        sampling_constants=sampling_constants,
        acquisition_timeline=acquisition_timeline,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        pulse=pulse,
        coordinate_conversions=coords_conversion,
        state_vectors=state_vectors,
        chirp_replica=chirp_replicas,
        noise=noise,
        antenna_pattern=antenna,
    )


def read_channel_calibration(
    xml_path: str | Path, radiometric_quantity: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT
) -> float:
    """Reading SAFE product channel calibration metadata.

    Parameters
    ----------
    xml_path : str | Path
        path to the channel calibration xml file
    radiometric_quantity : SARRadiometricQuantity, optional
        radiometric quantity selected, by default SARRadiometricQuantity.BETA_NOUGHT

    Returns
    -------
    float
        radiometric data scaling factor
    """
    radiometric_quantity = SARRadiometricQuantity(radiometric_quantity.value)

    xml_path = Path(xml_path)

    # loading the xml file
    root = etree.parse(xml_path).getroot()

    cal_vector_node = root.find("calibrationVectorList")
    return support.data_scaling_factor_from_metadata_node(
        calibration_vector_node=cal_vector_node, radiometric_quantity=radiometric_quantity
    )


def read_channel_data(
    raster_file: str | Path, block_to_read: list[int] = None, scaling_conversion: float = 1
) -> np.ndarray:
    """Reading SAFE tiff channel data file.

    Parameters
    ----------
    raster_file : str | Path
        Path to Tiff raster file to be read
    block_to_read : list[int], optional
        data block to be read, to be specified as a list of 4 integers, in the form:
            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        by default None
    scaling_conversion : float, optional
        scaling conversion to be applied to the data read

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    img_store = imread(raster_file, aszarr=True)
    z = zarr.open(img_store, mode="r")
    if block_to_read is None:
        target_area = z[:]
    else:
        target_area = z[
            block_to_read[0] : block_to_read[0] + block_to_read[2],
            block_to_read[1] : block_to_read[1] + block_to_read[3],
        ]
    img_store.close()

    # applying input scaling factor
    target_area = target_area * scaling_conversion

    return target_area


def open_product(pf_path: str | Path) -> support.S1Product:
    """Open a SAFE product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the SAFE product

    Returns
    -------
    S1Product
        S1Product object corresponding to the input SAFE product
    """

    pf_path = Path(pf_path)
    if not pf_path.exists() or not pf_path.is_dir():
        raise support.InvalidSAFEProduct(f"{pf_path}")

    if not pf_path.name.endswith(".SAFE"):
        raise support.InvalidSAFEProduct(f"{pf_path}")

    return support.S1Product(path=pf_path)
