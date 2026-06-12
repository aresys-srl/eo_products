# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""SAOCOM product format reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lxml import etree

import eo_products.saocom.utilities as support
from eo_products.common.utilities import RasterInfo, SARRadiometricQuantity


def read_channel_metadata(file_path: Path | str, channel_id: str) -> support.SAOCOMChannelMetadata:
    """Reading channel metadata info and storing them in a SAOCOMChannelMetadata dataclass.

    Parameters
    ----------
    file_path : Path | str
        Path to the metadata .xml file
    Returns
    -------
    support.SAOCOMChannelMetadata
        SAOCOMChannelMetadata metadata dataclass
    """
    file_path = Path(file_path)
    root = etree.parse(file_path).getroot()
    root = root.find("Channel")

    # general info
    general_info = support.SAOCOMGeneralChannelInfo.from_metadata(node=root, channel_id=channel_id)

    # raster info
    raster_info, binary_ordering_mode, header_offset, row_prefix = support.raster_info_from_metadata(
        node=root.find("RasterInfo")
    )

    # burst info
    burst_info = support.burst_info_from_metadata(node=root.find("BurstInfo"), raster_info=raster_info)

    # dataset info
    dataset_info = support.dataset_info_from_metadata(node=root.find("DataSetInfo"))

    # swath info
    swath_info = support.swath_info_from_metadata(node=root.find("SwathInfo"))

    # sampling constants
    sampling_constants = support.sampling_constants_from_metadata(node=root.find("SamplingConstants"))

    # pulse
    pulse = support.pulse_from_metadata(node=root.find("Pulse"))

    # state vectors
    state_vectors = support.state_vectors_from_metadata(node=root.find("StateVectorData"))

    # doppler centroid polynomial
    doppler_centroid_poly = support.doppler_poly_from_metadata(node=root, doppler_node_tag="DopplerCentroid")

    # doppler rate vector
    doppler_rate_poly = support.doppler_poly_from_metadata(node=root, doppler_node_tag="DopplerRate")

    # coordinates conversion
    coordinate_conversions = support.coordinates_conversions_from_metadata(node=root)

    return support.SAOCOMChannelMetadata(
        image_radiometric_quantity=SARRadiometricQuantity.SIGMA_NOUGHT,
        general_info=general_info,
        raster_info=raster_info,
        orbit=state_vectors.orbit,
        burst_info=burst_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversions,
        sampling_constants=sampling_constants,
        pulse=pulse,
        state_vectors=state_vectors,
        binary_ordering_mode=binary_ordering_mode,
        header_offset=header_offset,
        row_prefix=row_prefix,
    )


def read_channel_data(
    raster_file: str | Path,
    raster_info: RasterInfo,
    binary_ordering_mode: str,
    header_offset: int,
    row_prefix: int,
    block_to_read: list[int] = None,
) -> np.ndarray:
    """Reading SAOCOM channel raster files with raster info.

    Parameters
    ----------
    raster_file : str | Path
        Path to binary raster file to be read
    raster_info : RasterInfo
        channel raster info
    binary_ordering_mode : str
        binary ordering mode corresponding to the raster itself
    header_offset : int
        header offset of the raster file
    row_prefix : int
        row prefix of the raster file
    block_to_read : list[int] | None, optional
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    return support.read_raster(
        raster_file_name=Path(raster_file),
        num_of_samples=raster_info.samples.length,
        num_of_lines=raster_info.lines.length,
        data_type=raster_info.data_type,
        block_to_read=block_to_read,
        binary_ordering_mode=binary_ordering_mode,
        header_offset=header_offset,
        row_prefix=row_prefix,
    )


def open_product(pf_path: str | Path) -> support.SAOCOMProduct:
    """Open a SAOCOM product.

    NOTE: Saocom product must be provided as a path to a folder with a given name (i.e. L1_A_SLC) containing a .xemt
    metadata file with the same name (i.e. L1_A_SLC.xemt) and another subfolder (with the same name, L1_A_SLC)
    containing [Data, Config, Images, ...]

    Parameters
    ----------
    pf_path : str | Path
        Path to the SAOCOM product

    Returns
    -------
    SAOCOMProduct
        SAOCOMProduct object corresponding to the input SAOCOM product
    """

    if not support.is_saocom_product(product=pf_path):
        raise support.InvalidSAOCOMProduct(f"{pf_path}")

    return support.SAOCOMProduct(path=pf_path)
