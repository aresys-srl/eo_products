# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
EOS04 product format reader
---------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from lxml import etree
from tifffile import imread

import eo_products.eos04.utilities as support
from eo_products.common.utilities import SARRadiometricQuantity


def read_product_metadata(xml_path: str | Path, channels: list[str]) -> dict[str, support.EOS04ChannelMetadata]:
    """Reading EOS04 product channel metadata.

    Parameters
    ----------
    xml_path : str | Path
        path to the annotation xml file
    channels : list[str]
        channels ids for the current product

    Returns
    -------
    dict[str, support.EOS04ChannelMetadata]
        dictionary of EOS04ChannelMetadata dataclasses as values, channel name as key
    """

    xml_path = Path(xml_path)

    # loading the xml file
    root = etree.parse(xml_path).getroot()
    product_type = root.find("ProductType").text
    source_attributes_node = root.find("SourceAttributes")
    orbit_node = source_attributes_node.find("OrbitAndAttitude/OrbitInformation")
    attitude_node = source_attributes_node.find("OrbitAndAttitude/AttitudeInformation")
    image_generation_node = root.find("ImageGenerationParameters")
    image_attributes_node = root.find("ImageAttributes")

    # channel independent info
    state_vectors = support.state_vectors_from_metadata(orbit_information_node=orbit_node)
    attitude = support.EOS04Attitude.from_metadata_node(attitude_information_node=attitude_node)
    # forcing radiometric input as Beta Nought
    radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT

    channels_dict = dict.fromkeys(channels)
    for channel in channels:
        beam, polarization = support.unpack_channel_name(channel)

        # general info
        general_info = support.EOS04GeneralChannelInfo.from_metadata_node(
            source_attributes_node=source_attributes_node, channel_id=channel, product_type=product_type
        )

        # dataset info
        dataset_info = support.dataset_info_from_metadata_node(
            source_attributes_node=source_attributes_node, projection=general_info.projection
        )

        # swath info
        swath_info = support.swath_info_from_metadata(
            image_generation_parameters_node=image_generation_node,
            polarization=polarization,
            beam=beam,
            product_type=general_info.product_type,
        )

        # raster info
        raster_info = support.raster_info_from_metadata_nodes(
            image_generation_parameters_node=image_generation_node,
            image_attributes_node=image_attributes_node,
            beam_id=beam,
            polarization=polarization,
            product_type=general_info.product_type,
        )

        # burst info
        burst_info = support.burst_info_from_metadata(
            image_generation_parameters_node=image_generation_node,
            polarization=polarization,
            beam_id=beam,
            raster_info=raster_info,
            product_type=product_type,
        )

        # pulse info (no chirp direction)
        pulse = support.pulse_info_from_metadata_nodes(
            source_attributes_node=source_attributes_node, samples_step=raster_info.samples.step
        )

        # doppler centroid poly
        doppler_centroid_poly = support.doppler_centroid_poly_from_metadata_node(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # doppler rate poly
        doppler_rate_poly = support.doppler_rate_poly_from_metadata_node(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # coordinate conversion
        coordinate_conversion = support.coordinates_conversions_from_metadata(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # sampling constants
        sampling_constants = support.sampling_constants_from_metadata(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # image calibration factor
        calibration_constant_db = [
            float(c.text)
            for c in image_attributes_node.findall("CalibrationConstant_Beta0")
            if c.get("pol") == polarization.name
        ][0]
        calibration_factor = 1 / (10 ** (calibration_constant_db / 20))

        channels_dict[channel] = support.EOS04ChannelMetadata(
            channel_id=channel,
            general_info=general_info,
            raster_info=raster_info,
            burst_info=burst_info,
            state_vectors=state_vectors,
            orbit=state_vectors.orbit,
            dataset_info=dataset_info,
            sampling_constants=sampling_constants,
            swath_info=swath_info,
            attitude=attitude,
            image_calibration_factor=calibration_factor,
            image_radiometric_quantity=radiometric_quantity,
            doppler_centroid_poly=doppler_centroid_poly,
            doppler_rate_poly=doppler_rate_poly,
            coordinate_conversions=coordinate_conversion,
            pulse=pulse,
        )

    return channels_dict


def read_channel_data(
    raster_file: str | Path,
    block_to_read: list[int] | None = None,
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Reading EOS04 tif channel data file.

    Parameters
    ----------
    raster_file : str | Path
        Path to .tif raster file to be read
    block_to_read : list[int], optional
        data block to be read, to be specified as a list of 4 integers, in the form:
            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        by default None

    scaling_conversion : float, optional
        scaling conversion to be multiplied to the data read

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

    # SLC image is a two page tif, with real and imaginary part that must be recombined
    if target_area.ndim == 3:
        target_area = target_area[:, :, 0] + 1j * target_area[:, :, 1]

    # applying input scaling factor
    target_area = target_area * scaling_conversion

    return target_area


def open_product(path: str | Path) -> support.EOS04Product:
    """Open an EOS04 product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the EOS04 product

    Returns
    -------
    EOS04Product
        EOS04Product object corresponding to the input product
    """
    path = Path(path)

    if not support.is_eos04_product(product=path):
        raise support.InvalidEOS04Product(f"{path}")

    return support.EOS04Product(path=path)
