# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
RADARSAT-2 product format reader
--------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from lxml import etree
from tifffile import imread

import eo_products.radarsat.utilities as support
from eo_products.common.utilities import SARRadiometricQuantity


def read_product_metadata(
    xml_path: str | Path, beta_calibration_xml: str | Path | None
) -> dict[str, support.RADARSATChannelMetadata]:
    """Read RADARSAT-2 product metadata.

    Parameters
    ----------
    xml_path : str | Path
        Path to the .xml metadata file
    beta_calibration_xml : str | Path | None
        Path to the beta calibration .xml metadata file

    Returns
    -------
    dict[str, support.RADARSATChannelMetadata]
        dictionary with channel id as keys and RADARSATChannelMetadata dataclass as values
    """
    xml_path = Path(xml_path)

    _, product_type, channels_list, _ = support.get_basic_info_from_metadata(metadata_path=xml_path)
    acquisition_mode = support.get_acquisition_mode_from_product_type(prod_type=product_type)
    projection = support.get_projection_from_product_type(prod_type=product_type)
    out_dict = dict.fromkeys(channels_list)

    # loading the xml file
    root = etree.parse(xml_path).getroot()
    namespace = {"base" if k is None else k: v for k, v in root.nsmap.items()}
    source_attributes_node = root.xpath(".//base:sourceAttributes", namespaces=namespace)[0]
    image_generation_parameters_node = root.xpath(".//base:imageGenerationParameters", namespaces=namespace)[0]
    image_attributes_node = root.xpath(".//base:imageAttributes", namespaces=namespace)[0]
    orbit_data_node = source_attributes_node.xpath(".//base:orbitInformation", namespaces=namespace)[0]
    attitude_data_node = source_attributes_node.xpath(".//base:attitudeInformation", namespaces=namespace)[0]

    # CHANNEL INDEPENDENT INFO
    # raster info
    raster_info = support.raster_info_from_metadata(
        image_generation_parameters_node=image_generation_parameters_node,
        image_attributes_node=image_attributes_node,
        projection=projection,
        namespace=namespace,
    )
    pixel_ordering = support.RADARSATTimeOrdering(
        image_attributes_node.xpath(".//base:rasterAttributes/base:pixelTimeOrdering", namespaces=namespace)[0].text
    )
    lines_ordering = support.RADARSATTimeOrdering(
        image_attributes_node.xpath(".//base:rasterAttributes/base:lineTimeOrdering", namespaces=namespace)[0].text
    )

    # dataset info
    dataset_info = support.dataset_info_from_metadata(
        source_attributes_node=source_attributes_node,
        namespace=namespace,
        acq_mode=acquisition_mode,
        projection=projection,
    )

    # sampling constants
    sampling_constants = support.sampling_constants_from_metadata(
        image_generation_parameters_node=image_generation_parameters_node,
        namespace=namespace,
        raster_info=raster_info,
        projection=projection,
    )

    # state vectors and orbit
    state_vectors = support.state_vectors_from_metadata(orbit_data_node=orbit_data_node, namespace=namespace)

    # attitude
    attitude = support.RADARSATAttitude.from_metadata(attitude_data_node=attitude_data_node, namespace=namespace)

    # image calibration factor
    image_calibration_factor = support.calibration_factor_from_metadata(
        file_path=beta_calibration_xml, product_type=product_type
    )

    # burst info
    burst_info = support.burst_info_from_metadata(raster_info=raster_info)

    # swath info
    swath_info = support.swath_info_from_metadata(
        source_attributes_node=source_attributes_node, namespace=namespace, prod_type=acquisition_mode
    )

    # doppler centroid polynomial
    doppler_centroid_poly = support.doppler_centroid_poly_from_metadata(
        doppler_centroid_node=image_generation_parameters_node.xpath(".//base:dopplerCentroid", namespaces=namespace)[
            0
        ],
        namespace=namespace,
    )

    # doppler rate polynomial
    doppler_rate_poly = support.doppler_rate_poly_from_metadata(
        doppler_rate_node=image_generation_parameters_node.xpath(".//base:dopplerRateValues", namespaces=namespace)[0],
        namespace=namespace,
    )

    # coordinates conversion polynomials
    coordinates_conversions_poly = support.coordinates_conversions_from_metadata(
        image_generation_parameters_node=image_generation_parameters_node, namespace=namespace, raster_info=raster_info
    )

    # CHANNEL DEPENDENT INFO
    # NOTE: actually there is no dependence from channel
    for channel in channels_list:
        # general info
        general_info = support.RADARSATGeneralChannelInfo.from_metadata_node(
            root=root, namespace=namespace, name=xml_path.parent.name, channel_id=channel
        )

        out_dict[channel] = support.RADARSATChannelMetadata(
            general_info=general_info,
            image_calibration_factor=image_calibration_factor,
            image_radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
            raster_info=raster_info,
            burst_info=burst_info,
            dataset_info=dataset_info,
            swath_info=swath_info,
            doppler_centroid_poly=doppler_centroid_poly,
            doppler_rate_poly=doppler_rate_poly,
            coordinate_conversions=coordinates_conversions_poly,
            orbit=state_vectors.orbit,
            state_vectors=state_vectors,
            attitude=attitude,
            sampling_constants=sampling_constants,
            lines_ordering=lines_ordering,
            samples_ordering=pixel_ordering,
        )

    return out_dict


def read_channel_data(
    raster_file: str | Path,
    block_to_read: list[int] | None = None,
    scaling_conversion: float = 1,
) -> np.ndarray:
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
    if target_area.ndim == 3:
        # complex data to be re-arranged
        target_area = target_area[:, :, 0] + 1j * target_area[:, :, 1]

    return target_area * scaling_conversion


def open_product(pf_path: str | Path) -> support.RADARSATProduct:
    """Open a RADARSAT-2 product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the RADARSAT-2 product

    Returns
    -------
    RADARSATProduct
        RADARSATProduct object corresponding to the input RADARSAT-2 product
    """
    if not support.is_radarsat_product(product=pf_path):
        raise support.InvalidRADARSATProduct(f"{pf_path}")

    return support.RADARSATProduct(path=pf_path)
