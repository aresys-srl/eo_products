# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
TERRASAR-X product format reader
--------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lxml import etree

import eo_products.terrasarx.raster_reader as raster_reader
import eo_products.terrasarx.utilities as support
from eo_products.common.utilities import (
    BurstInfo,
    SARPolarization,
)


def read_channel_metadata(
    xml_path: str | Path, associated_cos_raster_path: str | Path | None = None
) -> support.TERRASARXChannelMetadata:
    """Generate TERRASARXChannelMetadata object from product metadata

    Parameters
    ----------
    xml_path : str | Path
        path to xml metadata file
    associated_cos_raster_path: str | Path | None
        path to the channel .cos raster, provide None if product is MGD, by default None

    Returns
    -------
    support.TERRASARXChannelMetadata
        TERRASAR-X channel metadata object
    """
    xml_path = Path(xml_path)

    # loading the xml file
    root = etree.parse(str(xml_path)).getroot()
    product_info_node = root.xpath(".//productInfo")[0]

    _, prod_variant, _, acq_mode = support.get_basic_info_from_metadata(xml_path)
    projection = support.get_SARProjection_from_TERRASARXProjection(
        support.TERRASARXProjections(product_info_node.xpath(".//productVariantInfo/projection")[0].text)
    )

    if associated_cos_raster_path:
        associated_cos_raster_path = Path(associated_cos_raster_path)
        beam_id = associated_cos_raster_path.name.split("_", 3)[-1].rsplit(".", 1)[0]
        polarization = SARPolarization("/".join(associated_cos_raster_path.name.split("_")[1]))
    else:
        beam_id = support.generate_channels_names(xml_path)[0].split("_")[0]
        polarization = SARPolarization("/".join(support.generate_channels_names(xml_path)[0].split("_")[1].upper()))

    rad_qt = support.get_radiometric_quantity_from_metadata(root)

    # beam independent info

    dataset_info = support.dataset_info_from_metadata(root, projection, acq_mode, prod_variant)

    state_vectors = support.state_vectors_from_metadata(root)

    attitude = support.attitude_data_from_metadata(root)

    # beam dependent info

    raster_info = support.raster_info_from_metadata(root, projection, beam_id)

    sampling_constants = support.sampling_constants_from_metadata(root, raster_info, beam_id, projection, acq_mode)

    calibration_factor = support.calibration_factor_from_metadata(root, beam_id)

    if prod_variant == support.TERRASARXProductVariant.SSC and acq_mode == support.TERRASARXAcquisitionModes.SCANSAR:
        assert associated_cos_raster_path is not None, "raster file .cos not provided"
        burst_info = support.burst_info_from_raster(raster_info=raster_info, raster_file=associated_cos_raster_path)
    else:
        burst_info = support.burst_info_from_metadata(
            raster_info=raster_info,
        )

    swath_info = support.swath_info_from_metadata(prod_variant, root, beam_id)

    doppler_centroid_poly = support.doppler_centroid_poly_from_metadata(root, beam_id)

    doppler_rate_poly = support.doppler_rate_poly_from_metadata(root, beam_id, prod_variant)

    coordinate_conversion_poly = support.coordinates_conversions_from_metadata(
        prod_variant, root, raster_info, state_vectors, doppler_centroid_poly
    )

    return support.TERRASARXChannelMetadata(
        image_calibration_factor=calibration_factor,
        image_radiometric_quantity=rad_qt,
        polarization=polarization,
        burst_info=burst_info,
        raster_info=raster_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        sampling_constants=sampling_constants,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversion_poly,
        state_vectors=state_vectors,
        attitude=attitude,
    )


def read_product_metadata(xml_path: str | Path) -> dict[str, support.TERRASARXChannelMetadata]:
    """Read all channels' metadata

    Parameters
    ----------
    xml_path : str | Path

    Returns
    -------
    dict[str, support.TERRASARXChannelMetadata]
        channels' metadata dictionary
    """
    channels_list = support.generate_channels_names(xml_path)
    out_dict = dict.fromkeys(channels_list)
    raster_paths = support.generate_raster_paths(xml_path)
    assert len(raster_paths) == len(channels_list), "number of raster files does not match number of channels"
    _, prod_variant, _, _ = support.get_basic_info_from_metadata(xml_path)
    for i, channel_id in enumerate(channels_list):
        if prod_variant == support.TERRASARXProductVariant.MGD:
            out_dict[channel_id] = read_channel_metadata(xml_path, None)
        else:
            out_dict[channel_id] = read_channel_metadata(xml_path, raster_paths[i])

    return out_dict


def open_product(file_path: str | Path) -> support.TERRASARXProduct:
    """Open a TERRASARX product

    Parameters
    ----------
    file_path : str | Path
        Path to the TERRASAR-X product

    Returns
    -------
    TERRASARXProduct
        TERRASARXProduct object corresponding to the input TERRASAR-X product
    """
    if not support.is_terrasarx_product(product=file_path):
        raise support.InvalidTERRASARXProduct(f"{file_path}")
    return support.TERRASARXProduct(path=file_path)


def read_channel_data(
    raster_file: str | Path,
    block_to_read: list[int],
    burst_info: BurstInfo,
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Read channel raster data

    Parameters
    ----------
    raster_file : str | Path
        path to channel raster data
    block_to_read : list[int]
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read
    burst_info : BurstInfo
        burst info object
    scaling_conversion : float, optional
        scaling conversion to be multiplied to the data read (sqrt of calibration factor), by default 1

    Returns
    -------
    np.ndarray
        ROI read from raster

    Raises
    ------
    ValueError
        out of bounds in lines
    ValueError
        out of bounds in samples
    ValueError
        invalid raster file
    """
    raster_file = Path(raster_file)
    if raster_file.suffix == raster_reader.GRD_RASTER_EXTENSION:
        # already scaled and cropped
        return raster_reader.read_tif_raster(raster_file, block_to_read, scaling_conversion)
    elif raster_file.suffix == raster_reader.SLC_RASTER_EXTENSION:
        if burst_info.num > 1:  # multiple bursts per single .cos file
            beam_id = _find_burst_index(burst_info.lines_per_burst, block_to_read[0])
            if block_to_read[0] + block_to_read[2] > np.cumsum(burst_info.lines_per_burst)[beam_id]:
                # out of bounds in lines
                raise ValueError("block_to_read goes out of burst bounds in lines")
            if block_to_read[1] + block_to_read[3] > burst_info.samples_per_burst:
                # out of bounds in samples
                raise ValueError("block_to_read goes out of burst bounds in samples")
            data, _ = raster_reader.read_binary_cos_file(raster_file, beam_id)
            cumulative_index = np.cumsum(burst_info.lines_per_burst)
            cumulative_index = np.insert(cumulative_index, 0, 0)
            relative_line_index_start = np.int64(block_to_read[0] - cumulative_index[beam_id])
            data = data[
                relative_line_index_start : relative_line_index_start + block_to_read[2],
                block_to_read[1] : block_to_read[1] + block_to_read[3],
            ]
        else:
            data, _ = raster_reader.read_binary_cos_file(raster_file)
            data = data[
                block_to_read[0] : block_to_read[0] + block_to_read[2],
                block_to_read[1] : block_to_read[1] + block_to_read[3],
            ]
    else:
        raise ValueError(f"{raster_file} is not a .tif or .cos file")
    return data * scaling_conversion


def _find_burst_index(relative_index: np.array, absolute_index: int) -> int:
    """Get burst_id given an axis index

    Parameters
    ----------
    relative_index : np.array
        relative indexes
    absolute_index : int
        absolute index

    Returns
    -------
    int
        burst_id to which the absolute index belongs to

    Raises
    ------
    ValueError
        Absolute index is out of bounds
    """
    total = 0
    for i, length in enumerate(relative_index):
        total += length
        if absolute_index < total:
            return i
    raise ValueError("Absolute index is out of bounds")


if __name__ == "__main__":
    # prod_path = r"C:\Users\marco.spadoni\Desktop\TDX1_SAR__MGD_RE___SC_S_SRA_20230408T155304_20230408T155326"
    # prod_path = r"C:\Users\marco.spadoni\Desktop\TDX1_SAR__MGD_RE___SC_S_SRA_20240406T153932_20240406T154001"
    # prod_path = r"C:\Users\marco.spadoni\Desktop\TSX1_SAR__SSC______ST_S_SRA_20191215T051104_20191215T051104"
    # prod_path = r"C:\Users\marco.spadoni\Desktop\TDX1_SAR__SSC______SC_S_SRA_20240524T101819_20240524T101830"
    prod_path = r"C:\Users\marco.spadoni\Desktop\TSX_2024\TDX1_SAR__SSC______SC_S_SRA_20240825T052632_20240825T052643"

    prod_path = Path(prod_path)
    prod = open_product(prod_path)
    channels = support.generate_channels_names(prod_path.joinpath(f"{prod_path.name}.xml"))
    channel_id = channels[-1]
    raster_file = prod.get_raster_files_from_channel_name(channel_id)

    print("channels:", channels)
    print("channel ID:", channel_id)
    print("raster file:", raster_file)

    # channel_metadata = read_channel_metadata(prod_path.joinpath(f"{prod_path.name}.xml"), channel_id)
    raster_file = prod.get_raster_files_from_channel_name(channel_id)
    channel_metadata = read_channel_metadata(prod_path.joinpath(f"{prod_path.name}.xml"), raster_file)
    block_to_read = None
    # block_to_read = [0, 0, 10, 10]
    # cal_factor = channel_metadata.image_calibration_factor
    data = (
        np.abs(
            read_channel_data(
                raster_file,
                block_to_read,
                channel_metadata.burst_info,
            )
        )
        ** 2
    )
    import matplotlib.pyplot as plt

    plt.imshow(10 * np.log(data))
    plt.show()
