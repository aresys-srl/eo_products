# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Synspective StriX L1 product format reader
------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import zarr
from sarpy.io.complex.sicd import SICDReader

import eo_products.strix.utilities as support
from eo_products.common.utilities import LookingDirection, OrbitDirection, RasterInfo, SwathInfo

# NOTE: notes on synspective assessment
# 1. GRD products always single channel?
# 2. GRD products do not have S2G/G2S polynomial conversions
# 3. SLC products do not have S2G/G2S polynomial conversions, or at least are strange
# 4. SLC doppler rate is a scale factor, to what?
# 5. SLC doppler centroid poly is in azimuth time, not range time
# 6. GRD products do not have doppler rate polynomials and doppler centroid polynomials
# 7. SLC raster is flipped in azimuth "The image is oriented shadows downward and a view from above the earth.", always?
# 8. GRD timeline is not clear, we start from center scene azimuth time and go back half lines value * sat velocity at
# mid scene
# 9. GRD need reference to slant range axis to at least be able to perform geocoding


def read_channel_metadata(file_path: Path | str, channel_id: str) -> support.StriXChannelMetadata:
    """Reading channel metadata info and storing them in a StriXChannelMetadata dataclass.

    The assumption here is that .xml files are associated ONLY to GRD (GeoTiFF raster file), as reported in the official
    documentation, while NITF file is SLC only.

    Parameters
    ----------
    file_path : Path | str
        Path to the metadata file, could be an .xml (GRD) or a .nitf file (SLC)
    channel_id : str
        channel id

    Returns
    -------
    support.StriXChannelMetadata
        StriXChannelMetadata metadata dataclass
    """
    file_path = Path(file_path)
    is_nitf = bool(str(file_path).endswith(support._SLC_DATA_EXTENSION))
    product_name = (
        file_path.name.replace(support._GRD_METADATA_PREFIX, support._GRD_DATA_PREFIX).replace(
            support._GRD_METADATA_EXTENSION, support._GRD_DATA_EXTENSION
        )
        if not is_nitf
        else file_path.name
    )

    # loading file root assuming that .hd5 file is SLC and .xml file is GRD
    if is_nitf:
        root_ = SICDReader(str(file_path))
        root = root_.sicd_meta
    else:
        root = file_path.read_text(encoding="UTF-8")

    # state vectors
    state_vectors = support.state_vectors_from_metadata(root=root)

    # general info
    general_info = support.StriXGeneralChannelInfo.from_metadata(
        root=root, product_name=product_name, orbit=state_vectors.orbit, channel_id=channel_id
    )
    general_info.orbit_direction = state_vectors.orbit_direction

    # raster info
    raster_info = support.raster_info_from_metadata(root=root, orbit=state_vectors.orbit)

    # burst info
    burst_info = support.burst_info_from_metadata(raster_info=raster_info)

    # dataset info
    dataset_info = support.dataset_info_from_metadata(root=root)

    # swath info
    swath_info = SwathInfo(
        swath=channel_id.split("_")[0],
        rank=0,
        azimuth_steering_rate_poly=(0, 0, 0),
        prf=1 / raster_info.lines.step,
    )

    # sampling constants
    sampling_constants = support.sampling_constants_from_metadata(root=root, lines_step=raster_info.lines.step)

    # calibration factor and radiometric quantity
    calibration_factor, radiometric_quantity = support.get_calibration_factor_and_quantity_from_metadata(root=root)

    # doppler centroid polynomial
    doppler_centroid_poly = support.doppler_centroid_poly_from_metadata_node(root=root, raster_info=raster_info)

    # doppler rate polynomial
    doppler_rate_poly = support.doppler_rate_poly_from_metadata(
        root=root, raster_info=raster_info, orbit=state_vectors.orbit
    )

    # coordinate conversions
    coordinate_conversions = support.coordinates_conversions_from_metadata(
        root=root, raster_info=raster_info, orbit=state_vectors.orbit
    )

    if is_nitf:
        # closing nitf file
        root_.close()

    return support.StriXChannelMetadata(
        general_info=general_info,
        orbit=state_vectors.orbit,
        image_calibration_factor=calibration_factor,
        image_radiometric_quantity=radiometric_quantity,
        burst_info=burst_info,
        raster_info=raster_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        sampling_constants=sampling_constants,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversions,
        state_vectors=state_vectors,
    )


def read_channel_data(
    raster_file: str | Path,
    raster_info: RasterInfo,
    orbit_direction: OrbitDirection,
    looking_side: LookingDirection,
    block_to_read: list[int] = None,
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Reading StriX data file. It can be a GeoTiff .tif file (for GRD products) or an NITF .nitf file (for SLC).

    Parameters
    ----------
    raster_file : str | Path
        Path to .tif or .nitf file
    raster_info : RasterInfo
        channel raster info
    orbit_direction : OrbitDirection
        orbit direction
    looking_side : LookingDirection
        looking side
    block_to_read : list[int] | None, optional
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None
    scaling_conversion : float, optional
        scaling conversion to be applied to the data read

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    raster_file = Path(raster_file)
    nitf_flag = bool(str(raster_file).endswith(support._SLC_DATA_EXTENSION))

    if nitf_flag:
        # SLC case
        match (looking_side, orbit_direction):
            case ("LEFT", OrbitDirection.ASCENDING):
                target_area = _data_reader_core(
                    raster_file=raster_file, block_to_read=block_to_read, raster_info=raster_info, flip=True
                )
            case ("LEFT", OrbitDirection.DESCENDING):
                target_area = _data_reader_core(
                    raster_file=raster_file, block_to_read=block_to_read, raster_info=raster_info, flip=True
                )
            case ("RIGHT", OrbitDirection.ASCENDING):
                target_area = _data_reader_core(
                    raster_file=raster_file, block_to_read=block_to_read, raster_info=raster_info, flip=False
                )
            case ("RIGHT", OrbitDirection.DESCENDING):
                target_area = _data_reader_core(
                    raster_file=raster_file, block_to_read=block_to_read, raster_info=raster_info, flip=False
                )
            case _:
                raise ValueError(f"Invalid combination: {orbit_direction.name}, {looking_side}")

    else:
        # GRD case
        img_store = tifffile.imread(raster_file, aszarr=True)
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
    return target_area * scaling_conversion


def _data_reader_core(
    raster_file: str | Path, block_to_read: list[int], raster_info: RasterInfo, flip: bool
) -> np.ndarray:
    """Core data reader function for .nitf SLC files.

    Parameters
    ----------
    raster_file : str | Path
        Path .nitf file
    block_to_read : list[int]
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None
    raster_info : RasterInfo
        channel raster info
    flip : bool
        whether to flip the data along azimuth axis

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    dataset = SICDReader(str(raster_file))
    if flip:
        if block_to_read is None:
            # NOTE: data is stored as (rng, az), so needs to be transposed, and then flipped along azimuth axis
            target_area = np.flip(dataset.read(), axis=1).T
        else:
            # NOTE: data is stored as (rng, az), so needs to be transposed, and then flipped along azimuth axis
            start_az = int(raster_info.lines.length - block_to_read[0])
            target_area = dataset.read(
                (block_to_read[1], block_to_read[1] + block_to_read[3], 1),
                (start_az - block_to_read[2], start_az, 1),
            )
            target_area = np.flip(target_area, axis=1).T
    else:
        if block_to_read is None:
            # NOTE: data is stored as (rng, az), so needs to be transposed,
            target_area = dataset.read().T
        if block_to_read is not None:
            # NOTE: data is stored as (rng, az), so needs to be transposed,
            target_area = dataset.read(
                (block_to_read[1], block_to_read[1] + block_to_read[3], 1),
                (block_to_read[0], block_to_read[0] + block_to_read[2], 1),
            ).T
    dataset.close()
    return target_area


def open_product(pf_path: str | Path) -> support.StriXProduct:
    """Open a StriX product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the StriX product

    Returns
    -------
    StriXProduct
        StriXProduct object corresponding to the input StriX product
    """

    if not support.is_strix_product(product=pf_path):
        raise support.InvalidStriXProduct(f"{pf_path}")

    return support.StriXProduct(path=pf_path)


if __name__ == "__main__":
    prod = open_product(
        pf_path=r"C:\Users\giorgio.parma\Aresys_DATA\sct_data\synspective\stripmap\IMG-VV-STRIX3-20240603T115211Z-SMSLC-SICD.nitf"
    )
    read_channel_metadata(channel_id=prod.channels_list[0], file_path=prod._product_path)
    ...
