# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""ICEYE product format reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tifffile
import zarr
from lxml import etree

import eo_products.iceye.cog_utilities as cog_support
import eo_products.iceye.raster as raster_support
import eo_products.iceye.utilities as support


def read_channel_metadata(file_path: Path | str, channel_id: str) -> support.ICEYEChannelMetadata:
    """Reading channel metadata info and storing them in a ICEYEChannelMetadata dataclass.

    The assumption here is that .xml files are associated ONLY to GRD, as reported in the official documentation
    while HDF5 file is SLC only. STAC .json files are associated to SLC-COG and GRD-COG products.

    note: AUXILIARY .XML METADATA FILE OPTIONALLY AVAILABLE FOR SLC WILL NOT BE USED

    Parameters
    ----------
    file_path : Path | str
        Path to the metadata file, could be an .xml (GRD), a .h5 file (SLC) or a .json file (SLC-COG, GRD-COG)
    channel_id : str
        selected channel id

    Returns
    -------
    support.ICEYEChannelMetadata
        ICEYEChannelMetadata metadata dataclass
    """
    file_path = Path(file_path)
    if file_path.suffix == ".json":
        return cog_support.read_cog_channel_metadata(metadata_file=file_path, channel_id=channel_id)

    hdf5_flag = bool(str(file_path).endswith(".h5"))

    # loading file root assuming that .hd5 file is SLC and .xml file is GRD
    root = h5py.File(file_path) if hdf5_flag else etree.parse(file_path).getroot()

    # general info
    general_info = support.ICEYEGeneralChannelInfo.from_metadata(root=root, channel_id=channel_id)

    # raster info
    raster_info = support.raster_info_from_metadata(root=root)

    # burst info
    burst_info = support.burst_info_from_metadata(raster_info=raster_info)

    # dataset info
    dataset_info = support.dataset_info_from_metadata(root=root)

    # swath info
    swath_info = support.swath_info_from_metadata(root=root)

    # doppler centroid polynomial
    doppler_centroid_poly = support.doppler_centroid_poly_from_metadata(root=root, raster_info=raster_info)

    # doppler rate vector
    doppler_rate_poly = support.doppler_rate_poly_from_metadata(root=root, raster_info=raster_info)

    # incidence angles polynomial
    incidence_angle_poly = support.ICEYEIncidenceAnglePolynomial.from_metadata(root=root)

    # sampling constants
    sampling_constants = support.sampling_constants_from_metadata(root=root)

    # pulse
    pulse = support.pulse_info_from_metadata(root=root)

    # image calibration factor
    calibration_factor, radiometric_quantity = support.calibration_factor_and_radiometric_quantity_from_metadata(
        root=root
    )

    # state vectors
    state_vectors = support.state_vectors_from_metadata(root=root)

    # coordinates conversion
    coordinate_conversions = support.coordinates_conversions_from_metadata(root=root)

    # raster layout, defined only for shadows-down legacy GRD rasters
    raster_layout = support.raster_layout_from_metadata(root=root)

    if hdf5_flag:
        # closing hdf5 file
        root.close()

    return support.ICEYEChannelMetadata(
        general_info=general_info,
        image_calibration_factor=calibration_factor,
        image_radiometric_quantity=radiometric_quantity,
        orbit=state_vectors.orbit,
        state_vectors=state_vectors,
        raster_info=raster_info,
        burst_info=burst_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversions,
        incidence_angles_poly=incidence_angle_poly,
        sampling_constants=sampling_constants,
        pulse=pulse,
        raster_layout=raster_layout,
    )


def read_channel_data(
    raster_file: str | Path,
    block_to_read: list[int] = None,
    scaling_conversion: float = 1,
    raster_layout: support.ICEYERasterLayout | None = None,
) -> np.ndarray:
    """Reading ICEYE data file. It can be a GeoTiff .tif file (for GRD products), an HDF5 .h5 file (for SLC) or a
    Cloud Optimized GeoTiff .tif file (for SLC-COG and GRD-COG products). Shadows-down rasters (COG products and
    shadows-down legacy GRD products) require their raster_layout.

    Parameters
    ----------
    raster_file : str | Path
        Path to GeoTiff .tif or HDF5 .h5 file
    block_to_read : list[int] | None, optional
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None
    scaling_conversion : float, optional
        scaling conversion to be applied to the data read
    raster_layout : support.ICEYERasterLayout | None, optional
        raster layout of shadows-down rasters (ICEYEChannelMetadata.raster_layout), None for native rasters,
        by default None

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    raster_file = Path(raster_file)
    if raster_layout is not None:
        # shadows-down rasters (COG, shadows-down legacy GRD): decoded and reoriented to (lines, samples)
        target_area = raster_support.read_shadows_down_raster(
            raster_file=raster_file, raster_layout=raster_layout, block_to_read=block_to_read
        )
        return target_area * scaling_conversion

    hdf5_flag = bool(str(raster_file).endswith(".h5"))

    if hdf5_flag:
        # SLC case
        dataset = h5py.File(raster_file)
        if block_to_read is None:
            data_real = dataset["s_i"][()]
            data_imaginary = dataset["s_q"][()]
        else:
            data_real = dataset["s_i"][
                block_to_read[0] : block_to_read[0] + block_to_read[2],
                block_to_read[1] : block_to_read[1] + block_to_read[3],
            ]
            data_imaginary = dataset["s_q"][
                block_to_read[0] : block_to_read[0] + block_to_read[2],
                block_to_read[1] : block_to_read[1] + block_to_read[3],
            ]
        target_area = data_real + 1j * data_imaginary

        dataset.close()
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


def open_product(pf_path: str | Path) -> support.ICEYEProduct:
    """Open a ICEYE product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the ICEYE product

    Returns
    -------
    ICEYEProduct
        ICEYEProduct object corresponding to the input ICEYE product
    """

    if not support.is_iceye_product(product=pf_path):
        raise support.InvalidICEYEProduct(f"{pf_path}")

    return support.ICEYEProduct(path=pf_path)
