# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
COSMO product format reader
---------------------------
"""

from __future__ import annotations

from pathlib import Path

import h5py

import eo_products.cosmo.utilities as support
from eo_products.common.utilities import SARRadiometricQuantity


def read_channel_metadata(file_path: str | Path, channel_id: str) -> support.COSMOChannelMetadata:
    """Reading channel metadata info and storing them in a COSMOChannelMetadata dataclass.

    Parameters
    ----------
    file_path : str | Path
        Path to the .h5 product
    channel_id : str
        selected channel id

    Returns
    -------
    support.COSMOChannelMetadata
        COSMOChannelMetadata metadata dataclass
    """

    root = h5py.File(file_path)
    root_attributes = root.attrs
    swath = root[channel_id.split("_")[0]]
    swath_attributes = swath.attrs
    raster = support.get_raster(root, channel_id)
    raster_attributes = raster.attrs

    # general info
    general_info = support.COSMOGeneralChannelInfo.from_metadata(root=root, channel_id=channel_id)

    # creating raster info
    raster_info = support.raster_info_from_metadata(root=root, channel_id=channel_id)

    # dataset info
    dataset_info = support.dataset_info_from_metadata(root_attributes=root_attributes)

    # state vectors
    state_vectors = support.state_vectors_from_metadata(root_attributes=root_attributes)

    # pulse info
    pulse_info = support.pulse_info_from_metadata(swath_attributes=swath_attributes)

    # sampling constants
    sampling_constants = support.sampling_constants_from_metadata(
        swath_attributes=swath_attributes,
        raster_info=raster_info,
    )

    # calibration factor
    calibration_factor = support.compute_calibration_factor(root=root, channel_id=channel_id)

    # swath info
    swath_info = support.swath_info_from_metadata(root=root, channel_id=channel_id)

    # burst info
    burst_info = support.burst_info_from_metadata(
        raster_info=raster_info, range_ref_time=raster_attributes["Zero Doppler Range First Time"]
    )

    # doppler polynomials
    doppler_rate_poly = support.doppler_rate_poly_from_metadata(root=root, channel_id=channel_id)
    doppler_centroid_poly = support.doppler_centroid_poly_from_metadata(root=root, channel_id=channel_id)

    # coordinate conversions
    range_step_m = (
        raster_info.samples.step
        if general_info.product_level == support.COSMOProductType.DGM
        else raster_info.samples.step / support.METERS_TO_SECONDS_CONVERSION
    )
    coordinate_conversions = support.coordinates_conversions_from_metadata(
        root_attributes=root_attributes,
        azimuth_ref=raster_info.lines.start,
        range_step_m=range_step_m,
        projection=general_info.projection,
    )

    root.close()

    return support.COSMOChannelMetadata(
        general_info=general_info,
        raster_info=raster_info,
        orbit=state_vectors.orbit,
        state_vectors=state_vectors,
        image_calibration_factor=calibration_factor,
        image_radiometric_quantity=SARRadiometricQuantity.SIGMA_NOUGHT,
        dataset_info=dataset_info,
        swath_info=swath_info,
        burst_info=burst_info,
        sampling_constants=sampling_constants,
        pulse=pulse_info,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversions,
    )


def read_channel_data(
    raster_file: str | Path,
    channel_id: str,
    block_to_read: list[int] | None = None,
    scaling_conversion: float = 1,
):
    """Reading COSMO data file as HDF5 .h5 file.

    Parameters
    ----------
    raster_file : str | Path
        Path to HDF5 .h5 file
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
    root = h5py.File(raster_file)
    raster = support.get_raster(root, channel_id)
    if block_to_read is not None:
        target_area = raster[
            block_to_read[0] : block_to_read[0] + block_to_read[2],
            block_to_read[1] : block_to_read[1] + block_to_read[3],
        ]
    else:
        target_area = raster[()]

    if len(target_area.shape) == 3:
        # With three dimension, last axis is real + imaginary
        target_area = target_area[:, :, 0] + 1j * target_area[:, :, 1]

    return target_area * scaling_conversion


def open_product(pf_path: str | Path) -> support.COSMOProduct:
    """Open a COSMO product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the COSMO product

    Returns
    -------
    COSMOProduct
        COSMOProduct object corresponding to the input COSMO product
    """

    if not support.is_cosmo_product(product=pf_path):
        raise support.InvalidCOSMOProduct(f"{pf_path}")

    return support.COSMOProduct(path=pf_path)
