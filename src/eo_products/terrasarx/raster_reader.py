# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""TERRASAR-X raster reader support module."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import zarr
from tifffile import imread

HEADER_SIZE_BYTES = 15
C_SAR_ID_HEX = "43534152"
SUPPORTED_VERSIONS = (1, 2)
TANDEM_X_FORMAT_PARAMETERS = [1, 5, 10]
HEADER_FILLER_VALUE = int("7f7f7f7f", 16)
GRD_RASTER_EXTENSION = ".tif"
SLC_RASTER_EXTENSION = ".cos"


class InvalidBinaryCosFile(RuntimeError):
    """Invalid binary .cos file"""


@dataclass
class BurstAnnotation:
    """Burst header annotation"""

    bytes_in_burst: int
    rng_sample_relative_index: int
    range_samples: int
    azimuth_samples: int
    burst_index: int
    oversampling_factor: int  # can be 1, 2, 3
    inverse_specan_scaling_rate: str
    azimuth_sample_relative_index: np.ndarray
    azimuth_sample_first_valid: np.ndarray
    azimuth_sample_last_valid: np.ndarray


def _compute_burst_annotation_bytes_size(width: int) -> int:
    return width * 4 * 4  # the second *4 is due to the uint32 size in bytes


def _compute_burst_data_bytes_size(width: int, lines: int) -> int:
    return width * lines * 4  # the *4 is due to the uint32 size in bytes


def _cos_file_header_reader(filename: Path) -> tuple[int, int, int]:
    """Reader of .cos binary header file. Implemented in Python and translated from MATLAB (see cosFileHeader.m).

    Original MATLAB code:

    David Young (2025). TerraSAR-X and TanDEM-X tools
    (https://it.mathworks.com/matlabcentral/fileexchange/45956-terrasar-x-and-tandem-x-tools),
    MATLAB Central File Exchange.

    Parameters
    ----------
    filename : Path
        Path to .cos binary file

    Returns
    -------
    int
        width of the file in 32-bit samples, i.e. the number of complex samples per range line plus 2
    int
        number of lines in the file including annotation lines
    int
        data version, 1 for 16-bit integer TerraSAR-X, 2 for 16-bit half-precision TanDEM-X
    """

    file_size = filename.stat().st_size

    # read header uint32 values (little-endian), swapping bytes endianness after
    header = np.fromfile(filename, dtype="<u4", count=HEADER_SIZE_BYTES).byteswap()

    # C-SAR ID check
    id_hex = f"{header[7]:08X}"
    if id_hex != C_SAR_ID_HEX:
        raise InvalidBinaryCosFile("C-SAR identifier not found")

    width_samples = int(header[2]) + 2  # width in samples
    range_total_number_of_bytes = int(header[5])
    if 4 * width_samples != range_total_number_of_bytes:
        raise InvalidBinaryCosFile(
            f"Inconsistent width in samples ({width_samples}) and bytes ({range_total_number_of_bytes})"
        )

    height_samples = int(header[6])  # height in samples
    if range_total_number_of_bytes * height_samples != file_size:
        raise InvalidBinaryCosFile(
            f"File size ({file_size}) inconsistent: width ({width_samples}), height ({height_samples})"
        )

    version = int(header[8])
    if version not in SUPPORTED_VERSIONS:
        raise InvalidBinaryCosFile(f"Version number {version} not in {SUPPORTED_VERSIONS}")

    if version == 2:
        format_parameters = header[12:15]
        if not np.array_equal(format_parameters, np.array(TANDEM_X_FORMAT_PARAMETERS, dtype=np.uint32)):
            raise InvalidBinaryCosFile(
                f"Expecting {TANDEM_X_FORMAT_PARAMETERS} format parameters, found {format_parameters.tolist()}"
            )
    return width_samples, height_samples, version


def _cos_burst_annotation_reader(filename: Path, bytes_offset: int, width: int) -> BurstAnnotation:
    """Reader of .cos binary file burst annotation. Implemented in Python and translated from MATLAB
    (see cosBurstHeader.m).

    Original MATLAB code:

    David Young (2025). TerraSAR-X and TanDEM-X tools
    (https://it.mathworks.com/matlabcentral/fileexchange/45956-terrasar-x-and-tandem-x-tools),
    MATLAB Central File Exchange.

    Parameters
    ----------
    filename : Path
        Path to .cos binary file
    bytes_offset : int
        offset in bytes from the beginning of the file
    width : int
        width of the file in 32-bit samples

    Returns
    -------
    BurstAnnotation
        Burst annotation from header
    """

    burst_annotation_count = width * 4
    # read burst annotation uint32 values (little-endian), swapping bytes endianness after
    burst_header = (
        np.fromfile(
            filename,
            dtype="<u4",
            count=burst_annotation_count,
            offset=bytes_offset,
        )
        .reshape((width, 4), order="F")
        .byteswap()
    )

    if not np.all(burst_header[0:2, 1:4] == HEADER_FILLER_VALUE):
        raise InvalidBinaryCosFile("Incorrect filler value found in burst header")

    return BurstAnnotation(
        bytes_in_burst=burst_header[0, 0],
        rng_sample_relative_index=burst_header[1, 0],
        range_samples=burst_header[2, 0],
        azimuth_samples=burst_header[3, 0],
        burst_index=burst_header[4, 0],
        oversampling_factor=burst_header[9, 0],
        inverse_specan_scaling_rate="Not implemented",
        azimuth_sample_relative_index=burst_header[2:width, 1],
        azimuth_sample_first_valid=burst_header[2:width, 2],
        azimuth_sample_last_valid=burst_header[2:width, 3],
    )


def _cos_samples_reader(filename: Path, bytes_offset: int, width: int, lines: int, version: int) -> np.ndarray:
    """Reader of .cos binary file burst raster data. Implemented in Python and translated from MATLAB
    (see cosSamples.m).

    Original MATLAB code:

    David Young (2025). TerraSAR-X and TanDEM-X tools
    (https://it.mathworks.com/matlabcentral/fileexchange/45956-terrasar-x-and-tandem-x-tools),
    MATLAB Central File Exchange.

    Parameters
    ----------
    filename : Path
        Path to .cos binary file
    bytes_offset : int
        offset in bytes from the beginning of the file
    width : int
        width of the file in 32-bit samples
    lines : int
        number of lines for the current burst
    version : int
        data version

    Returns
    -------
    np.ndarray
        complex burst data array, with shape (samples, lines)

    Raises
    ------
    ValueError
        version 2 is not supported
    """
    burst_raster_count = width * lines
    data = np.fromfile(filename, dtype="<u4", count=burst_raster_count, offset=bytes_offset).reshape(
        (width, lines), order="F"
    )
    data = data[2:, :].ravel("F")
    if version == 1:
        # data are int16
        data = data.view(np.int16).byteswap().copy()
    else:
        # check Matlab code, does not work for this version
        raise ValueError(f"Version {version} not supported")
    data = data.reshape((2, width - 2, lines), order="F")
    return data[0, :, :].astype(np.float32) + 1j * data[1, :, :].astype(np.float32)


def read_tif_raster(
    raster_file: str | Path,
    block_to_read: list[int] | None = None,
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Read tif raster_file

    Parameters
    ----------
    raster_file : str | Path
        path to tif raster file
    block_to_read : list[int] | None, optional
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None
    scaling_conversion : float, optional
        scaling conversion to be multiplied to the data read (sqrt of calibration factor), by default 1

    Returns
    -------
    np.ndarray
        ROI read from tif raster
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

    return target_area * scaling_conversion


@lru_cache(maxsize=5)
def read_binary_cos_file(filename: str | Path, burst_id: int = 0) -> tuple[np.ndarray, BurstAnnotation]:
    """Reader of .cos binary file, extracting burst raster data corresponding to the requested burst id.
    Implemented in Python and translated from MATLAB (see cosReader.m).

    Parameters
    ----------
    filename : str | Path
        Path to .cos binary file
    burst_id : int, optional
        index of the burst to be read, starting from 0 (first burst), by default 0

    Returns
    -------
    np.ndarray
        complex burst data array, with shape (lines, samples)
    BurstAnnotation
        burst annotation object

    Raises
    ------
    ValueError
        if selected burst id exceeds the number of bursts in raster file
    """
    filename = Path(filename)
    width_samples, height_samples, version = _cos_file_header_reader(filename)

    cursor_position = 0
    residual_samples = height_samples
    while True:
        burst_annotation = _cos_burst_annotation_reader(
            filename=filename,
            bytes_offset=cursor_position,
            width=width_samples,
        )
        # updating binary reading cursor position taking into account the size of the current burst annotation
        cursor_position += _compute_burst_annotation_bytes_size(width=width_samples)
        # now the cursor is positioned at the start of the burst raster data
        if burst_id + 1 == burst_annotation.burst_index:
            return (
                _cos_samples_reader(
                    filename=filename,
                    bytes_offset=cursor_position,
                    width=width_samples,
                    lines=burst_annotation.azimuth_samples,
                    version=version,
                ).T,
                burst_annotation,
            )
        else:
            cursor_position += _compute_burst_data_bytes_size(
                width=width_samples, lines=burst_annotation.azimuth_samples
            )
            residual_samples = residual_samples - burst_annotation.azimuth_samples - 4
            if residual_samples < 0:
                raise ValueError(f"Burst ID {burst_id} exceeds the number of bursts in raster file")
