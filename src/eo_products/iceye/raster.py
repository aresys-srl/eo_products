# SPDX-License-Identifier: MIT

"""ICEYE shadows-down raster reading: COG products (SLC-COG, GRD-COG) and shadows-down legacy GRD products.

Shadows-down rasters store range along rows (near range first) and azimuth along columns, in increasing time order for
right looking products and in decreasing time order for left looking products. Blocks are returned in the (azimuth lines
in time order, range samples) convention of native legacy products.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import zarr

from eo_products.iceye.utilities import ICEYEProductLevel, ICEYERasterLayout

POWER_AMPLITUDE_MAPPING = "power"  # y = (scale * A) ** exponent


def _invert_amplitude_mapping(values: np.ndarray, raster_layout: ICEYERasterLayout) -> np.ndarray:
    """Converting stored amplitude values back to linear amplitude."""
    if raster_layout.amplitude_mapping_function == POWER_AMPLITUDE_MAPPING:
        return np.power(values, 1 / raster_layout.amplitude_mapping_exponent) / raster_layout.amplitude_mapping_scale
    return values


def decode_bands(data: np.ndarray, raster_layout: ICEYERasterLayout) -> np.ndarray:
    """Decoding band values to complex samples (SLC) or amplitude (GRD).

    Parameters
    ----------
    data : np.ndarray
        raw band values, with bands along the last axis (SLC-COG) or without band axis (GRD-COG)
    raster_layout : ICEYERasterLayout
        COG raster layout

    Returns
    -------
    np.ndarray
        complex samples (SLC-COG) or amplitude (GRD-COG), float32 precision
    """

    def _scaled(band: np.ndarray, index: int) -> np.ndarray:
        return band.astype(np.float32) * raster_layout.band_scales[index] + raster_layout.band_offsets[index]

    if raster_layout.product_level == ICEYEProductLevel.GRD:
        band = data[..., 0] if data.ndim == 3 else data
        return _invert_amplitude_mapping(_scaled(band, 0), raster_layout)

    amplitude = _invert_amplitude_mapping(_scaled(data[..., 0], 0), raster_layout)
    phase = _scaled(data[..., 1], 1)
    samples = np.empty(amplitude.shape, dtype=np.complex64)
    samples.real = amplitude * np.cos(phase)
    samples.imag = amplitude * np.sin(phase)
    return samples


def read_shadows_down_raster(
    raster_file: Path | str, raster_layout: ICEYERasterLayout, block_to_read: list[int] | None = None
) -> np.ndarray:
    """Reading and decoding a block of a shadows-down ICEYE raster (COG products, shadows-down legacy GRD), in the
    (lines, samples) convention of native legacy products.

    Lines are azimuth lines in increasing time order, samples are range samples from near to far range.

    Parameters
    ----------
    raster_file : Path | str
        path to the GeoTiff raster file
    raster_layout : ICEYERasterLayout
        raster layout
    block_to_read : list[int] | None, optional
        data block to be read, as [first line, first sample, number of lines, number of samples],
        if None, the whole raster is read, by default None

    Returns
    -------
    np.ndarray
        decoded data with shape (lines, samples)
    """
    if block_to_read is None:
        block_to_read = [0, 0, raster_layout.azimuth_lines, raster_layout.range_samples]
    first_line, first_sample, lines, samples = (int(v) for v in block_to_read)
    first_column = raster_layout.azimuth_lines - first_line - lines if raster_layout.azimuth_reversed else first_line

    img_store = tifffile.imread(raster_file, aszarr=True, level=0)
    try:
        raster = zarr.open(img_store, mode="r")
        bands = len(raster_layout.band_scales)
        if raster.ndim == 3 and raster.shape[0] == bands and raster.shape[-1] != bands:
            # planar configuration: bands first
            data = np.moveaxis(
                raster[:, first_sample : first_sample + samples, first_column : first_column + lines], 0, -1
            )
        else:
            data = raster[first_sample : first_sample + samples, first_column : first_column + lines]
    finally:
        img_store.close()

    if raster_layout.azimuth_reversed:
        data = data[:, ::-1]
    # (samples, lines) -> (lines, samples)
    return decode_bands(np.swapaxes(data, 0, 1), raster_layout)
