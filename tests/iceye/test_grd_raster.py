# SPDX-License-Identifier: MIT

"""ICEYE legacy GRD raster reading unittests"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from eo_products.iceye.reader import read_channel_data


@pytest.mark.parametrize("overviews", [False, True])
def test_legacy_grd_raster_with_overviews(tmp_path, overviews):
    """GRD GeoTiffs with overview levels are read at full resolution"""
    data = np.arange(64 * 48, dtype=np.float32).reshape(64, 48)
    raster_file = tmp_path / "grd.tif"
    with tifffile.TiffWriter(raster_file) as tif:
        tif.write(data, tile=(16, 16), metadata=None)
        if overviews:
            tif.write(data[::2, ::2], tile=(16, 16), subfiletype=1, metadata=None)

    np.testing.assert_array_equal(read_channel_data(raster_file, [5, 7, 20, 30], 2.0), data[5:25, 7:37] * 2.0)
    np.testing.assert_array_equal(read_channel_data(raster_file), data)
