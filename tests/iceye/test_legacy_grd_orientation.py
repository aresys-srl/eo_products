# SPDX-License-Identifier: MIT

"""ICEYE legacy GRD raster orientation unittests"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from lxml import etree

from eo_products.iceye.reader import read_channel_data
from eo_products.iceye.utilities import ICEYEProductLevel, InvalidICEYEProduct, raster_layout_from_metadata

LINES, SAMPLES = 40, 24


def _grd_root(orientation: str | None, look: str = "left") -> etree._Element:
    root = etree.Element("product")
    if orientation is not None:
        etree.SubElement(root, "data_orientation").text = orientation
    etree.SubElement(root, "look_side").text = look
    etree.SubElement(root, "number_of_azimuth_samples").text = str(LINES)
    etree.SubElement(root, "number_of_range_samples").text = str(SAMPLES)
    return root


@pytest.mark.parametrize("orientation", [None, "native"])
def test_native_grd_has_no_layout(orientation):
    assert raster_layout_from_metadata(_grd_root(orientation)) is None


def test_unsupported_grd_orientation():
    with pytest.raises(InvalidICEYEProduct):
        raster_layout_from_metadata(_grd_root("north_up"))


@pytest.mark.parametrize("look", ["left", "right"])
def test_shadows_down_grd_read_in_native_convention(tmp_path, look):
    """A shadows-down GRD raster is read back in the native (azimuth time lines, range samples) convention"""
    native = np.arange(LINES * SAMPLES, dtype=np.float32).reshape(LINES, SAMPLES)  # (lines in time order, samples)
    stored = native.T if look == "right" else native[::-1].T  # rows: range, columns: azimuth (reversed if left)
    raster_file = tmp_path / "grd.tif"
    tifffile.imwrite(raster_file, stored, tile=(16, 16), metadata=None)

    layout = raster_layout_from_metadata(_grd_root("shadows_down", look))
    assert layout.product_level == ICEYEProductLevel.GRD
    assert (layout.azimuth_lines, layout.range_samples, layout.look_side) == (LINES, SAMPLES, look.upper())
    np.testing.assert_array_equal(read_channel_data(raster_file, raster_layout=layout), native)
    for block in ([0, 0, 5, 7], [35, 17, 5, 7], [12, 3, 20, 15]):
        np.testing.assert_array_equal(
            read_channel_data(raster_file, block, 2.0, raster_layout=layout),
            native[block[0] : block[0] + block[2], block[1] : block[1] + block[3]] * 2.0,
        )
