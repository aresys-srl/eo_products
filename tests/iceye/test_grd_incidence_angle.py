# SPDX-License-Identifier: MIT

"""ICEYE legacy GRD incidence angle polynomial unittests"""

from __future__ import annotations

import numpy as np
import pytest
from lxml import etree
from perseo_core.timing import PreciseDateTime

from eo_products.iceye.utilities import ICEYEIncidenceAnglePolynomial

# ground range [m] from first pixel -> incidence angle [deg], as annotated in legacy GRD .xml metadata
COEFFICIENTS = [20.0, 9.5e-5, -5.0e-11, -4.7e-17, 9.7e-23]
RANGE_SPACING = 6.0
SAMPLES = 19002
REFERENCE_TIME = "2026-01-01T12:00:00.000000"


def _grd_metadata_root(origin: float = 0.0) -> etree._Element:
    root = etree.Element("product")
    etree.SubElement(root, "range_spacing").text = repr(RANGE_SPACING)
    node = etree.SubElement(root, "Incidence_Angle_Coefficients")
    etree.SubElement(node, "zero_doppler_time").text = REFERENCE_TIME
    etree.SubElement(node, "ground_range_origin").text = repr(origin)
    for value in COEFFICIENTS:
        etree.SubElement(etree.SubElement(node, "coefficient"), "value").text = repr(value)
    return root


@pytest.mark.parametrize("origin", [0.0, 1200.0])
def test_grd_incidence_polynomial_in_ground_range(origin):
    poly = ICEYEIncidenceAnglePolynomial.from_metadata(_grd_metadata_root(origin))
    pixels = np.array([0, 1, 9500, SAMPLES - 1])
    expected = np.polynomial.Polynomial(COEFFICIENTS)(pixels * RANGE_SPACING - origin)
    evaluated = poly.evaluate_incidence_angle(PreciseDateTime.fromisoformat(REFERENCE_TIME), pixels)
    np.testing.assert_allclose(evaluated, expected, rtol=1e-12)


def test_grd_incidence_far_range():
    """Far range incidence matches the ground range evaluation, not the pixel index one"""
    poly = ICEYEIncidenceAnglePolynomial.from_metadata(_grd_metadata_root())
    far = poly.evaluate_incidence_angle(PreciseDateTime.fromisoformat(REFERENCE_TIME), np.array([SAMPLES - 1]))[0]
    assert far == pytest.approx(30.1, abs=0.1)
    assert far != pytest.approx(np.polynomial.Polynomial(COEFFICIENTS)(SAMPLES - 1), abs=1.0)
