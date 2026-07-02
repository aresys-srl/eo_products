# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Utilities for GETASSE30 DEM reader."""

import math
from itertools import product

import numpy as np
import numpy.typing as npt

LAT_BOUNDARIES_DEG = (-90, 90)
LON_BOUNDARIES_DEG = (-180, 180)

# Tile origins are the latitude/longitude of a tile's most south-west pixel.
MAX_TILE_LAT = 75  # northernmost tile origin (covers 75N..90N)
MIN_TILE_LAT = -90  # southernmost tile origin
MAX_TILE_LON = 165  # easternmost tile origin (covers 165E..180E)
MIN_TILE_LON = -180  # westernmost tile origin

TILE_SIZE_DEG = 15

TILE_EXTENSION = ".GETASSE30"

DEM_STEP_ARCSEC = 30
DEM_RESOLUTION_DEG = 30 / 3600  # 1/120 deg between adjacent pixel centres
HALF_RES_DEG = DEM_RESOLUTION_DEG / 2

# Number of pixels along each axis of a tile (15deg / (30 arcsec) = 1800).
DEM_TILE_SIZE = 1800


def raise_on_invalid_lat(lat: float | npt.NDArray[np.floating]) -> None:
    """Validate that latitude values fall within bounds"""
    lat = np.asarray(lat)
    invalid = (lat < LAT_BOUNDARIES_DEG[0]) | (lat > LAT_BOUNDARIES_DEG[1])
    if np.any(invalid):
        raise ValueError(
            f"Invalid latitude values: {lat[invalid]} outside of [{LAT_BOUNDARIES_DEG[0]},{LAT_BOUNDARIES_DEG[1]}]"
        )


def raise_on_invalid_lon(lon: float | npt.NDArray[np.floating]) -> None:
    """Validate that longitude values fall within bounds"""
    lon = np.asarray(lon)
    invalid = (lon < LON_BOUNDARIES_DEG[0]) | (lon > LON_BOUNDARIES_DEG[1])
    if np.any(invalid):
        raise ValueError(
            f"Invalid longitude values: {lon[invalid]} outside of [{LON_BOUNDARIES_DEG[0]},{LON_BOUNDARIES_DEG[1]}]"
        )


def _wrap_longitude(lon: float) -> float:
    """Wrap a longitude (or tile origin) into the ``[-180, 180)`` range.

    Maps ``180 -> -180`` and e.g. ``-195 -> 165`` so the antimeridian is treated
    as periodic.
    """
    return (lon + 180.0) % 360.0 - 180.0


def _tile_origin_candidates(coord: float) -> tuple[int, int | None]:
    """Axis-agnostic tile origin and optional neighbor arithmetic computation. Origin is raw in the sense that it does
    not take into account coordinate wrapping/clamping.

    Cases:

    1) interior: input coordinate lies within a single tile. The selected tile is the one containing the coordinates,
    no neighboring tile is selected.
    2) right gap: input coordinate lies beyond a tile last pixel center but before the tile right edge (last pixel
    center + half dem resolution). Therefore, the right neighboring tile is needed for proper interpolation.
    3) left gap: input coordinate lies before the first pixel center of a tile but after tile left edge (first pixel
    center - half dem resolution). Therefore, the left neighboring tile is needed for proper interpolation.

    Parameters
    ----------
    coords : float
        coordinate in degrees to assign related tiles to

    Returns
    -------
    int
        tile origin
    int | None
        neighbor tile origin, if any
    """

    # determining the tile origin for the current input coordinate
    tile_start_origin = math.floor((coord + HALF_RES_DEG) / TILE_SIZE_DEG) * TILE_SIZE_DEG

    if coord < tile_start_origin:
        # left gap: the coordinate lies within the half-pixel region immediately before this tile origin, previous tile
        # is needed for interpolation
        neighbor_tile_origin = tile_start_origin - TILE_SIZE_DEG
    elif coord > tile_start_origin + TILE_SIZE_DEG - DEM_RESOLUTION_DEG:
        # right gap: the coordinate lies within the half-pixel region immediately after this tile last pixel, next tile
        # is needed for interpolation
        neighbor_tile_origin = tile_start_origin + TILE_SIZE_DEG
    else:
        # the coordinate is safely inside the tile, and no neighboring tile is necessary
        neighbor_tile_origin = None

    return tile_start_origin, neighbor_tile_origin


def _lat_tile_origins(coord: float) -> list[int]:
    """Defining the eligible tile origins for the current latitude coordinate by applying latitude bounding discipline.

    Rationale: latitude clamping is enforced at boundaries.

    Parameters
    ----------
    coord : float
        latitude coordinate in degrees

    Returns
    -------
    list[int]
        latitude coordinate(s) corresponding to the south-west corner of the relevant tile(s)
    """
    tile_start_origin, neighbor_tile_origin = _tile_origin_candidates(coord)
    # capping latitude between its boundaries
    tile_start_origin = min(max(tile_start_origin, MIN_TILE_LAT), MAX_TILE_LAT)
    origins = [tile_start_origin]
    if (
        neighbor_tile_origin is not None
        and MIN_TILE_LAT <= neighbor_tile_origin <= MAX_TILE_LAT
        and neighbor_tile_origin != tile_start_origin
    ):
        origins.append(neighbor_tile_origin)
    return origins


def _lon_tile_origins(coord: float) -> list[int]:
    """Defining the eligible tile origins for the current longitude coordinate by applying longitude bounding
    discipline.

    Rationale: longitude is treated as periodic. It wraps: 180 -> -180, -195 -> 165.

    Parameters
    ----------
    coord : float
        longitude coordinate in degrees

    Returns
    -------
    list[int]
        longitude coordinate(s) corresponding to the south-west corner of the relevant tile(s)
    """
    tile_start_origin, neighbor_tile_origin = _tile_origin_candidates(coord)
    tile_start_origin = int(_wrap_longitude(tile_start_origin))
    origins = [tile_start_origin]
    if neighbor_tile_origin is not None:
        neighbor = int(_wrap_longitude(neighbor_tile_origin))
        if neighbor != tile_start_origin:
            origins.append(neighbor)
    return origins


def lat_lon_to_tile(coords: npt.NDArray[np.floating]) -> list[list[str]]:
    """GETASSE30 DEM tile name lookup with edge detection.

    Returns the tile(s) containing each coordinate. If a point falls on an edge
    pixel of a tile (within half a pixel of a tile boundary), the neighboring
    tile(s) are also returned so interpolation can cross tile boundaries. The
    antimeridian is handled as periodic; the poles clamp to a single tile.

    Parameters
    ----------
    coords : npt.NDArray[np.floating]
        Coordinates in degrees, either a single point of shape ``(2,)`` as
        ``[latitude, longitude]``, or multiple points of shape ``(N, 2)`` where
        each row is ``[latitude, longitude]``.

    Returns
    -------
    list[list[str]]
        One list of tile names per input point (1, 2, or 4 names depending on
        whether the point is interior, on an edge, or on a corner).
    """
    coords = np.atleast_2d(coords)
    lats, lons = coords[:, 0], coords[:, 1]

    raise_on_invalid_lat(lats)
    raise_on_invalid_lon(lons)

    results: list[list[str]] = []
    for lat, lon in zip(lats, lons, strict=True):
        lat_origins = _lat_tile_origins(float(lat))
        lon_origins = _lon_tile_origins(float(lon))
        results.append([generate_tile_name(lat_, lon_) for lat_, lon_ in product(lat_origins, lon_origins)])

    return results


def generate_tile_name(lat: int, lon: int) -> str:
    """Generate a GETASSE30 DEM tile name given its origin latitude and longitude."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):02d}{ns}{abs(lon):03d}{ew}{TILE_EXTENSION}"


def parse_tile_origin(tile_name: str) -> tuple[int, int]:
    """Return ``(latitude_origin, longitude_origin)`` parsed from a tile file name."""
    latitude_start = int(tile_name[0:2]) if tile_name[2] == "N" else -int(tile_name[0:2])
    longitude_start = int(tile_name[3:6]) if tile_name[6] == "E" else -int(tile_name[3:6])
    return latitude_start, longitude_start


def _axis_tile_range(start_deg: float, stop_deg: float) -> list[int]:
    """Tile origins (multiples of ``TILE_SIZE_DEG``) covering ``[start, stop]``."""
    first = int(math.floor(start_deg / TILE_SIZE_DEG) * TILE_SIZE_DEG)
    last = int(math.ceil(stop_deg / TILE_SIZE_DEG) * TILE_SIZE_DEG)
    if last == first:
        last = first + TILE_SIZE_DEG
    return list(range(first, last, TILE_SIZE_DEG))


def bbox_tile_origins(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> tuple[list[int], list[int], list[int]]:
    """Tile origins covering a bounding box.

    Returns ``(lat_origins, east_lon_origins, west_lon_origins)``. When the box
    crosses the antimeridian (``max_lon > 180``) the longitude origins are split
    into the eastern side (``[start, 180)``) and the wrapped western side
    (negative longitudes); otherwise ``west_lon_origins`` is empty.
    """
    lat_origins = _axis_tile_range(min_lat, max_lat)

    if max_lon <= LON_BOUNDARIES_DEG[1]:
        return lat_origins, _axis_tile_range(min_lon, max_lon), []

    east_lon_origins = list(
        range(int(math.floor(min_lon / TILE_SIZE_DEG) * TILE_SIZE_DEG), LON_BOUNDARIES_DEG[1], TILE_SIZE_DEG)
    )
    west_lon_origins = _axis_tile_range(LON_BOUNDARIES_DEG[0], max_lon - 360)
    return lat_origins, east_lon_origins, west_lon_origins
