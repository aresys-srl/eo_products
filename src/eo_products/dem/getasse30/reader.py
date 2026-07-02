# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""GETASSE30 DEM reader.

GETASSE30 ("Global Earth Topography And Sea Surface Elevation at 30 arc
seconds") is organised as 15deg x 15deg tiles. Each tile is a 1800 x 1800 grid
of big-endian signed 16-bit integers (metres, referenced to the WGS84
ellipsoid). The tile file name encodes the latitude/longitude of the *centre*
of its most south-west pixel, e.g. ``45S045W.GETASSE30``.

See https://step.esa.int/main/wp-content/help/versions/9.0.0/snap/org.esa.snap.snap.help/desktop/GETASSE30ElevationModel.html
"""

import math
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import numpy.typing as npt
import shapely
import xarray as xr

from eo_products.dem.getasse30.utilities import (
    DEM_RESOLUTION_DEG,
    DEM_TILE_SIZE,
    LAT_BOUNDARIES_DEG,
    LON_BOUNDARIES_DEG,
    TILE_SIZE_DEG,
    bbox_tile_origins,
    generate_tile_name,
    lat_lon_to_tile,
    parse_tile_origin,
    raise_on_invalid_lat,
    raise_on_invalid_lon,
)


def _tile_mosaic(tile_names: list[str], dem: Path) -> xr.DataArray:
    """Read and combine the given tiles into a single altitude DataArray."""
    if len(tile_names) == 1:
        return read_dem_tile(dem.joinpath(tile_names[0]))

    tiles = [read_dem_tile(dem.joinpath(name)) for name in tile_names]
    lon_mins = [float(t.lon.min()) for t in tiles]
    lon_maxs = [float(t.lon.max()) for t in tiles]

    # A gap between adjacent tile clusters wider than one tile means the set
    # straddles the antimeridian; lift the eastern (positive) tiles by +360 so
    # the seam becomes continuous (e.g. 165E..180 followed by 180..195).
    is_across_lon_seam = (max(lon_maxs) - min(lon_mins)) > (2 * TILE_SIZE_DEG)
    if is_across_lon_seam:
        shifted = []
        for tile in tiles:
            if float(tile.lon.min()) < 0:  # western (180W side) tile
                shifted.append(tile.assign_coords(lon=tile.lon + 360))
            else:
                shifted.append(tile)
        tiles = shifted

    return xr.combine_by_coords(tiles)["altitude"]


def _area_mosaic(
    lat_origins: list[int], east_lon_origins: list[int], west_lon_origins: list[int], dem: Path
) -> xr.Dataset:
    """Read the area tiles and combine them into a single mosaic Dataset.

    When ``west_lon_origins`` is non-empty the area crosses the antimeridian; the
    western tiles are shifted by +360deg so the combined longitude axis stays
    monotonic (e.g. eastern ``165E..180`` followed by shifted ``180..195``).
    Longitudes are left in this (possibly shifted) frame; callers restore the
    canonical ``[-180, 180)`` range afterwards.
    """
    east_tiles = [
        read_dem_tile(dem.joinpath(generate_tile_name(lat, lon))) for lat, lon in product(lat_origins, east_lon_origins)
    ]

    if not west_lon_origins:
        return xr.combine_by_coords(east_tiles)

    west_tiles = [
        read_dem_tile(dem.joinpath(generate_tile_name(lat, lon))) for lat, lon in product(lat_origins, west_lon_origins)
    ]
    west_mosaic_shifted = xr.combine_by_coords(west_tiles).assign_coords(
        lon=lambda da: da.lon + 360,
    )

    # The eastern side can be empty (e.g. an area starting exactly at lon 180):
    # in that case the shifted western mosaic alone covers the whole area.
    if not east_tiles:
        return west_mosaic_shifted

    east_mosaic = xr.combine_by_coords(east_tiles)
    return xr.concat([east_mosaic, west_mosaic_shifted], dim="lon")


@lru_cache(maxsize=64)
def _read_tile_cached(tile_path: Path) -> xr.DataArray:
    """Read and cache a single GETASSE30 DEM tile from disk"""
    path = Path(tile_path)
    expected = DEM_TILE_SIZE * DEM_TILE_SIZE
    raw = np.fromfile(path, dtype=">i2")
    if raw.size != expected:
        raise ValueError(
            f"GETASSE30 tile {path} appears to be corrupted: expected {expected} int16 values, found {raw.size}"
        )
    data = raw.reshape((DEM_TILE_SIZE, DEM_TILE_SIZE)).astype(np.int16)
    latitude_start, longitude_start = parse_tile_origin(path.name)
    lats = latitude_start + np.arange(DEM_TILE_SIZE) * DEM_RESOLUTION_DEG
    lons = longitude_start + np.arange(DEM_TILE_SIZE) * DEM_RESOLUTION_DEG

    da = xr.DataArray(
        data,
        coords={"lat": np.flip(lats), "lon": lons},
        dims=("lat", "lon"),
        name="altitude",
        attrs=dict(description="Altitude in meters", units="m"),
    )
    da["lat"].attrs["units"] = "deg"
    da["lon"].attrs["units"] = "deg"
    return da


def read_dem_tile(tile_path: str | Path) -> xr.DataArray:
    """Read GETASSE30 DEM data from a tile path.

    Missing tiles raise a FileNotFoundError. Results are
    cached, so repeated reads of the same tile are cheap.

    Parameters
    ----------
    tile_path : str | Path
        Path to the GETASSE30 tile.

    Returns
    -------
    xr.DataArray
        Altitude DEM data with latitude and longitude axes (int16, metres).
    """
    return _read_tile_cached(Path(tile_path).resolve())


def get_dem_altitudes(coords: npt.NDArray[np.floating], dem: str | Path) -> float | npt.NDArray[np.floating]:
    """Retrieve altitude from GETASSE30 DEM at given coordinate(s).

    Points are grouped by the set of tiles they require, so each tile (or tile
    mosaic) is built once and queried with a single vectorised interpolation.

    Parameters
    ----------
    coords : npt.NDArray[np.floating]
        Geographic coordinates as latitude/longitude in decimal degrees.
        Shape ``(2,)`` for a single point or ``(N, 2)`` for multiple points.
        Each row is ``[latitude, longitude]``, with latitude in ``[-90, 90]``
        and longitude in ``[-180, 180]``.
    dem : str | Path
        Path to the GETASSE30 DEM folder.

    Returns
    -------
    float | npt.NDArray[np.floating]
        Interpolated altitude(s) in metres relative to the WGS84 ellipsoid.
        A float for a single ``(2,)`` input, or an array of shape ``(N,)``.
    """
    dem = Path(dem)
    single_point = np.ndim(coords) == 1
    coords = np.atleast_2d(coords)
    n_coords = len(coords)
    altitudes = np.empty(n_coords)

    all_tiles = lat_lon_to_tile(coords)

    # group points belonging to the same tiles
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, tiles in enumerate(all_tiles):
        groups.setdefault(tuple(sorted(tiles)), []).append(i)

    for tile_key, idxs in groups.items():
        idx = np.asarray(idxs)
        query_lat = coords[idx, 0]
        query_lon = coords[idx, 1]

        # mosaic group tiles, if longitude goes past 180 apply shift to mach the wrapping
        mosaic = _tile_mosaic(list(tile_key), dem)
        if float(mosaic.lon.max()) > LON_BOUNDARIES_DEG[1]:
            # Seam-shifted mosaic (lon axis lifted into [165, 195)): lift the
            # negative-side query longitudes by +360 so they fall in the axis.
            query_lon = np.where(query_lon < 0, query_lon + 360, query_lon)
        else:
            # Non-shifted mosaic (lon axis in [-180, 180)): wrap the query into the
            # same periodic frame. This maps lon == 180 to -180, which is the SW
            # pixel of the 180W tile; without it interp(lon=180) on the 180W tile
            # (whose axis is [-180, -165)) falls out of range and returns NaN.
            query_lon = (query_lon + 180.0) % 360.0 - 180.0

        # interpolate all the points of the group
        sampled = mosaic.interp(
            lat=xr.DataArray(query_lat, dims="points"),
            lon=xr.DataArray(query_lon, dims="points"),
            method="linear",
        )
        altitudes[idx] = sampled.data

    return float(altitudes[0]) if single_point else altitudes


def get_dem_roi(
    lat_start_deg: float, lon_start_deg: float, lat_size_deg: float, lon_size_deg: float, dem: str | Path
) -> xr.DataArray:
    """Extract GETASSE30 DEM elevation data interpolated over a rectangular area.

    Parameters
    ----------
    lat_start_deg : float
        Latitude of the bottom-left corner of the ROI [deg].
    lon_start_deg : float
        Longitude of the bottom-left corner of the ROI [deg].
    lat_size_deg : float
        Latitudinal extent of the ROI [deg]. Must be greater than 30 arcsec
        (i.e. ``> 30/3600`` deg).
    lon_size_deg : float
        Longitudinal extent of the ROI [deg]. Must be greater than 30 arcsec
        (i.e. ``> 30/3600`` deg).
    dem : str | Path
        Path to the GETASSE30 DEM folder.

    Returns
    -------
    xr.DataArray
        2D array of elevation values (metres) with ``lat`` and ``lon``
        dimensions. The axes start at ``lat_start_deg`` / ``lon_start_deg`` and
        are sampled every 30 arcsec; the final sample is one step short of
        ``start + size`` (the stop value is excluded).
    """
    dem = Path(dem)
    raise_on_invalid_lat(lat_start_deg)
    raise_on_invalid_lon(lon_start_deg)
    if lat_size_deg <= DEM_RESOLUTION_DEG or lon_size_deg <= DEM_RESOLUTION_DEG:
        raise ValueError(
            f"Invalid ROI size: lat size {lat_size_deg}, lon size {lon_size_deg} must be > {DEM_RESOLUTION_DEG}"
        )

    lat_stop_deg = lat_start_deg + lat_size_deg
    lon_stop_deg = lon_start_deg + lon_size_deg

    if lat_stop_deg > LAT_BOUNDARIES_DEG[1]:
        raise ValueError(f"{lat_stop_deg} exceeds latitude boundary")
    if lon_size_deg >= 360:
        raise ValueError(f"{lon_size_deg} exceeds longitude size limits")

    lat_origins, east_lon_origins, west_lon_origins = bbox_tile_origins(
        lat_start_deg, lon_start_deg, lat_stop_deg, lon_stop_deg
    )
    wraps_antimeridian = bool(west_lon_origins)
    mosaic = _area_mosaic(lat_origins, east_lon_origins, west_lon_origins, dem)

    n_lat = math.ceil(lat_size_deg / DEM_RESOLUTION_DEG)
    n_lon = math.ceil(lon_size_deg / DEM_RESOLUTION_DEG)
    output_lat_axis = np.flip(np.arange(n_lat) * DEM_RESOLUTION_DEG + lat_start_deg)
    output_lon_axis = np.arange(n_lon) * DEM_RESOLUTION_DEG + lon_start_deg

    da_interp = mosaic.interp(lat=output_lat_axis, lon=output_lon_axis, method="linear")

    # restore longitude values after possible wrapping
    if wraps_antimeridian:
        da_interp = da_interp.assign_coords(lon=(da_interp.lon + 180) % 360 - 180)

    return da_interp["altitude"]


def get_dem_polygon(polygon: shapely.Polygon, dem: str | Path) -> xr.DataArray:
    """Extract GETASSE30 DEM elevation data over a polygon area.

    DEM data is returned as-is (no interpolation), covering the bounding box of
    the polygon's convex hull.

    !!! note

        GETASSE30 has a native resolution of 30 arc seconds.

    !!! warning

        A region that crosses the antimeridian must be expressed with unwrapped longitudes, extending past 180deg
        (e.g. use ``polygon=shapely.box(179, 34, 181, 36)`` to represent a 2deg x 2deg box crossing the antimeridian).

    Parameters
    ----------
    polygon : shapely.Polygon
        Area of interest. The DEM is extracted over the bounding box of this
        polygon's convex hull.
    dem : str | Path
        Path to the GETASSE30 DEM folder.

    Returns
    -------
    xr.DataArray
        2D array of elevation values (metres) with ``lat`` and ``lon``
        dimensions, covering the bounding box of ``polygon``.
    """
    dem = Path(dem)
    polygon = shapely.convex_hull(polygon)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    lat_origins, east_lon_origins, west_lon_origins = bbox_tile_origins(min_lat, min_lon, max_lat, max_lon)
    wraps_antimeridian = bool(west_lon_origins)
    mosaic = _area_mosaic(lat_origins, east_lon_origins, west_lon_origins, dem)

    # When the box crosses the antimeridian the mosaic lon axis is shifted into
    # [start, max_lon] (max_lon > 180); ``min_lon``/``max_lon`` already live in
    # that same shifted frame, so the slice below works unchanged.
    mosaic = mosaic.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    if wraps_antimeridian:
        mosaic = mosaic.assign_coords(lon=(mosaic.lon + 180) % 360 - 180)
    return mosaic["altitude"]
