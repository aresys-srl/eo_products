# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""GETASSE30 DEM unittests"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from eo_products.dem.getasse30.reader import (
    _read_tile_cached,
    get_dem_altitudes,
    get_dem_polygon,
    get_dem_roi,
    read_dem_tile,
)
from eo_products.dem.getasse30.utilities import (
    DEM_RESOLUTION_DEG,
    DEM_TILE_SIZE,
    HALF_RES_DEG,
    MAX_TILE_LAT,
    MIN_TILE_LAT,
    TILE_SIZE_DEG,
    _lat_tile_origins,
    _lon_tile_origins,
    _wrap_longitude,
    generate_tile_name,
    lat_lon_to_tile,
    parse_tile_origin,
    raise_on_invalid_lat,
    raise_on_invalid_lon,
)

TILES_FOLDER = "/mock_path"


@pytest.fixture(autouse=True)
def _clear_tile_cache():
    """Keep tile reads (and their warnings) deterministic across tests."""
    _read_tile_cached.cache_clear()
    yield
    _read_tile_cached.cache_clear()


@lru_cache(maxsize=None)
def _synthetic_tile(name: str) -> xr.DataArray:
    """Build a tile whose values are a known field ``round(10*lat + lon)``.

    The field is globally consistent across tiles, so mosaics and cross-tile
    interpolation are meaningful, and a query exactly on a pixel centre returns
    a predictable value.
    """
    lat0, lon0 = parse_tile_origin(name)
    lats = lat0 + np.arange(DEM_TILE_SIZE) * DEM_RESOLUTION_DEG
    lons = lon0 + np.arange(DEM_TILE_SIZE) * DEM_RESOLUTION_DEG
    lat_axis = np.flip(lats)
    field = np.round(np.add.outer(lat_axis * 10.0, lons)).astype(np.int16)
    return xr.DataArray(
        field,
        coords={"lat": lat_axis, "lon": lons},
        dims=("lat", "lon"),
        name="altitude",
        attrs=dict(description="Altitude in meters", units="m"),
    )


@pytest.fixture
def fake_tile():
    rng = np.random.default_rng(seed=42)
    data = rng.integers(-500, 5000, size=(DEM_TILE_SIZE, DEM_TILE_SIZE), dtype=np.int16)
    lat_origin, lon_origin = 45, 15
    lats = lat_origin + np.arange(DEM_TILE_SIZE) * DEM_RESOLUTION_DEG
    lons = lon_origin + np.arange(DEM_TILE_SIZE) * DEM_RESOLUTION_DEG
    return xr.DataArray(
        data,
        coords={"lat": np.flip(lats), "lon": lons},
        dims=("lat", "lon"),
        name="altitude",
        attrs=dict(description="Altitude in meters", units="m"),
    )


@pytest.fixture
def mock_read_tile(monkeypatch):
    """Patch read_tile so callers get a position-aware synthetic tile."""
    monkeypatch.setattr("eo_products.dem.getasse30.reader.read_dem_tile", lambda path: _synthetic_tile(Path(path).name))


@pytest.mark.parametrize(
    "lat, lon, tile_name",
    [
        (0, 15, "00N015E.GETASSE30"),
        (-15, -175, "15S175W.GETASSE30"),
        (15, -15, "15N015W.GETASSE30"),
        (0, 0, "00N000E.GETASSE30"),
    ],
)
def test_generate_tile_name(lat, lon, tile_name):
    assert generate_tile_name(lat, lon) == tile_name


@pytest.mark.parametrize(
    "name, origin",
    [
        ("45N015E.GETASSE30", (45, 15)),
        ("30S045W.GETASSE30", (-30, -45)),
        ("90S180W.GETASSE30", (-90, -180)),
        ("75N165E.GETASSE30", (75, 165)),
    ],
)
def test_parse_tile_origin(name, origin):
    assert parse_tile_origin(name) == origin


class TestWrapLongitude:
    @pytest.mark.parametrize(
        "lon, expected",
        [(0, 0), (180, -180), (-180, -180), (195, -165), (-195, 165), (165, 165)],
    )
    def test_wrap(self, lon, expected):
        assert _wrap_longitude(lon) == pytest.approx(expected)


class TestCheckLatValidity:
    def test_valid_scalar(self):
        raise_on_invalid_lat(0.0)
        raise_on_invalid_lat(90.0)
        raise_on_invalid_lat(-90.0)

    def test_valid_array(self):
        raise_on_invalid_lat(np.array([-90.0, 0.0, 45.0, 90.0]))

    @pytest.mark.parametrize("lat", [-90.1, 90.1, -180.0, 200.0])
    def test_invalid_scalar(self, lat):
        with pytest.raises(ValueError, match="latitude"):
            raise_on_invalid_lat(lat)

    def test_invalid_array_raises(self):
        with pytest.raises(ValueError, match="latitude"):
            raise_on_invalid_lat(np.array([0.0, 91.0, -10.0]))


class TestCheckLonValidity:
    def test_valid_boundaries(self):
        raise_on_invalid_lon(-180.0)
        raise_on_invalid_lon(180.0)
        raise_on_invalid_lon(0.0)

    def test_valid_array(self):
        raise_on_invalid_lon(np.array([-180.0, -90.0, 0.0, 90.0, 180.0]))

    @pytest.mark.parametrize("lon", [-180.1, 180.1, 360.0, -270.0])
    def test_invalid_scalar(self, lon):
        with pytest.raises(ValueError, match="longitude"):
            raise_on_invalid_lon(lon)


class TestAxisTileOrigins:
    def _lat(self, coord):
        return _lat_tile_origins(coord)

    def _lon(self, coord):
        return _lon_tile_origins(coord)

    def test_interior_returns_single(self):
        assert self._lat(37.5) == [30]
        assert self._lon(7.5) == [0]

    def test_left_edge_adds_lower_neighbour(self):
        # Just below a tile origin -> needs the tile below as well.
        origins = self._lat(0.0 - DEM_RESOLUTION_DEG * 0.1)
        assert set(origins) == {0, -TILE_SIZE_DEG}

    def test_right_edge_adds_upper_neighbour(self):
        # Just above a tile's last pixel centre -> needs the tile above as well.
        origins = self._lat(15 - HALF_RES_DEG * 0.5)
        assert set(origins) == {0, 15}

    def test_exact_origin_is_interior(self):
        assert self._lat(15.0) == [15]

    def test_exact_top_pixel_is_interior(self):
        # The north-east-most pixel centre (origin + 15 - RES) needs no neighbor.
        assert self._lat(15 - DEM_RESOLUTION_DEG) == [15 - TILE_SIZE_DEG]

    def test_pole_clamps_without_neighbour(self):
        assert self._lat(90.0) == [MAX_TILE_LAT]
        assert self._lat(90 - DEM_RESOLUTION_DEG * 0.1) == [MAX_TILE_LAT]
        assert self._lat(-90.0) == [MIN_TILE_LAT]

    def test_antimeridian_wraps_neighbour(self):
        # Just inside +180 -> primary is the wrapped 180W tile, neighbor 165E.
        origins = self._lon(180 - HALF_RES_DEG * 0.1)
        assert set(origins) == {-180, 165}

    def test_lon_180_maps_to_minus_180(self):
        assert self._lon(180.0) == [-180]

    def test_near_minus_180_is_interior(self):
        assert self._lon(-180 + HALF_RES_DEG * 0.1) == [-180]


class TestLatLonToTile:
    @pytest.mark.parametrize(
        "lat, lon, tile_name",
        [
            (0.0 - DEM_RESOLUTION_DEG * 0.1, 15.0, {"00N015E.GETASSE30", "15S015E.GETASSE30"}),
            (-19.5, -174.42, {"30S180W.GETASSE30"}),
            (15, -15 - DEM_RESOLUTION_DEG * 0.4, {"15N015W.GETASSE30", "15N030W.GETASSE30"}),
            (15, 15 - HALF_RES_DEG * 1.1, {"15N000E.GETASSE30", "15N015E.GETASSE30"}),
            (
                -30 - HALF_RES_DEG,
                30 - HALF_RES_DEG,
                {"45S015E.GETASSE30", "45S030E.GETASSE30", "30S015E.GETASSE30", "30S030E.GETASSE30"},
            ),
            (15, -15 + DEM_RESOLUTION_DEG * 0.4, {"15N015W.GETASSE30"}),
            (-90 + HALF_RES_DEG, -180 + HALF_RES_DEG, {"90S180W.GETASSE30"}),
            (85.345, 180, {"75N180W.GETASSE30"}),
            (90 - DEM_RESOLUTION_DEG, 173, {"75N165E.GETASSE30"}),
            (35, 90 - HALF_RES_DEG * 0.4, {"30N075E.GETASSE30", "30N090E.GETASSE30"}),
            (35, 90 + HALF_RES_DEG * 0.4, {"30N090E.GETASSE30"}),
            (90, 90, {"75N090E.GETASSE30"}),
            (-90, 90, {"90S090E.GETASSE30"}),
            (90, 180, {"75N180W.GETASSE30"}),
            (-90, -180, {"90S180W.GETASSE30"}),
            (37.7522, 14.9952, {"30N000E.GETASSE30", "30N015E.GETASSE30"}),
            # antimeridian wrap: just inside +180 needs both seam tiles
            (45, 180 - HALF_RES_DEG * 0.1, {"45N180W.GETASSE30", "45N165E.GETASSE30"}),
            # near the pole: clamp to a single tile, no spurious neighbor
            (90 - HALF_RES_DEG * 0.1, 20, {"75N015E.GETASSE30"}),
        ],
    )
    def test_single_valid_inputs(self, lat, lon, tile_name):
        assert set(lat_lon_to_tile([lat, lon])[0]) == tile_name

    def test_single_input_returns_list_of_lists(self):
        result = lat_lon_to_tile([45, 15])
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(t, str) for t in result[0])

    def test_explicit_2d_single_point(self):
        # A genuine (1, 2) batch must still return one result list.
        result = lat_lon_to_tile(np.array([[45, 15]]))
        assert result == [["45N015E.GETASSE30"]]

    def test_multiple_valid_inputs(self):
        coords = np.array(
            [
                [0.0 - DEM_RESOLUTION_DEG * 0.1, 15.0],
                [-30 - HALF_RES_DEG, 30 - HALF_RES_DEG],
                [-90, 90],
                [90, 180],
                [-90, -180],
            ]
        )
        assert lat_lon_to_tile(coords) == [
            ["00N015E.GETASSE30", "15S015E.GETASSE30"],
            ["30S030E.GETASSE30", "30S015E.GETASSE30", "45S030E.GETASSE30", "45S015E.GETASSE30"],
            ["90S090E.GETASSE30"],
            ["75N180W.GETASSE30"],
            ["90S180W.GETASSE30"],
        ]

    @pytest.mark.parametrize(
        "lat, lon",
        [
            (90 + DEM_RESOLUTION_DEG, 90),
            (45, 180 + DEM_RESOLUTION_DEG),
            (-90 - DEM_RESOLUTION_DEG, 90),
            (-45, -180 - DEM_RESOLUTION_DEG),
        ],
    )
    def test_single_invalid_inputs(self, lat, lon):
        with pytest.raises(ValueError):
            lat_lon_to_tile([lat, lon])

    def test_multiple_invalid_inputs(self):
        coords = np.array(
            [
                [0.0 - DEM_RESOLUTION_DEG * 0.1, 15.0],
                [-30 - HALF_RES_DEG, 30 - HALF_RES_DEG],
                [-90, 90],
                [90, 180 + 1],
                [-90, -180],
            ]
        )
        with pytest.raises(ValueError):
            lat_lon_to_tile(coords)


class TestReadTile:
    def test_fake_tile_shape(self, fake_tile):
        assert fake_tile.shape == (DEM_TILE_SIZE, DEM_TILE_SIZE)
        assert fake_tile.lat.shape == (DEM_TILE_SIZE,)
        assert fake_tile.lon.shape == (DEM_TILE_SIZE,)

    def test_fake_tile_dtype(self, fake_tile):
        assert fake_tile.dtype == np.int16
        assert fake_tile.lat.dtype == "float64"
        assert fake_tile.lon.dtype == "float64"

    def test_real_bytes_roundtrip(self, tmp_path):
        """Read actual big-endian bytes and verify SW-pixel naming convention."""
        arr = np.full((DEM_TILE_SIZE, DEM_TILE_SIZE), 100, dtype=">i2")
        arr[DEM_TILE_SIZE - 1, 0] = 777  # south-west pixel
        arr[0, 0] = 1000  # north-west pixel
        tile = tmp_path / "30S045W.GETASSE30"
        arr.tofile(tile)

        da = read_dem_tile(tile)
        assert da.shape == (DEM_TILE_SIZE, DEM_TILE_SIZE)
        assert da.dtype == np.int16
        # SW pixel centre is at the tile origin (-30, -45)
        assert float(da.sel(lat=-30, lon=-45)) == 777
        # NW pixel centre is at (origin_lat + 15 - RES, origin_lon)
        assert float(da.sel(lat=-30 + TILE_SIZE_DEG - DEM_RESOLUTION_DEG, lon=-45, method="nearest")) == 1000

    def test_corrupt_tile_raises(self, tmp_path):
        tile = tmp_path / "30S045W.GETASSE30"
        np.zeros(10, dtype=">i2").tofile(tile)
        with pytest.raises(ValueError, match="corrupted"):
            read_dem_tile(tile)

    def test_missing_tile_error(self):
        tile_path = Path(TILES_FOLDER).joinpath("00N0180E.GETASSE30")
        with pytest.raises(FileNotFoundError):
            read_dem_tile(str(tile_path))


class TestLatLonToAltitude:
    def test_single_input(self, mock_read_tile):
        coord = np.array([50, 20])
        assert isinstance(get_dem_altitudes(coord, TILES_FOLDER), float)

    def test_multiple_inputs(self, mock_read_tile):
        coords = np.array([[50, 20], [45, 25], [55.0, 18.0]])
        h = get_dem_altitudes(coords, TILES_FOLDER)
        assert len(h) == len(coords)
        assert isinstance(h[0], float)

    def test_altitude_is_finite(self, mock_read_tile):
        coord = np.array([50.0, 20.0])
        assert np.isfinite(get_dem_altitudes(coord, TILES_FOLDER))

    def test_multiple_altitudes_all_finite(self, mock_read_tile):
        coords = np.array([[50, 20], [45, 25], [55.0, 18.0]])
        result = get_dem_altitudes(coords, TILES_FOLDER)
        assert np.all(np.isfinite(result))

    def test_single_input_shape_is_scalar(self, mock_read_tile):
        coord = np.array([50.0, 20.0])
        result = get_dem_altitudes(coord, TILES_FOLDER)
        assert np.ndim(result) == 0 or isinstance(result, float)

    def test_explicit_2d_single_point_returns_array(self, mock_read_tile):
        result = get_dem_altitudes(np.array([[50.0, 20.0]]), TILES_FOLDER)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_value_on_pixel_centre(self, mock_read_tile):
        # Synthetic field is round(10*lat + lon); (45, 15) is a pixel centre.
        result = get_dem_altitudes(np.array([45.0, 15.0]), TILES_FOLDER)
        assert result == pytest.approx(10 * 45 + 15, abs=1)

    def test_two_tile_boundary(self, mock_read_tile):
        """Point in the half-pixel gap on the lon axis -> 2-tile mosaic."""
        coord = np.array([45.0, 30 - HALF_RES_DEG * 0.5])
        assert set(lat_lon_to_tile(coord)[0]) == {"45N015E.GETASSE30", "45N030E.GETASSE30"}
        assert np.isfinite(get_dem_altitudes(coord, TILES_FOLDER))

    def test_four_tile_corner(self, mock_read_tile):
        """Point in the half-pixel gap on both axes -> 4-tile mosaic."""
        coord = np.array([45 - HALF_RES_DEG * 0.5, 30 - HALF_RES_DEG * 0.5])
        assert len(lat_lon_to_tile(coord)[0]) == 4
        assert np.isfinite(get_dem_altitudes(coord, TILES_FOLDER))

    def test_antimeridian_seam_is_finite(self, mock_read_tile):
        """Across the +180/-180 seam the mosaic must be lifted so interp works."""
        coord = np.array([45.0, 180 - HALF_RES_DEG * 0.1])
        assert set(lat_lon_to_tile(coord)[0]) == {"45N180W.GETASSE30", "45N165E.GETASSE30"}
        assert np.isfinite(get_dem_altitudes(coord, TILES_FOLDER))

    def test_lon_exactly_180_is_finite(self, mock_read_tile):
        """Regression: lon == 180 maps to the single 180W tile (lon axis
        [-180, -165)); the query must be wrapped to -180 instead of returning NaN."""
        coord = np.array([35.0, 180.0])
        assert lat_lon_to_tile(coord)[0] == ["30N180W.GETASSE30"]
        result = get_dem_altitudes(coord, TILES_FOLDER)
        assert np.isfinite(result)

    def test_lon_180_equals_lon_minus_180(self, mock_read_tile):
        """+180 and -180 are the same meridian, so altitudes must match."""
        east = get_dem_altitudes(np.array([35.0, 180.0]), TILES_FOLDER)
        west = get_dem_altitudes(np.array([35.0, -180.0]), TILES_FOLDER)
        assert east == pytest.approx(west)

    def test_lon_180_value_on_pixel_centre(self, mock_read_tile):
        # 180 -> -180 (SW pixel of 180W tile); field is round(10*lat + lon).
        result = get_dem_altitudes(np.array([35.0, 180.0]), TILES_FOLDER)
        assert result == pytest.approx(10 * 35 + (-180), abs=1)

    def test_mixed_batch_with_antimeridian_all_finite(self, mock_read_tile):
        """A batch mixing interior and both-sides-of-antimeridian points stays finite."""
        coords = np.array([[50.0, 20.0], [35.0, 180.0], [35.0, -180.0], [70.0, 30.0]])
        result = get_dem_altitudes(coords, TILES_FOLDER)
        assert np.all(np.isfinite(result))


class TestExtractDEMROI:
    @pytest.mark.parametrize(
        "lat_start_deg, lon_start_deg, lat_size_deg, lon_size_deg",
        [
            (15, 15, HALF_RES_DEG, 10),
            (15, 15, 0.0, 10),
            (15, 15, 10, HALF_RES_DEG),
            (15, 15, HALF_RES_DEG / 2, HALF_RES_DEG / 2),
            (15, 15, 90, 10),  # latitude overflow (now ValueError)
            (15, 15, 85, 10),  # latitude overflow (now ValueError)
            (15, 15, 10, 360),  # longitude size limit (now ValueError)
        ],
    )
    def test_invalid_size_value(self, lat_start_deg, lon_start_deg, lat_size_deg, lon_size_deg):
        with pytest.raises(ValueError):
            get_dem_roi(lat_start_deg, lon_start_deg, lat_size_deg, lon_size_deg, TILES_FOLDER)

    def test_type(self, mock_read_tile):
        result = get_dem_roi(44.0, 9.0, 1.0, 1.0, TILES_FOLDER)
        assert isinstance(result, xr.DataArray)

    def test_axis(self, mock_read_tile):
        result = get_dem_roi(44.0, 9.0, 1.0, 1.0, TILES_FOLDER)
        assert set(result.dims) == {"lat", "lon"}

    def test_output_shape_matches_resolution(self, mock_read_tile):
        lat_size, lon_size = 1.0, 2.0
        result = get_dem_roi(44.0, 9.0, lat_size, lon_size, TILES_FOLDER)
        assert result.sizes["lat"] == round(lat_size / DEM_RESOLUTION_DEG)
        assert result.sizes["lon"] == round(lon_size / DEM_RESOLUTION_DEG)

    def test_lat_axis_is_decreasing(self, mock_read_tile):
        result = get_dem_roi(44.0, 9.0, 1.0, 1.0, TILES_FOLDER)
        assert (np.diff(result.lat.values) < 0).all()

    def test_lat_axis_starts_at_lat_start(self, mock_read_tile):
        result = get_dem_roi(44.0, 9.0, 1.0, 1.0, TILES_FOLDER)
        assert result.lat.values[-1] == pytest.approx(44.0)

    def test_lon_axis_starts_at_lon_start(self, mock_read_tile):
        result = get_dem_roi(44.0, 9.0, 1.0, 1.0, TILES_FOLDER)
        assert result.lon.values[0] == pytest.approx(9.0)

    def test_value_on_pixel_centre(self, mock_read_tile):
        # (44, 9) is a pixel centre; synthetic field is round(10*lat + lon).
        result = get_dem_roi(44.0, 9.0, 1.0, 1.0, TILES_FOLDER)
        sampled = result.sel(lat=44.0, lon=9.0, method="nearest")
        assert float(sampled) == pytest.approx(10 * 44 + 9, abs=1)

    def test_output_longitudes_in_canonical_range(self, mock_read_tile):
        result = get_dem_roi(44.0, 9.0, 1.0, 1.0, TILES_FOLDER)
        assert (result.lon.values >= -180).all()
        assert (result.lon.values <= 180).all()

    def test_wrapping_region_returns_dataarray(self, mock_read_tile):
        result = get_dem_roi(44.0, 179.0, 1.0, 2.0, TILES_FOLDER)
        assert isinstance(result, xr.DataArray)

    def test_wrapping_longitudes_canonical(self, mock_read_tile):
        result = get_dem_roi(44.0, 179.0, 1.0, 2.0, TILES_FOLDER)
        assert (result.lon.values >= -180).all()
        assert (result.lon.values <= 180).all()

    def test_wrapping_no_nan(self, mock_read_tile):
        result = get_dem_roi(44.0, 179.0, 1.0, 2.0, TILES_FOLDER)
        assert not np.isnan(result.values).any()

    def test_lon_stop_exactly_180_no_wrap(self, mock_read_tile):
        result = get_dem_roi(44.0, 178.0, 1.0, 2.0, TILES_FOLDER)
        assert isinstance(result, xr.DataArray)
        assert not np.isnan(result.values).any()

    def test_non_wrapping_region(self, mock_read_tile):
        result = get_dem_roi(40.0, 9.0, 2.0, 2.0, TILES_FOLDER)
        assert result.lon.values.max() < 180

    def test_start_exactly_180_empty_east_side(self, mock_read_tile):
        """Regression: lon_start == 180 makes the eastern tile list empty; the
        shifted western (180W) mosaic alone must still produce valid data."""
        result = get_dem_roi(35.0, 180.0, 0.5, 0.5, TILES_FOLDER)
        assert isinstance(result, xr.DataArray)
        assert not np.isnan(result.values).any()
        assert (result.lon.values >= -180).all()
        assert (result.lon.values <= 180).all()


class TestExtractDEMPolygon:
    @pytest.fixture
    def simple_square(self):
        """A small square polygon well within a single tile."""
        return box(9.1, 44.1, 9.5, 44.5)  # (min_lon, min_lat, max_lon, max_lat)

    @pytest.fixture
    def multi_tile_square(self):
        """A polygon spanning two longitude tiles (origins 0 and 15)."""
        return box(10.0, 10.0, 20.0, 12.0)

    @pytest.fixture
    def antimeridian_square(self):
        """A polygon crossing the antimeridian (max_lon > 180)."""
        return box(179.0, 34.0, 181.0, 36.0)

    def test_returns_dataarray(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert isinstance(result, xr.DataArray)

    def test_output_dims(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert set(result.dims) == {"lat", "lon"}

    def test_output_name_is_altitude(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert result.name == "altitude"

    def test_dtype(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert result.dtype == np.int16
        assert result.lat.dtype == "float64"
        assert result.lon.dtype == "float64"

    def test_lat_axis_is_decreasing(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert (np.diff(result.lat.values) < 0).all()

    def test_lon_axis_is_increasing(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert (np.diff(result.lon.values) > 0).all()

    def test_covers_bbox(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert result.lat.size > 0 and result.lon.size > 0
        assert result.lat.values.min() <= 44.1
        assert result.lon.values.max() >= 9.5 - DEM_RESOLUTION_DEG

    def test_no_nan_in_interior(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        assert not np.isnan(result.values).any()

    def test_output_resolution_matches_dem(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        lon_steps = np.diff(result.lon.values)
        assert np.allclose(lon_steps, DEM_RESOLUTION_DEG, atol=1e-10)

    def test_lat_resolution_matches_dem(self, mock_read_tile, simple_square):
        result = get_dem_polygon(simple_square, TILES_FOLDER)
        lat_steps = np.abs(np.diff(result.lat.values))
        assert np.allclose(lat_steps, DEM_RESOLUTION_DEG, atol=1e-10)

    def test_multi_tile_is_contiguous(self, mock_read_tile, multi_tile_square):
        """A polygon spanning two tiles must produce one contiguous lon axis."""
        result = get_dem_polygon(multi_tile_square, TILES_FOLDER)
        assert (np.diff(result.lon.values) > 0).all()
        assert np.allclose(np.diff(result.lon.values), DEM_RESOLUTION_DEG, atol=1e-10)
        assert not np.isnan(result.values).any()
