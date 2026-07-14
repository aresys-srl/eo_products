---
icon: lucide/mountain-snow
title: GETASSE Usage
tags:
    - DEM
    - GETASSE
---

# GETASSE-30 DEM Reader

GETASSE30 ("Global Earth Topography And Sea Surface Elevation at 30 arc seconds") is a global DEM organised as 15° x 15° tiles.
Each tile is a 1800 x 1800 grid of big-endian signed 16-bit integers (metres, referenced to the WGS84 ellipsoid).
The tile file name encodes the latitude/longitude of the *centre* of its most south-west pixel, e.g. `45S045W.GETASSE30`.

You can find all the available information about the GETASSE30 product in the
[official documentation](https://step.esa.int/main/wp-content/help/versions/9.0.0/snap/org.esa.snap.snap.help/desktop/GETASSE30ElevationModel.html).

## Prerequisites

The GETASSE30 DEM tiles must be downloaded separately and placed in a local directory. The reader expects the raw binary tile files
(e.g. `45N015E.GETASSE30`) to be present in the specified path.

Data can be downloaded from the [ESA official archive](https://step.esa.int/auxdata/dem/GETASSE30/). Tiles are stored as
compressed ZIP files and **they need to be extracted before usage**.

## Reading a Single Tile

Use `read_dem_tile` to load a single GETASSE30 tile from disk into an `xr.DataArray` with latitude and longitude coordinates.

```python title="Read a single tile"
from eo_products.dem.getasse30 import read_dem_tile

tile = read_dem_tile("/path/to/dem/45N015E.GETASSE30")
print(tile)
```

```title="Example output"
<xarray.DataArray 'altitude' (lat: 1800, lon: 1800)> Size: 6MB
array([[302, 302, 295, ...,  16,  16,  16],
       [286, 274, 269, ...,  16,  16,  16],
       [266, 276, 288, ...,  16,  16,  16],
       ...,
       [847, 748, 708, ...,  31,  31,  31],
       [794, 787, 706, ...,  31,  31,  31],
       [781, 708, 701, ...,  31,  31,  31]],
      shape=(1800, 1800), dtype=int16)
Coordinates:
  * lat      (lat) float64 14kB 59.99 59.98 59.98 59.97 ... 45.02 45.01 45.0
  * lon      (lon) float64 14kB 15.0 15.01 15.02 15.03 ... 29.98 29.98 29.99
Attributes:
    description:  Altitude in meters
    units:        m
```

!!! tip "Caching"
    Repeated reads of the same tile are cached, so calling `read_dem_tile` multiple times with the same path is cheap.

## Querying Altitudes at Coordinates

Use `get_dem_altitudes` to retrieve interpolated altitudes at one or more geographic coordinates. Points are grouped by
the tiles they fall on, and each tile (or tile mosaic) is read once with vectorized interpolation.

```python title="Single point query"
import numpy as np
from eo_products.dem.getasse30 import get_dem_altitudes

altitude = get_dem_altitudes(np.array([45.0, 15.0]), "/path/to/dem")
print(f"Altitude at (45°N, 15°E): {altitude:.1f} m")

# Output: Altitude at (45°N, 15°E): 781.0 m
```

```python title="Batch query"
coords = np.array([
    [48.0, 18.0],
    [46.0, 16.0],
    [45.0, 15.0],
])
altitudes = get_dem_altitudes(coords, "/path/to/dem")
print(altitudes)

# Output: [152. 257. 781.]
```

!!! tip "Antimeridian Handling"
    Points near the ±180° meridian are handled transparently: the reader wraps longitudes and mosaics the relevant tiles
    so interpolation works correctly across the seam.

## Extracting a Rectangular ROI

Use `get_dem_roi` to retrieve elevation data for a rectangular region of interest (ROI) at the native 30-arcsecond resolution.
The function returns an `xr.DataArray` containing bilinearly interpolated elevation values, with latitude (decreasing) and longitude (increasing) axes.
The axes begin exactly at `lat_start_deg` and `lon_start_deg` and are sampled at 30-arcsecond intervals (the DEM's native resolution).
The final sample is one step short of `start + size`.

```python title="Extract a rectangular region"
from eo_products.dem.getasse30 import get_dem_roi

roi = get_dem_roi(
    lat_start_deg=46.5,
    lon_start_deg=16.7,
    lat_size_deg=2.3,
    lon_size_deg=1.7,
    dem="/path/to/dem",
)
print(roi)
```

```title="Example output"
<xarray.DataArray 'altitude' (lat: 276, lon: 204)> Size: 450kB
array([[308., 289., 280., ..., 563., 651., 598.],
       [251., 236., 242., ..., 534., 527., 490.],
       [219., 218., 216., ..., 449., 425., 451.],
       ...,
       [277., 291., 278., ..., 161., 150., 195.],
       [263., 298., 286., ..., 160., 154., 153.],
       [291., 294., 272., ..., 150., 148., 152.]], shape=(276, 204))
Coordinates:
  * lat      (lat) float64 2kB 48.79 48.78 48.77 48.77 ... 46.52 46.51 46.5
  * lon      (lon) float64 2kB 16.7 16.71 16.72 16.72 ... 18.38 18.38 18.39
Attributes:
    description:  Altitude in meters
    units:        m
```

The latitude axis is decreasing (north to south) and the longitude axis is increasing (west to east). The ROI can cross
the antimeridian by specifying a `lon_start_deg` near 180° with a size that extends past it.

## Extracting Over a Polygon

Use `get_dem_polygon` to extract DEM data covering the bounding box of a polygon's convex hull.
The data is returned at the native DEM resolution **without interpolation**. Consequently, the latitude and longitude axes
begin at the nearest DEM grid coordinates **that lie within the bounding box** of the polygon's convex hull.

```python title="Extract DEM over a polygon"
from shapely.geometry import box
from eo_products.dem.getasse30 import get_dem_polygon

polygon = box(16.4, 46.2, 17.65, 50.34)  # (min_lon, min_lat, max_lon, max_lat)
dem_data = get_dem_polygon(polygon, "/path/to/dem")
print(dem_data)
```

```title="Example output"
<xarray.DataArray 'altitude' (lat: 497, lon: 151)> Size: 150kB
array([[808, 800, 795, ..., 287, 288, 289],
       [864, 825, 805, ..., 284, 282, 283],
       [954, 900, 813, ..., 279, 277, 278],
       ...,
       [285, 306, 304, ..., 212, 222, 238],
       [266, 276, 264, ..., 220, 215, 227],
       [336, 271, 245, ..., 224, 200, 212]], shape=(497, 151), dtype=int16)
Coordinates:
  * lat      (lat) float64 4kB 50.33 50.33 50.32 50.31 ... 46.22 46.21 46.2
  * lon      (lon) float64 1kB 16.4 16.41 16.42 16.43 ... 17.63 17.64 17.65
Attributes:
    description:  Altitude in meters
    units:        m
```

!!! warning "Antimeridian Crossing"
    A region that crosses the antimeridian must be expressed with unwrapped longitudes extending past 180°
    (e.g. `shapely.box(179, 34, 181, 36)` for a 2° x 2° box crossing the antimeridian).

## Notes

- **Tile naming convention**: Tile file names encode the latitude/longitude of the *centre* of the most south-west pixel, e.g. `45N015E.GETASSE30` covers 45°N-60°N, 15°E-30°E.
- **Caching**: Tiles are cached after the first read (up to 64 entries), so repeated queries over the same area are efficient.
- **Coordinate system**: All altitudes are in metres relative to the WGS84 ellipsoid.
- **Resolution**: Native resolution is 30 arc seconds (1/120°), approximately ~925 m at the equator.
- **Antimeridian**: Regions crossing the ±180° meridian are handled transparently, using unwrapped longitudes (e.g. 179 to 181) for polygon queries.
