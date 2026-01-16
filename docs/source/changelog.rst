Changelog
=========

v1.0.5
------

**Bug fixing**

- Sentinel-1: fixing bug in BurstInfo generation when products is Stripmap (in this case BurstInfo now is equal to RasterInfo)

**Other changes**

- Sentinel-1: adding Burst Sensing Times to metadata reader.

v1.0.4
------

**New features**

- Terrasar-X: added support for reading Terrasar-X products.

**Other changes**

- ``common.utilities.BurstInfo`` `lines_per_burst` and `samples_per_burst` are now `int | np.ndarray` instead of `int`.
- Sentinel-1: added support for reading Elevation Notch products.

v1.0.3
------

**New features**

- COSMO: adding support for reading Second Generation COSMO products.

**Other changes**

- Radarsat-2: improved support for reading Radarsat-2 products acquisition modes and proper conversion to SAR standard modes.
- StriX: improved support for reading Doppler Rate and Doppler Centroid polynomials for SLC products.

v1.0.2
------

**New features**

-  Added support for Synspective StriX L1 products, both GRD (GeoTiff + XML) and SLC (NITF).

**Bug fixing**

-  Sentinel-1: fixing bug in Orbit Type determination for Sentinel-1 products.
-  Sentinel-1: fixing bug in starting and stop indexes computation for Sentinel-1 External Orbit reader when providing time boundaries.

v1.0.1
------

**Bug fixing**

-  Sentinel-1: fixing bug in ``read_external_orbit`` for provided time boundaries out of orbit validity boundaries.
-  Sentinel-1: fixing bug in ``read_external_orbit`` when returning the whole orbit.

v1.0.0
------

First stable version.

v1.0.0.dev2
-----------

Bug fixing for SAOCOM reader.

v1.0.0.dev1
-----------

Bug fixing for EOS-04 and NovaSAR-1.

v1.0.0.dev0
-----------

First development version.
