---
icon: lucide/history
title: "Changelog"
tags:
    - changelog
    - release notes
---

# Changelog

## Unreleased

**New Features**

- ICEYE: support for SLC-COG and GRD-COG products (Cloud Optimized GeoTiff + STAC .json metadata), read into the same
  channel metadata dataclasses of legacy ICEYE products. Azimuth timing is derived from the zero Doppler start/end
  times, the Doppler rate is computed from the orbit (the annotated `iceye:doppler_rate_coeffs` is not an azimuth FM
  rate), raster bands are decoded (GDAL scale/offset, amplitude mapping) and reoriented to the legacy (azimuth time,
  range) convention for both look sides.

## v1.1.2

**Bug fixes**

- Fixed bug in Sentinel-1 external orbit reader

## v1.1.1

**Other Changes**

- Added image radiometric quantity to Sentinel-1 L1 channel metadata dataclass

**Bug Fixes**

- Fixed bug in Sentinel-1 L1 noise vector reader

## v1.1.0

**New Features**

- Added reader for GETASSE-30 DEM product

## v1.0.0

First official release.
