---
icon: lucide/history
title: "Changelog"
tags:
    - changelog
    - release notes
---

# Changelog

## Unreleased

**Bug fixes**

- ICEYE GRD: the incidence angle polynomial annotated in the .xml metadata is a function of the ground range in meters,
  it was evaluated at range pixel indexes, giving wrong incidence angles (and sigma nought to beta nought conversions)
  away from near range

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
