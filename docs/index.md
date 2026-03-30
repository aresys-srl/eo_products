---
icon: lucide/sparkles
title: Overview
tags:
    - SAR
    - SCT
---

# SAR L1 Products Readers

``EO Products`` is a Python package designed to simplify access to **Level-1 Synthetic Aperture Radar** (SAR) Earth Observation products.
It provides a unified collection of readers tailored to a variety of SAR product formats, enabling users to efficiently
extract and work with both metadata and raster data.

The readers included in this package are built to handle the most commonly used metadata fields and ensure full access to
the associated raster data, making them suitable for a wide range of scientific and operational applications.

!!! warning "Reader Completeness"

    While the implementation aims to cover the majority of relevant product information, it does not guarantee complete
    representation of every metadata element available in the original product specifications.

EO Products is intended for developers, researchers, and practitioners who need a practical and consistent interface for
ingesting SAR data into analysis workflows, without requiring deep familiarity with the complexities of individual
product formats.

These readers are designed to be flexible and adaptable, allowing users to customize their workflows based on their specific
requirements and the characteristics of the data they are working with.
Most of [SCT Product Formats Plugins](http://intranet.aresys.it/sardashboard/develop/sct-plugins/docs/docs/latest/) are
based on the readers included in this package.

## Supported Formats

EO Products currently supports the following SAR products:

- **Sentinel-1 SAFE**: SLC and GRD products [topsar, stripmap, wave, notch]
- **NovaSAR-1**: SLC, GRD, SCD and SRD products
- **ICEYE**: SLC and GRD products [topsar, stripmap, spotlight]
- **SAOCOM**: SLC and GRD products [topsar, stripmap]
- **EOS-04**: SLC, GRD products
- **COSMO SkyMed**: SLC, GRD products
- **RADARSAT-2**: SLC, GRD products [scansar, stripmap]
- **TerraSAR-X**: SLC and GRD products [scansar (GRD only), stripmap]
- **STRIX (Synspective)**: SLC and GRD products [stripmap, spotlight]
