# EO Products

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://python.org)
[![PyPI version](https://img.shields.io/pypi/v/eo-products)](https://pypi.org/project/eo-products/)

**EO Products** is a Python package designed to simplify access to **Level-1 Synthetic Aperture Radar** (SAR) Earth Observation products and SAR auxiliary data.
It provides a unified collection of readers tailored to a variety of SAR product formats, enabling users to efficiently
extract and work with both metadata and raster data.

The readers included in this package are built to handle the most commonly used metadata fields and ensure full access to
the associated raster data, making them suitable for a wide range of scientific and operational applications.

EO Products is intended for developers, researchers, and practitioners who need a practical and consistent interface for
ingesting SAR data into analysis workflows, without requiring deep familiarity with the complexities of individual
product formats.

> While the implementation aims to cover the majority of relevant product information, it does not guarantee complete
> representation of every metadata element available in the original product specifications.

## Supported Formats

EO Products currently supports the following SAR products:

| Format | Modes |
|--------|-------|
| **Sentinel-1 SAFE** | SLC and GRD [topsar, stripmap, wave, notch] |
| **NovaSAR-1** | SLC, GRD, SCD and SRD |
| **ICEYE** | SLC and GRD [topsar, stripmap, spotlight] |
| **SAOCOM** | SLC and GRD [topsar, stripmap] |
| **EOS-04** | SLC and GRD |
| **COSMO SkyMed** | SLC and GRD |
| **RADARSAT-2** | SLC and GRD [scansar, stripmap] |
| **TerraSAR-X** | SLC and GRD [scansar (GRD only), stripmap] |
| **STRIX (Synspective)** | SLC [stripmap, spotlight] |

## Installation

This project requires Python **3.11 or higher**.

This package can be installed using ``pip``:

```bash
pip install eo-products
```

## Documentation

Full documentation is available at [https://aresys-srl.github.io/eo_products](https://aresys-srl.github.io/eo_products).

## Contributing

Contributions are welcome! If you encounter a bug, have a feature request, or want to contribute code:

- **Report bugs & request features**: open an issue on [GitHub](https://github.com/aresys-srl/eo_products/issues). Include a clear description, steps to reproduce, and your environment details.
- **Submit changes**: fork the repository, create a feature branch, and open a pull request. Ensure your code passes the existing linting and test suite.
- **Questions**: use GitHub Discussions for general questions and discussions.

## License

This project is licensed under the MIT License.

Copyright &copy; 2026-present Aresys S.r.L. <info@aresys.it>
