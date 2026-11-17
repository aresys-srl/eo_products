# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Test specific product formats readers."""

from eo_products.sentinel1.reader import open_product
from eo_products.sentinel1.zarr_prod.reader import read_ard_channel_data, read_ard_channel_metadata

if __name__ == '__main__':
    prod_path = r"C:\Users\giorgio.parma\Aresys_DATA\sct_data\sentinel1\S01SIWNRB_20260221T053526_0026_a066_S245.zarr"
    prod = open_product(prod_path)
    channel = prod.channels_list[0]
    mtd = read_ard_channel_metadata(prod=prod, channel=channel)
    data = read_ard_channel_data(prod=prod, channel=channel, block_to_read=[30, 75, 1500, 2000])
    ...
