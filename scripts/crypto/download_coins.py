import os
import sys

from bdp.data.crypto.coingecko.downloads import (
    get_all_coins_and_markets,
    get_coin_timeseries_raw,
    get_one_coin_metadata
)

if __name__=="__main__":
    date_string = "2024-03-19"
    coin_gecko_key = "CG-rkg4RTUcfEWYAQ4xUejxPpkS"