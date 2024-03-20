import os
import sys
from tqdm import tqdm
import torch

from bdp.data.crypto.coingecko.downloads import (
    AllCoinsMetadata
)

from bdp.data.crypto.coingecko.timeseries_preprocessing import (
    get_df_timeserieses,
    timeseries_and_metadata,
    get_timeseries_as_torch
)

if __name__=="__main__":
    date_string = "2024-03-13"
    metadata_lists = AllCoinsMetadata(date_string=date_string) # all metadata objects from files