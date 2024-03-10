import os
import json
import pandas as pd
import numpy as np

import torch
from pycoingecko import CoinGeckoAPI
from deep_fields import models_path, data_path

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import os
import torch
import pymongo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from deep_fields import data_path
from deep_fields.data.crypto.create_dataset_files import create_merged_dataframe

client = pymongo.MongoClient()
db = client["crypto"]

collection = db['birth_2021-06-02']
crypto_folder = os.path.join(data_path, "raw", "crypto")
data_folder = os.path.join(crypto_folder, "2021-06-02")
collection.create_index([('survival_time',-1)])

top_coins_name = []
for a in collection.find().sort([("last_marketcap",-1)]).limit(10):
    top_coins_name.append(a["id"])

data_merged,coins_data = create_merged_dataframe(data_folder,
                                             collection,
                                             break_point=20,
                                             all_coins_ids=top_coins_name,
                                             span="month")
data_merged = data_merged.fillna(0.)
columns_ids = [coin_data["id"] for coin_data in coins_data]
price_df = data_merged[:-1]["price"]
price_df.columns = columns_ids

mu = expected_returns.mean_historical_return(price_df)
S = risk_models.sample_cov(price_df)

print(mu)