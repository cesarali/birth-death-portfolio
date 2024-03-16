import os
import torch
import pymongo
import numpy as np
import pandas as pd
from bdp import data_path
from matplotlib import pyplot as plt
from bdp.data.crypto.create_dataset_files import create_merged_dataframe


def top_from_collection(date,max_size):
    from bdp import data_path
    client = pymongo.MongoClient()
    db = client["crypto"]
    collection = db['birth_{0}'.format(date)]
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date)

    top_coins_name = []
    for a in collection.find().sort([("last_marketcap", -1)]).limit(max_size):
        top_coins_name.append(a["id"])

    return top_coins_name

if __name__=="__main__":
    client = pymongo.MongoClient()
    db = client["crypto"]

    database_name = "2021-06-14"
    collection = db["birth_{0}".format(database_name)]
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, database_name)
    collection.create_index([('survival_time',-1)])

    top_coins_name = []
    for a in collection.find().sort([("last_marketcap",-1)]).limit(10):
        top_coins_name.append(a["id"])


    data_merged,coins_data = create_merged_dataframe(data_folder,
                                                     collection,
                                                     break_point=20,
                                                     all_coins_ids=top_coins_name,
                                                     span="full")
    data_merged = data_merged.fillna(0.)