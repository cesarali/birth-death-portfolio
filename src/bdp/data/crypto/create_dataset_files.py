import os
import json
import sys
import torch

import pandas as pd
import numpy as np
import pymongo
import random

from datetime import datetime,timedelta
from bdp.data.utils import divide_data

def read_and_fix(data_folder,coin_file_name="pinkcoin_full.csv",span="full"):
    """
    parse the dates
    """
    coin_file_1 = os.path.join(data_folder,coin_file_name)
    data_frame_1 = pd.read_csv(coin_file_1).set_index('Unnamed: 0')
    data_frame_1.index.name = "time"
    data_frame_1.index = pd.to_datetime(data_frame_1.index,format="%Y-%m-%d %H:%M:%S")
    if span=="full":
        data_frame_1 = data_frame_1.resample(pd.offsets.Day()).agg([np.max])
    elif span=="month":
        data_frame_1 = data_frame_1.resample(pd.offsets.Hour()).agg([np.max])
    else:
        data_frame_1 = data_frame_1.resample(pd.offsets.Second()).agg([np.max])

    return data_frame_1

def create_merged_dataframe(data_folder,collection,break_point=8000,all_coins_ids=None,span="full",gap_filter=0.9):
    """
    reads the data after it was downloaded and creates a concateenated data frame in order to
    ensure that al dates are aligned, it seacrh meta data from coins

    """
    if all_coins_ids is None:
        all_coins_names = [a for a in os.listdir(data_folder)]
        random.shuffle(all_coins_names)
    else:
        all_coins_names = list(map(lambda x: x + "_{0}.csv".format(span), all_coins_ids))

    coin_file_name = all_coins_names[0]
    data_merged = read_and_fix(data_folder,coin_file_name=coin_file_name,span=span)
    coin_id = coin_file_name.split("_")[0]

    coin_data = collection.find_one({"id": coin_id})
    coin_data["index"] = 0
    coins_data = [coin_data]

    j = 1
    for coin_file_name in all_coins_names[1:]:
        if j > break_point:
            break
        try:
            coin_id = coin_file_name.split("_")[0]
            coin_data = collection.find_one({"id": coin_id})

            if span == "full":
                not_nans = coin_data['not_nans']
                survival_time = float(coin_data['survival_time'])
                proportion = not_nans/survival_time

            if proportion > gap_filter and span == "full":
                coin_data["index"] = j
                coins_data.append(coin_data)
                print("Current Coin {0} {1}".format(j,coin_id))
                try:
                    data_frame_2 = read_and_fix(data_folder, coin_file_name=coin_file_name, span=span)
                    data_merged = pd.concat([data_merged, data_frame_2], axis=1, join="outer")
                    j += 1
                except:
                    coins_data.pop()
                    pass
            elif span != "full":
                coin_data["index"] = j
                coins_data.append(coin_data)
                print("Current Coin {0} {1}".format(j,coin_id))
                try:
                    data_frame_2 = read_and_fix(data_folder, coin_file_name=coin_file_name, span=span)
                    data_merged = pd.concat([data_merged, data_frame_2], axis=1, join="outer")
                    j += 1
                except:
                    coins_data.pop()
                    pass
        except:
            print(sys.exc_info())
            pass

    return data_merged,coins_data

def fix_dates(x):
    try:
        del x["_id"]
    except:
        pass
    x["last_date"] = str(x["last_date"])
    x["birth_date"] = str(x["birth_date"])
    return x

def create_file(data_folder, collection,break_point=8000,span="month"):
    """
    create tensors from dataframe concatenate and store (convert nans to 0.)
    also dumps json file
    """
    index_file = os.path.join(data_folder, "{0}_ecosystem_datetime.csv".format(span))

    train_datafile = os.path.join(data_folder, "{0}_ecosystem_train".format(span))
    test_datafile = os.path.join(data_folder, "{0}_ecosystem_test".format(span))
    val_datafile = os.path.join(data_folder, "{0}_ecosystem_val".format(span))

    index_train_datafile = os.path.join(data_folder, "{0}_index_ecosystem_train".format(span))
    index_test_datafile = os.path.join(data_folder, "{0}_index_ecosystem_test".format(span))
    index_val_datafile = os.path.join(data_folder, "{0}_index_ecosystem_val".format(span))

    metadatafile = os.path.join(data_folder, "{0}_meta_ecosystem.json".format(span))

    data_merged, coins_data = create_merged_dataframe(data_folder, collection, break_point,None,span)

    price_df = data_merged["price"]
    price_df = price_df.fillna(0.) #FILL NANs WITH ZEROS <--------------------------------------------------

    marketcap_df = data_merged["market_cap"]
    marketcap_df = marketcap_df.fillna(0.)

    volume_df = data_merged["volume"]
    volume_df = volume_df.fillna(0.)

    prices_ = torch.Tensor(price_df.values).unsqueeze(-1).permute(1,0,2)
    volumes_ = torch.Tensor(volume_df.values).unsqueeze(-1).permute(1,0,2)
    marketcap_ = torch.Tensor(marketcap_df.values).unsqueeze(-1).permute(1,0,2)
    coin_index = torch.linspace(0, len(prices_) - 1, len(prices_)).long()

    #data
    data_ = torch.cat([prices_, volumes_, marketcap_], dim=2)
    data_ = data_
    train_data, test_data, validation_data = divide_data(data_)
    train_coin_index, test_coin_index, validation_coin_index = divide_data(coin_index)

    np.save(train_datafile, train_data)
    np.save(test_datafile, test_data)
    np.save(val_datafile, validation_data)

    np.save(index_train_datafile, train_coin_index)
    np.save(index_test_datafile, test_coin_index)
    np.save(index_val_datafile, validation_coin_index)

    #meta
    coins_data = list(map(fix_dates, coins_data))
    json.dump(coins_data, open(metadatafile, "w"))
    #index
    data_merged["price"].to_csv(index_file, columns=[], header=None)

    return data_merged, coins_data

if __name__=="__main__":
    from deep_fields import data_path

    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date_string)

    client = pymongo.MongoClient()
    db = client["crypto"]
    collection = db['birth_{0}'.format(date_string)]

    #top_coins_name = []
    #for a in collection.find().sort([("last_marketcap", -1)]).limit(10):
    #    top_coins_name.append(a["id"])
    #JUST READ DATA
    #data_merged, coins_data = create_merged_dataframe(data_folder,
    #                                                  collection,
    #                                                  break_point=100,
    #                                                  all_coins_ids=None,
    #                                                  span="month")

    #CREATE FILE
    d0 = datetime.now()
    data_merged, coins_data = create_file(data_folder,collection,8000,span="full")
    df = datetime.now()
    print((df-d0).total_seconds()/3600.)