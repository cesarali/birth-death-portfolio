import os
import torch
import pymongo
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

from bdp import models_path

crypto_plots_dir = os.path.join(models_path,"crypto_plots")
if not os.path.isdir(crypto_plots_dir):
    os.makedirs(crypto_plots_dir)

from bdp import data_path
from bdp.data.crypto.create_dataset_files import create_merged_dataframe

def coins_in_a_window(date0,datef,collection,survival=True):
    if not survival:
        coins_ = [a["id"] for a in collection.find({"birth_date":{"$gt":date0,"$lt":datef}})]
    else:
        coins_ = [(a["survival_time"],a["id"])
                  for a in collection.find({"birth_date":{"$gt":date0,"$lt":datef}})]
        coins_.sort()
        coins_ = coins_[::-1]
    return coins_

def coins_before(date0,collection):
    coins_ = [a["id"] for a in collection.find({"birth_date":{"$lt":date0}})]
    return coins_

def top_before(collection,date0,top=10):
    top_coins_name = []
    for a in collection.find({"birth_date":{"$lt":date0}}).sort([("last_marketcap", -1)]).limit(top):
        top_coins_name.append(a["id"])
        #survival_time.append(a["survival_time"])
        #birth_dates.append(a["birth_date"])
    return top_coins_name

def top_and_birth(date,date0,datef,top=10,max_size=500):
    """
    in order to obtain a first taste of the results with minimum computation, we
    select a subset of the market growth

    we pick
    :return:

    """
    from deep_fields import data_path
    client = pymongo.MongoClient()
    db = client["crypto"]
    collection = db['birth_{0}'.format(date)]

    tb = top_before(collection,date0,top=top)
    cw = coins_in_a_window(date0, datef, collection, survival=True)
    market_names = [name for name in tb]

    current_size = len(tb)
    birth_index = 0
    while current_size < max_size:
        market_names.append(cw[birth_index][1])
        current_size+=1
        birth_index+=1

    return tb, market_names

def birth_time_series(collection):
    birth_dates = []
    for a in collection.find():
        birth_dates.append(a["birth_date"])

    unique_birth_dates = list(set(birth_dates))
    unique_birth_dates.sort()
    birth_time_series = dict(zip(unique_birth_dates,np.zeros(len(unique_birth_dates))))
    for birth in birth_dates:
        birth_time_series[birth]+=1
    birth_time_series = pd.Series(birth_time_series)
    return birth_time_series

if __name__=="__main__":
    date0 = datetime(2018, 1, 1)
    datef = datetime(2019, 1, 1)

    client = pymongo.MongoClient()
    db = client["crypto"]
    db.collection_names()
    collection = db['birth_2021-06-14']

    #collection.create_index([('survival_time', -1)])
    #collection.create_index([('survival_time', 1)])

    #bts = birth_time_series(collection)
    #bts.plot()
    #plt.show()


