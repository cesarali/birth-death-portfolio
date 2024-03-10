import os
import sys
import torch
import pymongo
from tqdm import tqdm

import datetime
import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime as dt

from matplotlib import pyplot as plt

from deep_fields.data.m5.database_preprocessing import created_count_series

if __name__=="__main__":

    client = pymongo.MongoClient()
    db = client["M5"]
    collection = db['sales_prices']
    sales_collection = db['sales_prices']
    calendar_collection = db['calendar']
    sell_prices_collection = db['sell_prices']

    calendar_collection.create_index([("d", 1)])
    sell_prices_collection.create_index([('store_id', 1), ('item_id', 1)])

    create_again = True

    document_example = next(sales_collection.find({}))
    create_again = True
    create_number = 1e6
    created = 0
    if create_again:
        db.drop_collection("series_count_covariates")
        collection_new = db["series_count_covariates"]
        BULK = []
        for document_example in tqdm(collection.find({})):
            new_doc = created_count_series(document_example, calendar_collection, sell_prices_collection)
            BULK.append(new_doc)
            created+=1
            if created > create_number:
                break
            if len(BULK) == 100:
                try:
                    collection_new.insert_many(BULK)
                    BULK = []
                    print("Insertion!")
                except:
                    print(sys.exc_info())
                    BULK = []
                    print("Trouble!")

        collection_new.insert_many(BULK)
        # CHECK
        #https: // github.com / Automattic / mongoose / issues / 6629