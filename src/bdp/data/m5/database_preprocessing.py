import numpy as np
from tqdm import tqdm
import pymongo
import sys

miss_str = '<miss>'
dates_kes = ["d_{0}".format(a) for a in range(1,1942)]
not_dates_keys = ['id','item_id' ,'dept_id' ,'cat_id' ,'store_id' ,'state_id']

def created_count_series(document_example,calendar_collection,sell_prices_collection):
    """
    creates the time series of assosiated sequential covariates i.e. price and festivities
    """
    series = []
    series_input = []
    covariates_series = {"wday" :[],
                         "month" :[],
                         "year" :[],
                         "event_name_1" :[],
                         "event_type_1" :[],
                         "event_name_2" :[],
                         "event_type_2" :[],
                         "snap" :[],
                         "price" :[]}

    count = 0
    min_day = min([int(a.split("_")[-1]) for a in document_example.keys() if "d_" in a])
    max_day = max([int(a.split("_")[-1]) for a in document_example.keys() if "d_" in a])

    state_id = document_example["state_id"]
    item_id = document_example['item_id']
    store_id = document_example['store_id']

    all_prices = {price_document['wm_yr_wk']:price_document['sell_price'] \
                  for price_document in sell_prices_collection.find({'store_id': store_id,'item_id': item_id})}

    for d_index in range(min_day,max_day+1):
        day_ = "d_{0}".format(d_index)
        day_count = document_example[day_]
        series_input.append(d_index)
        series.append(day_count)

        covariates = day_covariates(day_,document_example,calendar_collection,sell_prices_collection,all_prices)
        for k ,v in covariates.items():
            covariates_series[k].append(v)
        count +=day_count

    new_doc = {k:document_example[k] for k in not_dates_keys}
    new_doc["count"] = count
    new_doc["series_target"] = series
    new_doc["series_input"] = series_input

    for k ,v in covariates_series.items():
        new_doc[k] = v
        if k == "price":
            where_not_infinite = np.where(np.asarray(v) != np.inf)[0]
            new_doc["max_day"] = int(max(where_not_infinite))
            new_doc["min_day"] = int(min(where_not_infinite))
            new_doc["lifetime"] = int(len(where_not_infinite))
            new_doc["weird_count"] = int(sum(np.asarray(series)[np.asarray(v) == np.inf]))
    return new_doc

def day_covariates(day_, document,calendar_collection,sell_prices_collection,item_prices):
    state_id = document["state_id"]

    calendar_data = calendar_collection.find_one({"d": day_})
    wm_yr_wk = calendar_data['wm_yr_wk']
    wday = calendar_data['wday']
    month = calendar_data['month']
    year = calendar_data['year']
    event_name_1 = calendar_data['event_name_1']
    event_type_1 = calendar_data['event_type_1']
    event_name_2 = calendar_data['event_name_2']
    event_type_2 = calendar_data['event_type_2']
    snap = calendar_data['snap_{0}'.format(state_id)]

    if wm_yr_wk in item_prices.keys():
        price = item_prices[wm_yr_wk]
    else:
        price = np.infty

    return {"wday": wday,
            "month": month,
            "year": year,
            "event_name_1": event_name_1,
            "event_type_1": event_type_1,
            "event_name_2": event_name_2,
            "event_type_2": event_type_2,
            "snap": snap,
            "price": price}