import os
import json
import torch
import pymongo
import numpy as np
from tqdm import tqdm

from deep_fields.data.m5.dataloaders import covariates_info

non_sequential_covariates, sequential, basic_covariates_final_maps, not_covariate = covariates_info()

def create_covariates(document_example,size_end):
    """
    non_sequentials ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sequentials ["wday", "month", "year", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap"]

    returns
    -------
    COVARIATES, SEQUENCES

    COVARIATES = non_sequentials
    SEQUENCE = np.stack([series_target_life,price_life,series_input_life]) + sequentials
    """
    COVARIATES =  []
    # set non sequentials
    for covariate_name in non_sequential_covariates:
        covariate_map = basic_covariates_final_maps[covariate_name]["str_to_id"]
        COVARIATES.append(covariate_map[document_example[covariate_name]])

    # set sequentials
    life_time_range = [a for a in range(document_example["min_day"],document_example["max_day"]+1)]
    # set infinite prices to negative values
    price = np.asarray(document_example["price"])
    series_target = np.asarray(document_example["series_target"])
    series_input = np.asarray(document_example["series_input"])

    price_life = price[life_time_range]
    series_target_life = series_target[life_time_range]
    series_input_life = series_input[life_time_range]

    not_selling_indexes = np.where(price_life == np.inf)[0]
    price_life[not_selling_indexes] = -1.

    SERIES = np.stack([series_target_life,price_life,series_input_life])
    _,life_time = SERIES.shape
    missing_steps = size_end - life_time
    SERIES = np.pad(SERIES,((0,0),(1,0)),constant_values=-1) # start of sequence
    SERIES = np.pad(SERIES,((0,0),(missing_steps+1,0)),constant_values=-2) # padding

    for covariate_name in sequential:
        covariate_map = basic_covariates_final_maps[covariate_name]["str_to_id"]
        covariate_sequence = document_example[covariate_name]
        covariate_sequence_life = np.asarray(covariate_sequence)[life_time_range]
        covariate_sequence = []
        for u in covariate_sequence_life:
            if u in covariate_map.keys():
                covariate_sequence.append(covariate_map[u])
            else:
                covariate_sequence.append(covariate_map["<unk>"])
        covariate_sequence = np.pad(covariate_sequence,((1,0)),constant_values=covariate_map["<sos>"]) # start of sequence
        covariate_sequence = np.pad(covariate_sequence,((missing_steps+1,0)),constant_values=covariate_map["<pad>"]) # padding
        SERIES = np.vstack((SERIES,covariate_sequence))

    return COVARIATES, SERIES

# --------------- data base stuff --------------------------------------------------------------------------------------
client = pymongo.MongoClient()

db = client["M5"]
collection = db['sales_prices']
sales_collection = db['sales_prices']
calendar_collection = db['calendar']
sell_prices_collection = db['sell_prices']
covariates_collection = db["series_count_covariates"]
basic_stats_collection = db['basic_stats']
# ----------------------------------------------------------------------------------------------------------------------

data_dir = "C:/Users/cesar/Desktop/Projects/General/deep_random_fields/src/deep_fields/data/sample_data/M5/"
json_path = os.path.join(data_dir, "covariates_basic.json")
basic_covariates_final = json.load(open(json_path, "r"))

SIZES = [a for a in range(0,2250,250)]
size_index = 6
size_start = SIZES[size_index]
size_end = SIZES[size_index+1]
cursor = covariates_collection.find({"lifetime":{"$gt":size_start,
                                                 "$lt":size_end}})

print(covariates_collection.count_documents({"lifetime":{"$gt":size_start,
                                                         "$lt":size_end}}))
#document_example = next(cursor)
final_dir = "C:/Users/cesar/Desktop/Projects/NeuralProcessesUncertainty/data/preprocessed/"

if __name__=="__main__":
    count = 0
    FULL_BLOCK_SERIES =  []
    FULL_BLOCK_COVARIATES = []
    for document_example in tqdm(cursor):
        COVARIATES, SERIES = create_covariates(document_example, size_end)
        FULL_BLOCK_COVARIATES.append(COVARIATES)
        FULL_BLOCK_SERIES.append(SERIES)
        count +=1

    path_series = os.path.join(final_dir,"series_covariates")
    path_covariates = os.path.join(final_dir,"covariates")

    FULL_BLOCK_COVARIATES = np.stack(FULL_BLOCK_COVARIATES)
    FULL_BLOCK_SERIES = np.stack(FULL_BLOCK_SERIES)

    np.save(path_covariates,FULL_BLOCK_COVARIATES)
    np.save(path_series,FULL_BLOCK_SERIES)

