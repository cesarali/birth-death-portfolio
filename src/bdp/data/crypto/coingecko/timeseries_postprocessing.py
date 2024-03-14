import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime,timedelta
from dataclasses import dataclass

from bdp.data.crypto.coingecko.downloads import (
    metadataLists
)

from bdp.data.crypto.coingecko.coingecko_dataclasses import PriceChangeData

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def read_csv(ts_coin_pathdir):
    ts = pd.read_csv(ts_coin_pathdir, index_col=0)  # Use the first column as the index
    # Convert the index back to datetime format since it's read as a string by default
    ts.index = pd.to_datetime(ts.index)
    return ts

from dataclasses import dataclass
from typing import Optional
from pandas import Timestamp

@dataclass
class PredictionSummary:
    prices_max_value_predicted: float
    prices_max_index_predicted: Timestamp
    prices_max_elapsed_predicted: int
    prices_min_value_predicted: float
    prices_min_index_predicted: Timestamp
    prices_min_elapsed_predicted: int
    prices_end_value_predicted: float
    prices_min_max_spread: float
    prices_max_past_percentage: float
    prices_min_past_percentage: float
    prices_past_value:float
    prices_max_end_spread:float
    prices_end_past_percentage:float
    prices_after_max_min_predicted:float

    market_caps_max_value_predicted: float
    market_caps_max_index_predicted: Timestamp
    market_caps_max_elapsed_predicted: int
    market_caps_min_value_predicted: float
    market_caps_min_index_predicted: Timestamp
    market_caps_min_elapsed_predicted: int
    market_caps_end_value_predicted: float
    market_caps_min_max_spread: float
    market_caps_max_past_percentage: float
    market_caps_min_past_percentage: float
    market_caps_past_value:float
    market_caps_max_end_spread:float
    market_caps_end_past_percentage:float
    market_caps_after_max_min_predicted:float

    total_volumes_max_value_predicted: float
    total_volumes_max_index_predicted: Timestamp
    total_volumes_max_elapsed_predicted: int
    total_volumes_min_value_predicted: float
    total_volumes_min_index_predicted: Timestamp
    total_volumes_min_elapsed_predicted: int
    total_volumes_end_value_predicted: float
    total_volumes_min_max_spread: float
    total_volumes_max_past_percentage: float
    total_volumes_min_past_percentage: float
    total_volumes_past_value:float
    total_volumes_max_end_spread:float
    total_volumes_end_past_percentage:float
    total_volumes_after_max_min_predicted:float

@dataclass
class timeseriesMetadata:
    id:str= None
    ts:pd.DataFrame = None
    max_time:datetime = None
    min_time:datetime = None

    past_body:pd.DataFrame=None
    prediction_head:pd.DataFrame = None
    time_10_before:datetime = None
    max_hours:int= None
    num_price_values:int = None
    are_values_same:bool = None
    coin_id:str = None
    prediction_summary:PredictionSummary = None

def summarize_prediction_head_dataframe(df,past_head=None):
    summary = {}
    for column in df.columns:
        max_value = df[column].max()
        min_value = df[column].min()
        end_value = df[column].iloc[-1]
        max_index = df[column].idxmax()
        min_index = df[column].idxmin()

        after_max_min = df[column][df[column].index > max_index].min()

        if column != "elapsed_hours":
            summary[f'{column}_max_value_predicted'] = max_value
            summary[f'{column}_max_index_predicted'] = max_index
            summary[f'{column}_max_elapsed_predicted'] = df["elapsed_hours"].loc[max_index]
            summary[f'{column}_after_max_min_predicted'] = after_max_min

            summary[f'{column}_min_value_predicted'] = min_value
            summary[f'{column}_min_index_predicted'] = min_index
            summary[f'{column}_min_elapsed_predicted'] = df["elapsed_hours"].loc[min_index]

            summary[f'{column}_end_value_predicted'] = end_value

            if past_head is not None:
                past_value = past_head[column] 
                summary[f'{column}_min_max_spread'] = (max_value - min_value)/past_value

                summary[f'{column}_max_past_percentage'] = (max_value - past_value)/past_value
                summary[f'{column}_min_past_percentage'] = (min_value - past_value)/past_value
                summary[f'{column}_end_past_percentage'] = (end_value - past_value)/past_value

                summary[f'{column}_past_value'] = past_value
                summary[f'{column}_max_end_spread'] = (max_value - end_value)/past_value

    return summary

def elapsed_hours(ts):
    # Assuming final_df is your DataFrame with datetime index
    # Step 1: Convert the datetime index to a Series
    time_series = ts.index.to_series()
    # Step 2: Calculate elapsed time from the first timestamp
    # The first timestamp is time_series[0]
    elapsed_time = time_series - time_series[0]
    # Step 3: Convert elapsed time to hours
    elapsed_hours = elapsed_time / pd.Timedelta(hours=1)
    # You can now add this as a new column to your DataFrame
    ts['elapsed_hours'] = np.ceil(elapsed_hours.values).astype(int)
    return ts

def preprocess_dataframe(ts,coin_metadata:PriceChangeData)->timeseriesMetadata:
    """
    creates a timeseries metadata object that prepares the time series for statistical assesment
    and creation of tensor objects for machine learning NOTE: normalization is not done 
    """
    max_time = max(ts.index)
    min_time = min(ts.index)
    time_10_before = max_time - timedelta(days=10)
    ts = elapsed_hours(ts)

    past_body = ts[ts.index < time_10_before]
    prediction_head = ts[time_10_before <= ts.index]
    past_head = past_body.iloc[-1]
    prediction_head_summary = summarize_prediction_head_dataframe(prediction_head,past_head)
    
    prediction_head_summary = PredictionSummary(**prediction_head_summary)
    
    max_hours = max(ts['elapsed_hours'])

    num_price_values:int = np.isreal(ts['prices'].values).sum()
    num_market_cap_values:int = np.isreal(ts['market_caps'].values).sum()
    num_volume_values:int = np.isreal(ts['total_volumes'].values).sum()
    are_values_same = (num_price_values == num_market_cap_values == num_volume_values)

    tsmd = timeseriesMetadata(id=coin_metadata.id,
                              ts=ts,
                              past_body=past_body,
                              prediction_head=prediction_head,
                              max_time = max_time,
                              min_time = min_time,
                              time_10_before=time_10_before,
                              max_hours = max_hours,
                              num_price_values = num_price_values,
                              are_values_same = are_values_same,
                              prediction_summary=prediction_head_summary)
    return tsmd



def get_timeseries_as_torch(metadata_lists):
    """
    returns
    -------
    padded_sequences
    """
    lengths = []
    time_series_ids = []
    tensor_series_list = []
    for coin in tqdm(metadata_lists.uniswap_coins):
        ts_filename = coin.id + ".csv"
        ts_coin_pathdir = metadata_lists.uniswap_coins_date_pathdir / ts_filename
        ts = read_csv(ts_coin_pathdir)
        tsmd = preprocess_dataframe(ts,coin.id)

        if tsmd.num_price_values > 200:
            time_series_ids.append(coin.id)
            tensor_series_list.append(torch.tensor(tsmd.ts.values))
            lengths.append(tsmd.ts.values.shape[0])

    # lengths need to be in decreasing order if enforcing_sorted=True or use enforce_sorted=False
    padded_sequences = pad_sequence(tensor_series_list, batch_first=True, padding_value=0)
    # Pack the padded sequences
    # lengths need to be in decreasing order if enforcing_sorted=True or use enforce_sorted=False
    packed_sequences = pack_padded_sequence(padded_sequences, lengths=lengths, batch_first=True, enforce_sorted=False)
    return lengths,padded_sequences,packed_sequences


def stats_of_tsdm_objects(time_series_metadatas):
    are_the_same = []
    all_num_price_values = []
    all_num_price_values_ = []
    for tsmd in time_series_metadatas:
        all_num_price_values.append((tsmd.num_price_values,tsmd.coin_id))
        all_num_price_values_.append(tsmd.num_price_values)
        are_the_same.append(tsmd.are_values_same)
    all_num_price_values.sort()
    all_num_price_values_ = np.asarray(all_num_price_values_)
    are_the_same = np.asarray(are_the_same)
    some_one_difference = (~are_the_same).sum()
    return all_num_price_values,all_num_price_values_,some_one_difference

if __name__=="__main__":
    metadata_lists:metadataLists = metadataLists()
    print(metadata_lists.num_uniswap_ids_ready)

    lengths,padded_sequences,packed_sequences = get_timeseries_as_torch(metadata_lists)

    print(lengths)
    print(padded_sequences.shape)
    