import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime,timedelta
from dataclasses import dataclass
import math 
from bdp.data.crypto.coingecko.dataloaders import TimeSeriesTorchForTraining

from bdp.data.crypto.coingecko.coingecko_dataclasses import (
    AllCoinsMetadata,
    CoinMetadata
)

from typing import List,Dict,Tuple

from bdp.data.crypto.coingecko.coingecko_dataclasses import PriceChangeData

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from bdp.data.crypto.coingecko.metadata_postprocessing import price_change_data_to_dataframe
from bdp.data.crypto.coingecko.downloads import get_coins_to_download
from bdp.data.crypto.coingecko.coingecko_dataclasses import AllCoinsMetadata

def read_csv(ts_coin_pathdir):
    ts = pd.read_csv(ts_coin_pathdir, index_col=0)  # Use the first column as the index
    # Convert the index back to datetime format since it's read as a string by default
    ts.index = pd.to_datetime(ts.index)
    return ts

from dataclasses import dataclass
from typing import Optional
from pandas import Timestamp

from dataclasses import dataclass,asdict

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
    total_volumes_max_end_spread:float
    total_volumes_end_past_percentage:float
    total_volumes_after_max_min_predicted:float

@dataclass
class CoinTimeseriesMetadata:
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
    prediction_summary:PredictionSummary = None

    max_prices:float = None
    max_market_caps:float = None
    max_total_volumns:float = None

    max_prices_index:datetime = None
    max_market_caps_index:datetime = None
    max_total_volumns_index:datetime = None

    past_value_non_zero:Dict[str,Tuple[float,datetime]] = None

def prediction_summary_to_tensor(instance: PredictionSummary) -> torch.tensor:
    if not isinstance(instance,dict):
        # Convert dataclass instance to dictionary
        instance_dict = asdict(instance)
    else:
        instance_dict = instance

    # Define keys to exclude
    keys_to_exclude = {
        'market_caps_max_index_predicted', 
        'market_caps_max_elapsed_predicted', 
        'market_caps_min_index_predicted', 
        'market_caps_min_elapsed_predicted',
        'total_volumes_max_elapsed_predicted',
        'total_volumes_max_index_predicted',
        'total_volumes_min_elapsed_predicted',
        'total_volumes_min_index_predicted',
        'prices_max_elapsed_predicted',
        'prices_max_index_predicted',
        'prices_min_elapsed_predicted',
        'prices_min_index_predicted',
    }

    # Adjust the list comprehension to replace None with math.nan for float_values and exclude certain keys
    float_values = [
        value if isinstance(value, float) else math.nan 
        for key, value in instance_dict.items() 
        if (isinstance(value, float) or value is None) and key not in keys_to_exclude
    ]

    # Adjust the keys list to exclude certain keys
    keys = [
        key 
        for key, value in instance_dict.items() 
        if (isinstance(value, float) or value is None) and key not in keys_to_exclude
    ]

    # Now, when converting float_values to a tensor, None values are already replaced by NaN
    float_array = torch.tensor(float_values, dtype=torch.float32)
    return float_array,keys

def summarize_prediction_head_dataframe(df,past_body=None):
    past_head = past_body.iloc[-1]

    summary = {}
    for column in df.columns:
        if not df[column].dropna().empty:
            max_value = df[column].max()
            min_value = df[column].min()
            end_value = df[column].iloc[-1]
            max_index = df[column].idxmax()
            min_index = df[column].idxmin()

            after_max_min = df[column][df[column].index > max_index].min()
            if np.isnan(after_max_min):
                after_max_min = max_value
                
            if column != "elapsed_hours":

                summary[f'{column}_max_value_predicted'] = max_value
                summary[f'{column}_max_index_predicted'] = max_index
                summary[f'{column}_max_elapsed_predicted'] = df["elapsed_hours"].loc[max_index]
                summary[f'{column}_after_max_min_predicted'] = after_max_min

                summary[f'{column}_min_value_predicted'] = min_value
                summary[f'{column}_min_index_predicted'] = min_index
                summary[f'{column}_min_elapsed_predicted'] = df["elapsed_hours"].loc[min_index]

                summary[f'{column}_end_value_predicted'] = end_value

                if past_body is not None:
                    past_value = past_head[column] 
                    if pd.notnull(past_value) and past_value != 0:
                        summary[f'{column}_min_max_spread'] = (max_value - min_value)/past_value
                        summary[f'{column}_max_past_percentage'] = (max_value - past_value)/past_value
                        summary[f'{column}_min_past_percentage'] = (min_value - past_value)/past_value
                        summary[f'{column}_end_past_percentage'] = (end_value - past_value)/past_value
                        summary[f'{column}_max_end_spread'] = (max_value - end_value)/past_value
                    else:
                        summary[f'{column}_min_max_spread'] = None
                        summary[f'{column}_max_past_percentage'] = None
                        summary[f'{column}_min_past_percentage'] = None
                        summary[f'{column}_end_past_percentage'] = None
                        summary[f'{column}_max_end_spread'] = None
        else:
            if column != "elapsed_hours":
                summary[f'{column}_max_value_predicted'] = None
                summary[f'{column}_max_index_predicted'] = None
                summary[f'{column}_max_elapsed_predicted'] = None
                summary[f'{column}_after_max_min_predicted'] = None

                summary[f'{column}_min_value_predicted'] = None
                summary[f'{column}_min_index_predicted'] = None
                summary[f'{column}_min_elapsed_predicted'] = None

                summary[f'{column}_end_value_predicted'] = None
            
                summary[f'{column}_min_max_spread'] = None
                summary[f'{column}_max_past_percentage'] = None
                summary[f'{column}_min_past_percentage'] = None
                summary[f'{column}_end_past_percentage'] = None
                summary[f'{column}_max_end_spread'] = None
           
    #numerical_values,keys = prediction_summary_to_tensor(summary)

    return summary

def elapsed_hours(ts):
    # Assuming final_df is your DataFrame with datetime index
    # Step 1: Convert the datetime index to a Series
    time_series = ts.index.to_series()
    # Step 2: Calculate elapsed time from the first timestamp
    # The first timestamp is time_series[0]
    elapsed_time = time_series - time_series.iloc[0]
    # Step 3: Convert elapsed time to hours
    elapsed_hours = elapsed_time / pd.Timedelta(hours=1)
    # You can now add this as a new column to your DataFrame
    ts['elapsed_hours'] = np.ceil(elapsed_hours.values).astype(int)
    return ts

def normalize(past_body,prediction_head):
    #normalization --------------------------------------------------
    max_values = past_body.max()
    max_indices = past_body.idxmax()

    # Normalize 'past_body' DataFrame
    past_body_columns = ["prices", 'market_caps', 'total_volumes']
    for col in past_body_columns:
        # Adjust indexing for 'max_values' if it's a Series or a single-row DataFrame
        max_value = max_values[col].iloc[0] if isinstance(max_values, pd.DataFrame) else max_values[col]
        past_body.loc[:, col] = past_body.loc[:, col] / max_value

    # Normalize 'prediction_head' DataFrame
    prediction_head_columns = ["prices", 'market_caps', 'total_volumes']
    for col in prediction_head_columns:
        # Adjust indexing for 'max_values' if it's a Series or a single-row DataFrame
        max_value = max_values[col].iloc[0] if isinstance(max_values, pd.DataFrame) else max_values[col]
        prediction_head.loc[:, col] = prediction_head.loc[:, col] / max_value
    return max_values,max_indices,past_body,prediction_head

def preprocess_timeseries_dataframe(ts:pd.DataFrame,coin_id:str)->CoinTimeseriesMetadata:
    """
    creates a timeseries metadata object that prepares the time series for statistical assesment
    and creation of tensor objects for machine learning NOTE: normalization is not done 

    1. We divide the time series in past and head
    2. We normalize the time series according to the full maximum 
    3. We include the elapsed hours in the dataframe 
    4. We check how many values are stored
    """
    #================
    #times
    max_time = max(ts.index)
    min_time = min(ts.index)
    time_10_before = max_time - timedelta(days=10)

    # hours
    ts = elapsed_hours(ts)

    past_body = ts[ts.index < time_10_before]
    prediction_head = ts[time_10_before <= ts.index]

    if len(past_body) > len(prediction_head) and len(prediction_head) > 10:

        max_values,max_indices,past_body,prediction_head = normalize(past_body,prediction_head)

        max_prices = max_values['prices']
        max_market_caps = max_values['market_caps']
        max_total_volumns = max_values['total_volumes']

        max_prices_index = max_indices['prices']
        max_market_caps_index = max_indices['market_caps']
        max_total_volumns_index = max_indices['total_volumes']
        # ---------------------------------------------------------------

        #prediction head
        prediction_head_summary = summarize_prediction_head_dataframe(prediction_head,past_body)
        prediction_head_summary = PredictionSummary(**prediction_head_summary)
        
        #check values available
        max_hours = max(ts['elapsed_hours'])
        num_price_values:int = np.isreal(ts['prices'].values).sum()
        num_market_cap_values:int = np.isreal(ts['market_caps'].values).sum()
        num_volume_values:int = np.isreal(ts['total_volumes'].values).sum()
        are_values_same = (num_price_values == num_market_cap_values == num_volume_values)

        tsmd = CoinTimeseriesMetadata(id=coin_id,
                                    ts=ts,
                                    past_body=past_body,
                                    prediction_head=prediction_head,
                                    max_time = max_time,
                                    min_time = min_time,
                                    time_10_before=time_10_before,
                                    max_hours = max_hours,
                                    num_price_values = num_price_values,
                                    are_values_same = are_values_same,
                                    prediction_summary=prediction_head_summary,
                                    max_prices = max_prices,
                                    max_market_caps = max_market_caps,
                                    max_total_volumns = max_total_volumns,
                                    max_prices_index = max_prices_index,
                                    max_market_caps_index = max_market_caps_index,
                                    max_total_volumns_index = max_total_volumns_index)   
        return tsmd
    else:
        return None

def valid_values_for_dataframe(values):
    if isinstance(values,pd.DataFrame):
        return False
    if isinstance(values,PredictionSummary):
        return False
    return True
    
def timeseries_metadata_to_dataframe(data_list: List[CoinTimeseriesMetadata] | Dict[str,CoinTimeseriesMetadata]) -> pd.DataFrame:
    """
    we create a data frame with the statistics of all the time series metadata
    """
    if isinstance(data_list,dict):
        data_list = list(data_list.values())
    # Convert the list of PriceChangeData instances to a list of dictionaries.
    # Each dictionary represents the attributes of a PriceChangeData instance.
    data_dicts = []
    for data_instance in data_list:
         vars_tsmd = {k:v for k,v in vars(data_instance).items() if valid_values_for_dataframe(v)}
         vars_tsmd.update(vars(data_instance.prediction_summary))
         data_dicts.append(vars_tsmd)
    # Create a pandas DataFrame from the list of dictionaries.
    df = pd.DataFrame(data_dicts)
    return df

def timeseries_and_metadata(coin_metadata:AllCoinsMetadata)->Dict[str,CoinTimeseriesMetadata]:
    """
    we create a dictionary with all the coins timeseries stored with its metadata as a dataclass object
    """
    timeseries_and_metadata = {}
    for coin_id,coin_df in tqdm(coin_metadata.df_time_series.items()):
        tsmd  = preprocess_timeseries_dataframe(coin_df,coin_id)
        timeseries_and_metadata[coin_id] = tsmd
    return timeseries_and_metadata

def get_timeseries_as_torch(timeseries_and_metadata:Dict[str,CoinTimeseriesMetadata],
                            metadata_lists:AllCoinsMetadata)->TimeSeriesTorchForTraining:
    """
    here we create all the torch objects requiered for the training of a neural network 
    style prediction for the coins time series

    returns
    -------
    TimeSeriesTorchForTraining
    """
    index_to_id = {}
    indexes = []
    lengths_past = []
    lengths_prediction = []
    time_series_ids = []
    past_tensor_list = []
    prediction_tensor_list = []

    covariates_list = []
    prediction_summary_list = []
    tsmd:CoinTimeseriesMetadata

    filter_none = lambda x: -1 if x is None else x

    coin_index = 0
    for coin_id,tsmd in tqdm(timeseries_and_metadata.items()):
        if tsmd is not None:
            if tsmd.num_price_values > 200:

                index_to_id[coin_index] = coin_id
                
                #covariates
                coin_metadata:CoinMetadata
                coin_metadata = metadata_lists.uniswap_coins[coin_id]

                covariates_list.append(torch.tensor([filter_none(coin_metadata.watchlist_portfolio_users),
                                                    filter_none(coin_metadata.sentiment_votes_up_percentage)]))
                
                time_series_ids.append(coin_id)
                tsmd.past_body.fillna(-1, inplace=True)
                tsmd.prediction_head.fillna(-1, inplace=True)
                past_tensor_list.append(torch.tensor(tsmd.past_body.values))
                prediction_tensor_list.append(torch.tensor(tsmd.prediction_head.values))
                lengths_past.append(tsmd.past_body.shape[0])
                lengths_prediction.append(tsmd.prediction_head.shape[0])

                #prediction summary
                prediction_values,prediction_keys = prediction_summary_to_tensor(tsmd.prediction_summary)
                prediction_summary_list.append(prediction_values)

                indexes.append(coin_index)
                coin_index+=1

    lengths_past = torch.tensor(lengths_past)
    lengths_prediction = torch.tensor(lengths_prediction)

    indexes = torch.tensor(indexes)
    # lengths need to be in decreasing order if enforcing_sorted=True or use enforce_sorted=False
    past_padded_sequences = pad_sequence(past_tensor_list, batch_first=True, padding_value=0)
    prediction_padded_sequences = pad_sequence(prediction_tensor_list, batch_first=True, padding_value=0)

    prediction_summary_list = torch.vstack(prediction_summary_list)
    covariates_list = torch.vstack(covariates_list)

    tsdt =  TimeSeriesTorchForTraining(time_series_ids=time_series_ids,
                                    index_to_id=index_to_id,
                                    indexes=indexes,
                                    covariates=covariates_list,
                                    lengths_past=lengths_past,
                                    lengths_prediction=lengths_prediction,
                                    past_padded_sequences=past_padded_sequences,
                                    prediction_padded_sequences=prediction_padded_sequences,
                                    prediction_summary=prediction_summary_list,
                                    prediction_keys=prediction_keys)

    #=====================================
    #save
    torch.save(tsdt,metadata_lists.torch_pathdir)
    return tsdt

if __name__=="__main__":
    coingecko_key = "CG-rkg4RTUcfEWYAQ4xUejxPpkS"
    date_string="2024-03-19"

    selected_coins = get_coins_to_download(date_string=date_string,key=coingecko_key,number_of_coins_to_download=1500,percentage_on_top = .1,number_of_pages=4,redo=False)
    all_coins_metadata = AllCoinsMetadata(date_string=date_string,coingecko_key=coingecko_key)
    some_coins_to_download = selected_coins[:1500]
    #all_coins_metadata.download_coins_metadata(coins_to_download=some_coins_to_download)
    all_coins_metadata.download_df_timeseries()

    #===============================================================
    # ONE TIME SERIES PREPROCESSING
    # SELECTED_COIN_ID = 'the-open-network' 
    #SELECTED_COIN_ID = 'leo-token'
    
    #ts = all_coins_metadata.df_time_series[SELECTED_COIN_ID]
    #tsmd = preprocess_timeseries_dataframe(ts,coin_id=SELECTED_COIN_ID)

    #===============================================================
    # ALL TIME SERIES DOWNLOAD 
    all_coins_timeseries = timeseries_and_metadata(all_coins_metadata) # dict of all timeseries metadata with ts
    
    #===============================================================
    # CREATE TORCH

    torch_data = get_timeseries_as_torch(all_coins_timeseries,all_coins_metadata)
