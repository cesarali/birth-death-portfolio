import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm 

def days_elapsed(coin_df,date0):
    df = pd.to_datetime(coin_df.iloc[:, 0])
    date0 = pd.to_datetime(date0)
    coin_df["time"] = (df - date0).dt.days
    # Given 'date0'
    return coin_df

def files_to_arrays(data_folder,files_names,date0,datef,columns_to_use=["price"]):
    """
    """
    if isinstance(date0,str):
        date0 = pd.to_datetime(date0)
    if isinstance(datef,str):
        datef = pd.to_datetime(datef)

    if isinstance(data_folder,str):
        data_folder = Path(data_folder)

    number_of_days = (datef - date0).days
    number_of_coins = len(files_names)

    data_array = np.zeros((number_of_days,number_of_coins*len(columns_to_use) + 1))
    filter_days_indexes = lambda days: (days >= 0) & (days < number_of_days)
    
    for coin_id,file_name in tqdm(enumerate(files_names)):
        coin_file_path = data_folder / file_name
        coin_df = pd.read_csv(coin_file_path)
        coin_df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)

        coin_df = days_elapsed(coin_df,date0)
        days = coin_df["time"].values
        valid_days_indexes = filter_days_indexes(days)
        number_of_columns = len(columns_to_use)

        for column_id,column_name in enumerate(columns_to_use):
            valid_days = days[valid_days_indexes]
            values = coin_df[column_name].values[valid_days_indexes]
            data_array[valid_days,(number_of_columns*coin_id)+1+column_id] = values
    return data_array


if __name__=="__main__":
    from bdp.data_.preprocess_summary_crypto import all_coins_summary
    from bdp.data_.summary_selection_crypto import in_date_range

    data_path_string = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortafolioChoice\Codes\birth-death-portfolio\data\raw"
    #path_to_data_date = Path(data_path_string) / '2021-10-24'
    #path_to_data_date = Path(data_path_string) / '2021-06-08'
    path_to_data_date = Path(data_path_string) / '2021-06-14'

    date0 = '2015-01-01'
    datef = '2021-05-01'
    summary_pd = all_coins_summary(path_to_data_date,redo=False)
    coins_names = in_date_range(summary_pd,date0,datef)
    data_array = files_to_arrays(path_to_data_date,coins_names,date0,datef,columns_to_use=["price"])
    print(data_array.shape)

