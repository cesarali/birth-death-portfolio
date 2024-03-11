import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm 

def days_elapsed(coin_df,date0,datef):
    df = pd.to_datetime(coin_df.iloc[:, 0])
    date0 = pd.to_datetime(date0)
    datef = pd.to_datetime(datef)
    coin_df["time"] = (df - date0).dt.days
    # Given 'date0'
    return coin_df

def files_to_arrays_days(data_folder,filter_coins_summary_pd,date0,datef,columns_to_use=["price"],name_to_save="price"):
    """
    this function reads the .csv downloaded by ht ecrawler and creates a numpy array 
    where the first column corresponds to the days that eah security is recorded.

    returns
    -------
    coin_data  np.array([number_of_days_after_date0,number_of_coins*number_of_columns + 1]),
    meta_data dict 
    """
    if isinstance(date0,str):
        date0 = pd.to_datetime(date0)
    if isinstance(datef,str):
        datef = pd.to_datetime(datef)
    if isinstance(data_folder,str):
        data_folder = Path(data_folder)

    files_names = filter_coins_summary_pd["filename"]
    number_of_days = (datef - date0).days
    number_of_coins = len(files_names)

    data_array = np.zeros((number_of_days,number_of_coins*len(columns_to_use) + 1))
    data_array[:,0] = np.linspace(0,number_of_days-1,number_of_days)
    filter_days_indexes = lambda days: (days >= 0) & (days < number_of_days)

    meta_data = {}
    for coin_id,file_name in tqdm(enumerate(files_names)):
        coin_file_path = data_folder / file_name
        coin_df = pd.read_csv(coin_file_path)
        coin_df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)

        coin_df = days_elapsed(coin_df,date0,datef)
        days = coin_df["time"].values
        valid_days_indexes = filter_days_indexes(days) #only positive and zero and less than maximum 
        number_of_columns = len(columns_to_use)

        meta_data[str(file_name)] = {"recorded_days":int(len(valid_days_indexes)),
                                "min_day":int(min(days)),
                                "max_days":int(max(days))}

        for column_id,column_name in enumerate(columns_to_use):
            valid_days = days[valid_days_indexes]
            values = coin_df[column_name].values[valid_days_indexes]
            data_array[valid_days,(number_of_columns*coin_id)+1+column_id] = values

    # ****************
    # print file
    array_filename = name_to_save + f"_{str(date0.date())}_{str(datef.date())}.npy"
    array_file_path = data_folder / array_filename

    meta_file_name = name_to_save + f"_{str(date0.date())}_{str(datef.date())}.json"
    meta_file_path = data_folder / meta_file_name
    
    np.save(array_file_path,data_array)
    with open(meta_file_path,"w") as file:
        json.dump(meta_data,file)    

    return data_array,meta_data

if __name__=="__main__":
    from pprint import pprint
    from bdp.data_.preprocess_summary_crypto import all_coins_summary
    from bdp.data_.summary_selection_crypto import in_date_range

    data_path_string = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortafolioChoice\Codes\birth-death-portfolio\data\raw"
    #path_to_data_date = Path(data_path_string) / '2021-10-24'
    #path_to_data_date = Path(data_path_string) / '2021-06-08'
    path_to_data_date = Path(data_path_string) / '2021-06-14'
    
    date0 = '2015-01-01'
    datef = '2021-05-01'
    summary_pd = all_coins_summary(path_to_data_date,redo=False)
    filter_coins_summary_pd = in_date_range(summary_pd,date0,datef,names_only=False)
    data_array,meta_data = files_to_arrays_days(path_to_data_date,
                                                filter_coins_summary_pd,date0,datef,columns_to_use=["price"])
    print(data_array.shape)
    pprint(meta_data)

