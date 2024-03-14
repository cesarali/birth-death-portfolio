import os
import json
import time
import pandas as pd
import random
import pickle
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List
from bdp import data_path

from bdp.data.crypto.coingecko.utils import (
    get_current_and_past_timestamps,
    RateLimitedRequester,
    parse_raw_prices_to_dataframe,
)

from bdp.data.crypto.coingecko.coingecko_dataclasses import (
    PriceChangeData,
    prepare_dict_for_dataclasss,
)

from requests_tor import RequestsTor
from tqdm import tqdm

# If you use the Tor browser

headers = {
    "x-cg-demo-api-key": "CG-v3j5ob13whoN4a8AzYDJXDht"
}

def filter_coin_id_and_contract(coin_ids,contract="ethereum"):
    if "id" in coin_ids.keys() and "platforms" in coin_ids.keys():
        if contract in coin_ids["platforms"]:
            return {"id":coin_ids["id"],"contract":coin_ids["platforms"][contract]}
        else:
            return None
    else:
        return None

def get_coins_to_download(from_sorted = False,number_of_pages=4):
    """
    gets either top list with ethereum or random
    """
    data_and_contracts = get_all_coins_and_contracts_data()
    data_and_markets = get_all_coins_and_markets(number_of_pages)
    sorted_from_market_cap = [dnm["id"] for dnm in data_and_markets]

    if from_sorted:
        data_and_contracts_dict = {dnc["id"]:dnc for dnc in data_and_contracts}
        data_and_contracts_filtered = [filter_coin_id_and_contract(data_and_contracts_dict[coin_ids],contract="ethereum") for coin_ids in sorted_from_market_cap]
    else:
        data_and_contracts = get_all_coins_and_contracts_data()
        random.shuffle(data_and_contracts)
        data_and_contracts_filtered = [filter_coin_id_and_contract(coin_ids,contract="ethereum") for coin_ids in data_and_contracts]
    data_and_contracts_filtered = [cnd for cnd in data_and_contracts_filtered if cnd is not None]
    return data_and_contracts_filtered

def get_request(url):
    # Sending a GET request to the URL
    response = requests.get(url,headers=headers)

    # Checking if the request was successful
    if response.status_code == 200:
        # Convert the JSON response into a Python dictionary
        data = response.json()
        return data
    else:
        return None

def get_tor_request(url):
    # Sending a GET request to the URL
    rt = RequestsTor()
    response = rt.get(url)
    # Checking if the request was successful
    if response.status_code == 200:
        # Convert the JSON response into a Python dictionary
        data = response.json()
        return data
    else:
        return None
    
def get_all_coins_and_contracts_data(date_string):
    if date_string is None:
        date_string = str(datetime.now().date())
    coins_pathdir = data_path / "raw" / "uniswap" / date_string
    filename = coins_pathdir / "all_coins.pck"

    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    if not os.path.exists(filename):
            data = get_request(url)
            if data:
                pickle.dump(data,open(filename,"wb"))
                return data
            else:
                return None
    else:
        data = pickle.load(open(filename,"rb"))
        return data

def get_all_coins_and_markets(number_of_pages=3,tor=False):
    current_date = str(datetime.now().date())
    coins_pathdir = data_path / "raw" / "uniswap" / current_date

    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    data = []
    for page in range(number_of_pages):
        filename = f"all_coins_markets_{page}.pck"
        filename = coins_pathdir / filename
        if not os.path.exists(filename):
            url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}&sparkline=false&locale=en"
            if tor:
                data_ = get_tor_request(url)
            else:
                data_ = get_request(url)

            if data_:
                pickle.dump(data_,open(filename,"wb"))
                data.extend(data_)
        else:
            data.extend(pickle.load(open(filename,"rb")))
    return data
    
def get_coin_data(id="archangel-token",contract="0x36e43065e977bc72cb86dbd8405fae7057cdc7fd",tor=False):
    # URL of the CoinGecko API for the Archangel Token contract details
    #url = f"https://api.coingecko.com/api/v3/coins/{id}/contract/{contract}"
    url = f"https://api.coingecko.com/api/v3/coins/{id}?tickers=true&market_data=true&community_data=true&sparkline=true"

    if tor:
        data = get_tor_request(url)
    else:
        data = get_request(url)
    if data:
        data.update({"response":True,"id":id,"contract":contract})
        return data
    else:
        data = {"response":False,"id":id,"contract":contract}
        return data

def get_coin_time_series_raw(coin_id,number_of_days=90,tor=False):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={number_of_days}"
    if tor:
        data = get_tor_request(url)
    else:
        data = get_request(url)
    return data

@dataclass
class metadataLists:
    """
    And object that handles the download of coins and stores different list according to 
    1. download uniswap 2. download not uniswap 2. not downloaded (errors)
    """
    uniswap_coins = []
    not_uniswap = []
    not_downloaded = []
    date_string:str = None
    uniswap_file:Path = None
    not_downloaded_file:Path = None
    not_uniswap_file:Path = None
    uniswap_coins_date_pathdir:Path = None
    uniswap_ids_ready:List[str] = None
    not_uniswap_ids:List[str] = None
    num_total_downloads:int = 0
    num_uniswap_ids_ready:int = 0
    num_not_uniswap_ids:int = 0

    
    def __post_init__(self):
        if self.date_string is None:
            self.date_string = str(datetime.now().date())
        self.uniswap_coins_date_pathdir = data_path / "raw" / "uniswap" / self.date_string
        if not os.path.exists(self.uniswap_coins_date_pathdir):
            os.makedirs(self.uniswap_coins_date_pathdir)        
        self.uniswap_file = self.uniswap_coins_date_pathdir / f"uniswap_metadata_{self.date_string}.pck"
        self.not_downloaded_file = self.uniswap_coins_date_pathdir / f"not_downloaded_{self.date_string}.pck"
        self.not_uniswap_file = self.uniswap_coins_date_pathdir / f"not_uniswap_metadata_{self.date_string}.pck"

        if os.path.exists(self.uniswap_file):
            with open(self.uniswap_file,"rb") as file1:
                self.uniswap_coins = pickle.load(file1)
        
        if os.path.exists(self.not_uniswap_file):
            with open(self.not_uniswap_file,"rb") as file2:
                self.not_uniswap = pickle.load(file2)

        if os.path.exists(self.not_downloaded_file):
            with open(self.not_downloaded_file,"rb") as file3:
                self.not_downloaded = pickle.load(file3)

        self.uniswap_ids_ready = [coin_object.id for coin_object in self.uniswap_coins]
        self.not_uniswap_ids = [coin_object.id for coin_object in self.not_uniswap]

        self.num_uniswap_ids_ready = len(self.uniswap_ids_ready)
        self.num_not_uniswap_ids = len(self.not_uniswap_ids)
        self.num_total_downloads = self.num_uniswap_ids_ready + self.num_not_uniswap_ids
        self.include_symbols_and_name()

    def save_lists(self,redo=True):
        if redo:
            with open(self.uniswap_file,"wb") as file1:
                pickle.dump(self.uniswap_coins,file1)
            
            with open(self.not_downloaded_file,"wb") as file2:
                pickle.dump(self.not_downloaded,file2)
        
            with open(self.not_uniswap_file,"wb") as file3:
                pickle.dump(self.not_uniswap,file3)

    def list_into_dicts(self):
        self.uniswap_coins = {coin.id:coin for coin in self.uniswap_coins}
        self.not_uniswap = {coin.id:coin for coin in self.not_uniswap}

    def include_symbols_and_name(self):
        data_names = get_all_coins_and_contracts_data(self.date_string)
        self.to_symbol = {coin_dict["id"]:{"symbol":coin_dict["symbol"],"name":coin_dict["name"]} for coin_dict in data_names}

        if isinstance(self.uniswap_coins,list):
            for coin in self.uniswap_coins:
                coin:PriceChangeData
                coin.name = self.to_symbol[coin.id]["name"]
                coin.symbol = self.to_symbol[coin.id]["symbol"]
        if isinstance(self.uniswap_coins,dict):
            for coin_id,coin in self.uniswap_coins.items():
                coin:PriceChangeData
                coin.name = self.to_symbol[coin.id]["name"]
                coin.symbol = self.to_symbol[coin.id]["symbol"]


    
def download_all_uniswap_coins_metadata(trials=19,fails=10,tor=False,from_sorted=True)->metadataLists:
    """
    obtains coins id and contracts, download general data (market volume)
    checks if is uniswap and stores if so in a dataclas called PriceChangeData

    returns
    -------
    list[PriceChangeData]
    """

    metadata_list = metadataLists()
    coins_id_and_contract_to_download = get_coins_to_download(from_sorted=from_sorted)
    print("Coins to Download: ")
    print(len(coins_id_and_contract_to_download))
    rate_limiter = RateLimitedRequester()

    trial = 0
    for coin_id_and_contract in coins_id_and_contract_to_download:
        if coin_id_and_contract: # coin data downloaded if contract is ethereum
            not_ready = coin_id_and_contract["id"] not in metadata_list.uniswap_ids_ready
            probably_swap =  coin_id_and_contract["id"] not in metadata_list.not_uniswap_ids
            if not_ready and probably_swap:
                print(f"Downloading trial {trial} download n {rate_limiter.downloaded_in_session}")
                print(coin_id_and_contract)
                rate_limiter.wait_for_rate_limit()
                coin_data_downloaded = get_coin_data(**coin_id_and_contract,tor=tor) #download all coin metadata
                if coin_data_downloaded["response"]:
                    rate_limiter.up_one_download()
                    coin_data_dict = prepare_dict_for_dataclasss(coin_data_downloaded,PriceChangeData)
                    
                    coin_data_object:PriceChangeData = PriceChangeData(**coin_data_dict)
                    if coin_data_object.uniswap:
                        metadata_list.uniswap_coins.append(coin_data_object)
                    else:
                        metadata_list.not_uniswap.append(coin_data_object)
                else:
                    metadata_list.not_downloaded.append(coin_data_downloaded)
                    rate_limiter.up_one_fail()
                    if rate_limiter.num_fails > rate_limiter.max_num_fails:
                        metadata_list.save_lists()
                        rate_limiter.wait_and_reset()
            trial+=1
            if trial > trials:
                break

    metadata_list.save_lists()    
    return metadata_list
            
def get_df_timeserieses(metadata_lists:metadataLists):
    """
    Get timeseries
    """
    coin:PriceChangeData

    rate_limiter = RateLimitedRequester()
    missing_time_series = {}
    uniswap_time_series = {}
    
    if isinstance(metadata_lists.uniswap_coins,list):
        coins_metadata_list = metadata_lists.uniswap_coins
    if isinstance(metadata_lists.uniswap_coins,dict):
        coins_metadata_list = list(metadata_lists.uniswap_coins.values())

    for coin in tqdm(coins_metadata_list):#might change for other coins
        ts_filename = coin.id + ".csv"
        ts_coin_pathdir = metadata_lists.uniswap_coins_date_pathdir / ts_filename
        if not os.path.exists(ts_coin_pathdir): 
            print(f"Downloading: {coin.id}")
            rate_limiter.wait_for_rate_limit()
            timeseries_dict = get_coin_time_series_raw(coin.id,number_of_days=90) #download
            if timeseries_dict:
                ts = parse_raw_prices_to_dataframe(timeseries_dict)
                ts.to_csv(ts_coin_pathdir,index=True)
                uniswap_time_series[coin.id] = ts
            else:
                missing_time_series.append(coin.id)
        else:
            # Read the DataFrame back from the CSV file
            ts = pd.read_csv(ts_coin_pathdir, index_col=0)  # Use the first column as the index
            # Convert the index back to datetime format since it's read as a string by default
            ts.index = pd.to_datetime(ts.index)
            uniswap_time_series[coin.id] = ts

    print(f"Obtained {len(uniswap_time_series)} timeserieses Missing {len(missing_time_series)}")
    return uniswap_time_series,missing_time_series

if __name__ == "__main__":
    date_string = "2024-03-13"
    #data = get_all_coins_data()
    #0chain
    #0xb9ef770b6a5e12e45983c5d80545258aa38f3b78
    #print(get_coin_data(coin_id="0chain",
    #                    contract="0xb9ef770b6a5e12e45983c5d80545258aa38f3b78"))
    #now,past = get_current_and_past_timestamps()
    #time_series_dict = get_coin_time_series("0chain",past,now,tor=True)
    #print(time_series_dict)
    #metadata_list = download_all_uniswap_coins_metadata(trials=200,from_sorted=False)
    metadata_lists:metadataLists = metadataLists(date_string=date_string)
    #print(metadata_lists.num_total_downloads)
    #get_df_timeserieses(metadata_lists)
