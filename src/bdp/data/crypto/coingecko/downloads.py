from dataclasses import dataclass
import os
import json
from pathlib import Path
import time
from typing import List
import pandas as pd
import random
import pickle
import requests
from datetime import datetime, timedelta
from bdp import data_path

from bdp.data.crypto.coingecko.downloads_utils import (
    get_request,
    get_tor_request,
    filter_coin_id_and_contract,
    RateLimitedRequester
)

from requests_tor import RequestsTor
from tqdm import tqdm

# If you use the Tor browser

def get_all_coins_and_contracts_data(date_string,key):
    if date_string is None:
        date_string = str(datetime.now().date())
    coins_pathdir = data_path / "raw" / "uniswap" / date_string
    filename = coins_pathdir / "all_coins.pck"

    if not os.path.exists(coins_pathdir):
        os.makedirs(coins_pathdir)
        
    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    if not os.path.exists(filename):
        data = get_request(url,key)
        if data:
            pickle.dump(data,open(filename,"wb"))
            return data
        else:
            return None
    else:
        data = pickle.load(open(filename,"rb"))
        return data
    
def get_all_coins_and_markets(date_string,key=None,number_of_pages=3,tor=False):
    if date_string is None:
        date_string = str(datetime.now().date())
    coins_pathdir = data_path / "raw" / "uniswap" / date_string
    if not os.path.exists(coins_pathdir):
        os.makedirs(coins_pathdir)

    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    data = []
    for page in range(number_of_pages):
        filename = f"all_coins_markets_{page}.pck"
        filename = coins_pathdir / filename
        if not os.path.exists(filename):
            url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}&sparkline=false&locale=en"
            if tor:
                data_ = get_tor_request(url,key)
            else:
                data_ = get_request(url,key)

            if data_:
                pickle.dump(data_,open(filename,"wb"))
                data.extend(data_)
        else:
            data.extend(pickle.load(open(filename,"rb")))
    return data

def get_coins_to_download(date_string=None,
                          key=None,
                          number_of_coins_to_download=3000,
                          percentage_on_top = .1,
                          number_of_pages=4,
                          redo=True):
    """
    gets either top list with ethereum or random
    """
    if date_string is None:
        date_string = str(datetime.now().date())
    coins_pathdir = data_path / "raw" / "uniswap" / date_string
    if not os.path.exists(coins_pathdir):
        os.makedirs(coins_pathdir)

    filename = f"selected_coins.pck"
    filename = coins_pathdir / filename
    if not os.path.exists(filename) or redo:
        number_on_top = int(percentage_on_top*number_of_coins_to_download)
        number_on_random = number_of_coins_to_download - number_on_top

        data_1 = get_all_coins_and_contracts_data(date_string,key)
        random.shuffle(data_1)

        print(f"Coins available: {len(data_1)}")
        print(len(data_1))
        id_to_contracts = {coin["id"]:coin for coin in data_1}

        data_2 = get_all_coins_and_markets(date_string,key,number_of_pages)
        sorted_ids = [coin["id"] for coin in data_2]

        ids_to_download = []
        from_top = 0
        while from_top < min(len(sorted_ids),number_on_top):
            top_id = sorted_ids[from_top]
            coin_ids_and_contract = id_to_contracts[top_id]
            filtered_ = filter_coin_id_and_contract(coin_ids_and_contract)
            if filtered_:
                ids_to_download.append(filtered_)
            from_top+=1

        number_on_random = number_of_coins_to_download - len(ids_to_download)
        on_random = 0
        for coin in data_1:
            coin_id = coin["id"]
            if coin_id not in ids_to_download:
                coin_ids_and_contract = id_to_contracts[coin_id]
                filtered_ = filter_coin_id_and_contract(coin_ids_and_contract)
                if filtered_:
                    ids_to_download.append(filtered_)
                    on_random+=1
                if on_random > number_on_random:
                    break
        pickle.dump(ids_to_download,open(filename,"wb"))
    else:
        ids_to_download = pickle.load(open(filename,"rb"))
    return ids_to_download
    
def get_one_coin_metadata(id="archangel-token",key=None):
    # URL of the CoinGecko API for the Archangel Token contract details
    url = f"https://api.coingecko.com/api/v3/coins/{id}?tickers=true&market_data=true&community_data=true&sparkline=true"

    data = get_request(url,key)
    if data:
        data.update({"response":True,"id":id})
        return data
    else:
        data = {"response":False,"id":id}
        return data

def get_coin_timeseries_raw(coin_id,key=None,number_of_days=90,tor=False):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={number_of_days}"
    if tor:
        data = get_tor_request(url)
    else:
        data = get_request(url,key)
    return data
    
if __name__=="__main__":
    date_string = None
    coingecko_key = "CG-rkg4RTUcfEWYAQ4xUejxPpkS"
    selected_coins = get_coins_to_download(date_string=None,key=coingecko_key,number_of_coins_to_download=3000,percentage_on_top = .1,number_of_pages=5)
    print(selected_coins)