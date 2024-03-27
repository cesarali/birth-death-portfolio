import os
import pickle
from typing import Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
from bdp import data_path
from typing import List,Dict,Tuple
from dataclasses import dataclass, fields,asdict
from tqdm import tqdm
from bdp.data.crypto.coingecko.downloads_utils import RateLimitedRequester

from bdp.data.crypto.coingecko.downloads import (
    get_coins_to_download,
    get_one_coin_metadata,
    get_all_coins_and_contracts_data,
    get_coin_timeseries_raw
)

from bdp.data.crypto.coingecko.utils import parse_raw_prices_to_dataframe

@dataclass
class PriceChangeData:
    """
    After proprocessing stores values from:
    
    """
    id:Optional[str] = None
    contract:Optional[str] = None
    symbol:Optional[str] = None
    name:Optional[str] = None

    #SENTIMENT
    sentiment_votes_up_percentage:Optional[int] = None
    watchlist_portfolio_users:Optional[int] = None
    market_cap_rank:Optional[int] = None
    #MARKET VALUES
    price_change_percentage_24h: Optional[float] = None
    price_change_percentage_7d: Optional[float] = None
    price_change_percentage_14d: Optional[float] = None
    price_change_percentage_30d: Optional[float] = None
    price_change_percentage_60d: Optional[float] = None
    price_change_percentage_200d: Optional[float] = None
    price_change_percentage_1y: Optional[float] = None
    price_change_percentage_1h_in_currency: Optional[float] = None
    price_change_percentage_24h_in_currency: Optional[float] = None
    price_change_percentage_7d_in_currency: Optional[float] = None
    price_change_percentage_14d_in_currency: Optional[float] = None
    price_change_percentage_30d_in_currency: Optional[float] = None
    price_change_percentage_60d_in_currency: Optional[float] = None
    price_change_percentage_200d_in_currency: Optional[float] = None
    price_change_percentage_1y_in_currency: Optional[float] = None
    current_price: Optional[float] = None
    total_value_locked: Optional[float] = None
    mcap_to_tvl_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    uniswap:Optional[bool] = False

@dataclass
class CoinMetadata:
    """
    After proprocessing stores values from:

    """
    id:Optional[str] = None
    contract:Optional[str] = None
    symbol:Optional[str] = None
    name:Optional[str] = None

    #SENTIMENT
    sentiment_votes_up_percentage:Optional[int] = None
    watchlist_portfolio_users:Optional[int] = None
    market_cap_rank:Optional[int] = None
    #MARKET VALUES
    price_change_percentage_24h: Optional[float] = None
    price_change_percentage_7d: Optional[float] = None
    price_change_percentage_14d: Optional[float] = None
    price_change_percentage_30d: Optional[float] = None
    price_change_percentage_60d: Optional[float] = None
    price_change_percentage_200d: Optional[float] = None
    price_change_percentage_1y: Optional[float] = None
    price_change_percentage_1h_in_currency: Optional[float] = None
    price_change_percentage_24h_in_currency: Optional[float] = None
    price_change_percentage_7d_in_currency: Optional[float] = None
    price_change_percentage_14d_in_currency: Optional[float] = None
    price_change_percentage_30d_in_currency: Optional[float] = None
    price_change_percentage_60d_in_currency: Optional[float] = None
    price_change_percentage_200d_in_currency: Optional[float] = None
    price_change_percentage_1y_in_currency: Optional[float] = None
    current_price: Optional[float] = None
    total_value_locked: Optional[float] = None
    mcap_to_tvl_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    uniswap:Optional[bool] = False

#===========================================================================================
# FUNCTIONS UTILS DICT TO CoinMetadata CLASS
#===========================================================================================
def obtain_tickers(data_tickers):
    bid_ask_spread_percentage = None
    uniswap = False
    if isinstance(data_tickers,list):
        for ticker in data_tickers:
            if isinstance(ticker,dict):
                if "market" in ticker.keys():
                    if 'uniswap' in ticker["market"]["identifier"]:
                        uniswap = True
                bid_ask_spread_percentage = ticker['bid_ask_spread_percentage']
    return {"uniswap":uniswap,
            "bid_ask_spread_percentage":bid_ask_spread_percentage}

def filter_dict_for_dataclass(input_dict, dataclass_type,currency="usd"):
    dataclass_fields = {f.name for f in fields(dataclass_type)}
    filtered_dict = {}
    for k, v in input_dict.items():
        if k in dataclass_fields:
            if not isinstance(v,dict):
                filtered_dict[k] = v
            else:
                if currency in v.keys():
                    filtered_dict[k] = v[currency] 
    return filtered_dict
           
def prepare_dict_for_dataclass(data:dict,dataclass_type:PriceChangeData,currency="usd")->dict:
    """
    here data comes from the gecko api, the idea is to prepare the dict such that
    we are able to initialize the classes like:

    PriceChangeData(**filter_data_dict)
    """
    data_dict = {}
    data_dict.update(data)
    if "market_data"in data.keys():
        data_dict.update(data["market_data"])
    if "tickers" in data.keys():
        tickers_data = obtain_tickers(data["tickers"])
        data_dict.update(tickers_data)
    data_dict = filter_dict_for_dataclass(data_dict,dataclass_type,currency)
    return data_dict

@dataclass
class AllCoinsMetadata:
    """
    And object that handles the download of coins and stores different list according to wether they were already downloaded 
    or if they are in fact uniswap or not.

    It also holds all the paths for the coins metadata, timeseries, torch.

    1. download uniswap 
    2. download not uniswap 
    3. not downloaded (errors)
    """
    uniswap_coins = []
    not_uniswap = []
    fail_to_download = []

    missing_time_series = []
    df_time_series = {}

    date_string:str = None
    coingecko_key:str = None

    all_coins_date_pathdir:Path = None

    uniswap_file:Path = None
    not_downloaded_file:Path = None
    not_uniswap_file:Path = None

    torch_pathdir:Path = None
    raw_metadata_dir:Path = None

    uniswap_ids_ready:List[str] = None
    not_uniswap_ids:List[str] = None

    num_total_downloads:int = 0
    num_uniswap_ids_ready:int = 0
    num_not_uniswap_ids:int = 0
    redo_names:bool = False # THIS IS FOR THE DEPRECATED PRICECHANGE OBJECTS
    def __post_init__(self):
        """
        We create all the dirs and files paths for the coins
        """
        if self.date_string is None:
            self.date_string = str(datetime.now().date())
        self.all_coins_date_pathdir = data_path / "raw" / "uniswap" / self.date_string
        if not os.path.exists(self.all_coins_date_pathdir):
            os.makedirs(self.all_coins_date_pathdir)

        self.uniswap_file = self.all_coins_date_pathdir / f"uniswap_metadata_{self.date_string}.pck"
        self.not_downloaded_file = self.all_coins_date_pathdir / f"not_downloaded_{self.date_string}.pck"
        self.not_uniswap_file = self.all_coins_date_pathdir / f"not_uniswap_metadata_{self.date_string}.pck"
        self.torch_pathdir = self.all_coins_date_pathdir / "preprocess_data_torch.tr"
        self.raw_metadata_dir = self.all_coins_date_pathdir / "raw_metadata/"

        if not os.path.exists(self.raw_metadata_dir):
            os.makedirs(self.raw_metadata_dir)

        if os.path.exists(self.uniswap_file):
            with open(self.uniswap_file,"rb") as file1:
                self.uniswap_coins = pickle.load(file1)

        if os.path.exists(self.not_uniswap_file):
            with open(self.not_uniswap_file,"rb") as file2:
                self.not_uniswap = pickle.load(file2)

        if os.path.exists(self.not_downloaded_file):
            with open(self.not_downloaded_file,"rb") as file3:
                self.fail_to_download = pickle.load(file3)

        self.list_into_dicts()

        self.uniswap_ids_ready = [coin_object.id for key,coin_object in self.uniswap_coins.items()]
        self.not_uniswap_ids = [coin_object.id for key,coin_object in self.not_uniswap.items()]
        for uniswap_coin_ready in self.uniswap_ids_ready:
            if uniswap_coin_ready in self.fail_to_download:
                self.fail_to_download.remove(uniswap_coin_ready)

        for not_uniswap_coin_ready in self.not_uniswap_ids:
            if not_uniswap_coin_ready in self.fail_to_download:
                self.fail_to_download.remove(not_uniswap_coin_ready)

        self.num_uniswap_ids_ready = len(self.uniswap_ids_ready)
        self.num_not_uniswap_ids = len(self.not_uniswap_ids)
        self.num_total_downloads = self.num_uniswap_ids_ready + self.num_not_uniswap_ids

        if self.redo_names:
            self.include_symbols_and_name()

    def save_lists(self,redo=True):
        if redo:
            with open(self.uniswap_file,"wb") as file1:
                pickle.dump(self.uniswap_coins,file1)

            with open(self.not_downloaded_file,"wb") as file2:
                pickle.dump(self.fail_to_download,file2)

            with open(self.not_uniswap_file,"wb") as file3:
                pickle.dump(self.not_uniswap,file3)

    def list_into_dicts(self):
        if isinstance(self.uniswap_coins,list):
            self.uniswap_coins = {coin.id:coin for coin in self.uniswap_coins}
        if isinstance(self.not_uniswap,list):
            self.not_uniswap = {coin.id:coin for coin in self.not_uniswap}

    def include_symbols_and_name(self):
        data_names = get_all_coins_and_contracts_data(self.date_string,self.coingecko_key)
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

    def download_coins_metadata(
            self,
            coins_to_download=None,
            trials=10000,
            from_sorted=True):
        """
        obtains coins id and contracts, download general data (market volume)
        checks if is uniswap and stores if so in a list of data classes dataclass called CoinMetadata

        """
        if coins_to_download is None:
            coins_to_download = get_coins_to_download(from_sorted=from_sorted)

        print("Coins to Download: ")
        print(len(coins_to_download))

        rate_limiter = RateLimitedRequester()
        def download(in_swap,in_not_swap,in_fail_to_download):
            if in_fail_to_download:
                return True
            if in_swap or in_not_swap:
                return False
            return True

        trial = 0
        for coin_id in coins_to_download:
            if coin_id: # coin data downloaded if contract is ethereum
                in_swap = coin_id in self.uniswap_ids_ready
                in_not_swap =  coin_id in self.not_uniswap_ids
                in_fail_to_download = coin_id in self.fail_to_download

                if download(in_swap,in_not_swap,in_fail_to_download):
                    print(f"Downloading trial {trial} download n {rate_limiter.downloaded_in_session}")
                    print(coin_id)
                    rate_limiter.wait_for_rate_limit()
                    coin_data_downloaded = get_one_coin_metadata(coin_id,key=self.coingecko_key) #download all coin metadata

                    if coin_data_downloaded["response"]:
                        
                        rate_limiter.up_one_download()

                        coin_data_dict = prepare_dict_for_dataclass(coin_data_downloaded,CoinMetadata)
                        coin_data_object:CoinMetadata = CoinMetadata(**coin_data_dict)

                        if coin_data_object.uniswap:
                            self.uniswap_coins[coin_id] = coin_data_object
                            self.uniswap_ids_ready.append(coin_id)
                        else:
                            self.not_uniswap[coin_id] = coin_data_object
                            self.not_uniswap_ids.append(coin_id)

                        if coin_id in self.fail_to_download:
                            self.fail_to_download.remove(coin_id)    
                            
                    else:
                        self.fail_to_download.append(coin_data_downloaded["id"])
                        rate_limiter.up_one_fail()
                        if rate_limiter.num_fails > rate_limiter.max_num_fails:
                            self.save_lists()
                            rate_limiter.wait_and_reset()
                    
                    #save every so trials 
                    trial+=1
                    if trial % 100 == 0:
                        self.save_lists()

                    if trial > trials:
                        break
        self.save_lists()

    def download_df_timeseries(self,only_uniswap=True):
        """
        Get timeseries
        """
        rate_limiter = RateLimitedRequester()

        if only_uniswap:
            coins_metadata_list = self.uniswap_ids_ready
        else:
            coins_metadata_list = self.uniswap_ids_ready + self.not_uniswap_ids

        for coin_id in tqdm(coins_metadata_list):#might change for other coins
            ts_filename = coin_id + ".csv"
            ts_coin_pathdir = self.all_coins_date_pathdir / ts_filename
            if not os.path.exists(ts_coin_pathdir): 
                print(f"Downloading: {coin_id}")
                rate_limiter.wait_for_rate_limit()
                timeseries_dict = get_coin_timeseries_raw(coin_id,key=self.coingecko_key,number_of_days=90) #download
                if timeseries_dict:
                    ts = parse_raw_prices_to_dataframe(timeseries_dict)
                    ts.to_csv(ts_coin_pathdir,index=True)
                    self.df_time_series[coin_id] = ts
                else:
                    self.missing_time_series.append(coin_id)
            else:
                # Read the DataFrame back from the CSV file
                ts = pd.read_csv(ts_coin_pathdir, index_col=0)  # Use the first column as the index
                # Convert the index back to datetime format since it's read as a string by default
                ts.index = pd.to_datetime(ts.index)
                self.df_time_series[coin_id] = ts

        print(f"Obtained {len(self.df_time_series)} timeserieses Missing {len(self.missing_time_series)}")

if __name__=="__main__":
    coingecko_key = "CG-rkg4RTUcfEWYAQ4xUejxPpkS"
    date_string="2024-03-19"

    selected_coins = get_coins_to_download(date_string=None,key=coingecko_key,number_of_coins_to_download=3000,percentage_on_top = .1,number_of_pages=4,redo=False)
    all_coins_metadata = AllCoinsMetadata(date_string=date_string,coingecko_key=coingecko_key)
    
    some_coins_to_download = selected_coins[:1000]
    all_coins_metadata.download_coins_metadata(coins_to_download=some_coins_to_download)
    all_coins_metadata.download_df_timeseries()