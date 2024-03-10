import os
import json
import requests
import numpy as np

import sys
import time
import random
import pymongo
import pandas as pd
from deep_fields import data_path
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta


def request_from_coingecko(coin_string, date0, datef, currency='usd'):
    headers = {
        'accept': 'application/json',
    }
    timestamp_0 = int(date0.timestamp())
    timestamp_f = int(datef.timestamp())

    params = (
        ('vs_currency', currency),
        ('from', timestamp_0),
        ('to', timestamp_f),
    )

    get_string = 'https://api.coingecko.com/api/v3/coins/{0}/market_chart/range'.format(coin_string)
    response = requests.get(get_string, headers=headers, params=params)
    return response

def request_trending():
    headers = {
        'accept': 'application/json',
    }
    get_string = 'https://api.coingecko.com/api/v3/search/trending/'
    response = requests.get(get_string, headers=headers)
    coins = json.loads(response.text)
    return coins

def get_time_span_coin(coin_id='ethereum',cg=CoinGeckoAPI(),span="full",df_=datetime.now()):
    """
    param
    -----
    coin_id: string as obtaine from
    span: full, month, day

    queries coingecko for the data starting 2015-1-1 until today for a particular coin

    cg = CoinGeckoAPI()
    price = cg.get_price(ids='bitcoin', vs_currencies='usd')
    coins_names = cg.get_coins_list()
    """
    if span == "full":
        d0_ = datetime(2015, 1, 1)
        d0 = int(d0_.timestamp())
        df = int(df_.timestamp())
    elif span == "month":
        d0_ = df_ - timedelta(days=90)
        d0 = int(d0_.timestamp())
        df = int(df_.timestamp())
    elif span == "day":
        d0_ = df_ - timedelta(hours=24)
        d0 = int(d0_.timestamp())
        df = int(df_.timestamp())

    raw_time_series = cg.get_coin_market_chart_range_by_id(id=coin_id,
                                                           vs_currency='usd',
                                                           from_timestamp=d0,
                                                           to_timestamp=df)

    prices_raw_ = np.asarray(raw_time_series["prices"])
    marketcap_raw_ = np.asarray(raw_time_series['market_caps'])
    total_volumes_ = np.asarray(raw_time_series['total_volumes'])

    if len(prices_raw_) > 0 and len(total_volumes_) > 0:
        dates_spand_price = list(map(datetime.fromtimestamp, (prices_raw_[:, 0] / 1000.)))
        dates_spand_market = list(map(datetime.fromtimestamp, (marketcap_raw_[:, 0] / 1000.)))
        dates_spand_volume = list(map(datetime.fromtimestamp, (total_volumes_[:, 0] / 1000.)))

        if span == "full":
            birth_date = dates_spand_price[0]
            birth_date = birth_date.replace(hour=0, minute=0, second=0, microsecond=0)
            survival_time = (df_ - dates_spand_price[0]).days
            non_nans = len(dates_spand_price)
        else:
            survival_time = None
            birth_date = None

        prices_ = prices_raw_[:, 1]
        market_cap_ = marketcap_raw_[:, 1]
        total_volumes_ = total_volumes_[:, 1]

        prices_series = pd.Series(prices_, index=dates_spand_price)
        marketcap_series = pd.Series(market_cap_, index=dates_spand_market)
        volumes_series = pd.Series(total_volumes_, index=dates_spand_volume)

        data_frame = pd.DataFrame({"price": prices_series,
                                   "market_cap": marketcap_series,
                                   "volume": volumes_series})

        return data_frame, survival_time, birth_date, non_nans
    else:
        return None,None,None

def gather_data(db,cg,span="full",date_gather=None,max_number=None,gaps_stats=False):
    """
    save the csv files as well as database information from coin gecko
    """
    if gaps_stats:
        GAPS = []
    coins_names = cg.get_coins_list()
    random.shuffle(coins_names)

    #get the current datetime for storing in folder
    if date_gather is None:
        date_gather = datetime.now()
        date_gather = date_gather.replace(hour=0, minute=0, second=0, microsecond=0)
        df = date_gather
        date_gather = str(date_gather.date())
    else:
        df = datetime.strptime(date_gather, "%Y-%m-%d")

    gather_dir = os.path.join(data_path, "raw", "crypto", str(date_gather))

    collection = db["birth_{0}".format(date_gather)]
    coins_ready = [a["id"] for a in collection.find()]

    print("{0} coins ready!".format(len(coins_ready))),
    trending_coins = request_trending()
    trending_coins = [coin_["item"]["id"] for coin_ in trending_coins["coins"]]
    trending_coins = [a for a in coins_names if a["id"] in trending_coins]

    markets = cg.get_coins_markets(vs_currency="usd")
    markets = [a["id"] for a in markets]
    market_coins = [a for a in coins_names if a["id"] in markets]

    for trending in trending_coins:
        coins_names.remove(trending)
        try:
            market_coins.remove(trending)
        except:
            pass

    for mark in market_coins:
        coins_names.remove(mark)

    coins_names =   coins_names + trending_coins + market_coins
    if not os.path.exists(gather_dir):
        os.makedirs(gather_dir)

    trials = 0
    while len(coins_names) > 0 and trials < 7e4:
        #coins_names = coins_names[::-1]
        coin = coins_names.pop()
        coin_id = coin["id"]
        series_file = os.path.join(gather_dir, coin_id + "_" + span + ".csv")
        if coin_id not in coins_ready:
            #try:
            trials+=1

            data_frame, survival_time, birth_date,non_nans = get_time_span_coin(coin_id=coin_id, cg=cg, span=span,df_=df)
            if gaps_stats:
                data_index = data_frame.index
                GAPS.extend([(data_index[i+1]-data_index[i]).days for i in range(len(data_index)-1)])

            if data_frame is not None:
                data_frame.to_csv(series_file)
                last_price = data_frame["price"][-1]
                last_volume = data_frame["volume"][-1]
                last_marketcap = data_frame["market_cap"][-1]
                last_date = data_frame.index[-1].to_pydatetime()

                info_ = {"last_price": last_price,
                         "last_volume": last_volume,
                         "last_marketcap": last_marketcap,
                         "survival_time": survival_time,
                         "last_date": last_date,
                         "birth_date": birth_date,
                         "non_nans":non_nans}

                mongo_entry = {**coin,**info_}
                collection.insert_one(mongo_entry)
                sleep_time = random.choice([1,1,1,1,1,11,1,1,1,1,2,2,2,2,2,
                                            1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,
                                            5, 5, 5, 5, 5, 10, 15, 30, 30]) #lazy pareto to pretend human behavior

                print("Coin {0} Stored {1}".format(coin_id,last_price))
                time.sleep(sleep_time)
            """"
            except:
                try:
                    if "Incorrect path " in sys.exc_info()[1].args[0]["error"]:
                        pass
                    else:
                        print("Problem with {0}".format(coin_id))
                        print("We sleep")
                        coins_names.append(coin)
                        print(sys.exc_info())
                        time.sleep(6)
                except:
                    print("Problem with {0}".format(coin_id))
                    print("We sleep")
                    coins_names.append(coin)
                    print(sys.exc_info())
                    time.sleep(6)
                pass
            """
        else:
            print("Coin {0} Ready!".format(coin_id))

        if max_number is not None:
            if trials > max_number:
                break
    if gaps_stats:
        return GAPS

if __name__=="__main__":

    client = pymongo.MongoClient()
    db = client["crypto"]

    cg = CoinGeckoAPI()
    GAPS = gather_data(db,cg,max_number=10,span="full",gaps_stats=True)

    print(GAPS)