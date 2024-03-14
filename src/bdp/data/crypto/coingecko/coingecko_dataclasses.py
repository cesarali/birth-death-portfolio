from dataclasses import dataclass, fields,asdict
from typing import Optional

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
           
def prepare_dict_for_dataclasss(data:dict,dataclass_type:PriceChangeData,currency="usd")->dict:
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


if __name__ == "__main__":
    from bdp import data_path
    from pprint import pprint
    from bdp.data.crypto.coingecko.downloads import get_coin_data

    data = get_coin_data(coin_id="archangel-token",contract="0x36e43065e977bc72cb86dbd8405fae7057cdc7fd")
    #data = get_coin_data(coin_id="0chain",contract="0xb9ef770b6a5e12e45983c5d80545258aa38f3b78")
    
    data_dict = prepare_dict_for_dataclasss(data, PriceChangeData)

    pprint(asdict(PriceChangeData(**data_dict)))