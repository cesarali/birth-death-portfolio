import pandas as pd
from bdp.data.crypto.coingecko.coingecko_dataclasses import PriceChangeData
from typing import List
from typing import Union, Dict
def price_change_data_to_dataframe(data_list: List[PriceChangeData] | Dict[str,PriceChangeData]) -> pd.DataFrame:
    if isinstance(data_list,dict):
        data_list = list(data_list.values())
    # Convert the list of PriceChangeData instances to a list of dictionaries.
    # Each dictionary represents the attributes of a PriceChangeData instance.
    data_dicts = [vars(data_instance) for data_instance in data_list]
    
    # Create a pandas DataFrame from the list of dictionaries.
    df = pd.DataFrame(data_dicts)
    
    return df
