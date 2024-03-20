import time
import random
import pandas as pd
from datetime import datetime, timedelta

def get_current_and_past_timestamps(days_before: int = 90):
    """
    Get the current timestamp and the timestamp for a specified number of days before today.

    Parameters:
    - days_before (int): The number of days before today for which to get the timestamp.

    Returns:
    - tuple: A tuple containing the current timestamp and the timestamp for the specified number of days before today.
    """
    now = datetime.now()
    past_date = now - timedelta(days=days_before)
    now_timestamp = now.timestamp()
    past_timestamp = past_date.timestamp()
    return now_timestamp, past_timestamp

def parse_raw_prices_to_dataframe(data)->pd.DataFrame:
    """
    Parses a dictionary containing price information into a pandas DataFrame.

    Parameters:
    - data (dict): A dictionary with a keys 'prices', 'market_caps', 'total_volumes' 
        containing a list of [timestamp, value] pairs.

    Returns:
    - DataFrame: A pandas DataFrame with columns 'timestamp' and 'price'.
    """
    dfs = []
    for key in data.keys():
        # Extract timestamps and values
        timestamps, values = zip(*data[key])
        
        # Convert timestamps from milliseconds to datetime
        timestamps = pd.to_datetime(timestamps, unit='ms')
        
        # Create a DataFrame for the current series
        df = pd.DataFrame(data=values, index=timestamps, columns=[key])
        dfs.append(df)

    # Combine all DataFrames into a single DataFrame, aligning by index (timestamp)
    final_df = pd.concat(dfs, axis=1)
    return final_df



        