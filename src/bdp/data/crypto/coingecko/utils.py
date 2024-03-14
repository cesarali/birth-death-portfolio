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

class RateLimitedRequester:
    """
    This class is suposse to handle the request load to coingecko such
    that one is not 
    """
    def __init__(self, max_num_fails=5, rate_limit_per_minute=30):
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_timestamps = []
        self.num_fails = 0
        self.max_num_fails = max_num_fails
        self.downloaded_in_session = 0

    def wait_for_rate_limit(self):
        """Ensure that the rate limit is not exceeded by waiting if necessary."""
        # Current time
        current_time = time.time()

        # Filter out timestamps that are older than 60 seconds
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]

        # Check if we have reached the rate limit
        if len(self.request_timestamps) >= self.rate_limit_per_minute:
            # Calculate how long to sleep
            sleep_time = 60 - (current_time - self.request_timestamps[0])+1
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)

        # Update the list of timestamps with the current time after making the request
        self.request_timestamps.append(time.time())
    
    def wait(self,wait_time=2):
        if wait_time is not None:
            wait_time = random.sample([5,5,10,30],1)
        #wait_time = random.randint(0,3)
        time.sleep(wait_time[0])

    def up_one_fail(self):
        self.num_fails+=1

    def up_one_download(self):
        self.downloaded_in_session+=1

    def wait_and_reset(self):
        time.sleep(30)
        self.num_fails = 0

        