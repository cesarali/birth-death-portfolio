from requests_tor import RequestsTor
from tqdm import tqdm
import requests
from dataclasses import field,fields
import time
import random

my_key = "CG-v3j5ob13whoN4a8AzYDJXDht"

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

def get_request(url,coin_gecko_key=None):
    if coin_gecko_key is not None:
        headers = {
            "x-cg-demo-api-key": coin_gecko_key,
        }
        # Sending a GET request to the URL
        response = requests.get(url,headers=headers)
    else:
        response = requests.get(url)

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
    
def filter_coin_id_and_contract(coin_ids,contract="ethereum"):
    if "id" in coin_ids.keys() and "platforms" in coin_ids.keys():
        if contract in coin_ids["platforms"]:
            return coin_ids["id"]
        else:
            return None
    else:
        return None