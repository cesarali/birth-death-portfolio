import json
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import json
from bs4 import BeautifulSoup
import requests
import calendar
from datetime import datetime, date
import calendar
from time import sleep
import sys
import pickle


def timestamp2date(timestamp):
    # function converts a Unix timestamp into Gregorian date
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')


def date2timestamp(date_today):
    # function coverts Gregorian date in a given format to timestamp
    return calendar.timegm(date_today.timetuple())


def fetchCryptoOHLC(fsym, tsym):
    # function fetches a crypto price-series for fsym/tsym and stores
    # it in pandas DataFrame

    cols = ['date', 'timestamp', 'open', 'high', 'low', 'close']
    lst = ['time', 'open', 'high', 'low', 'close']

    timestamp_today = calendar.timegm(datetime.today().timetuple())
    curr_timestamp = timestamp_today

    for j in range(2):
        df = pd.DataFrame(columns=cols)
        url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=" + fsym + "&tsym=" + tsym + "&toTs=" + str(
            int(curr_timestamp)) + "&limit=2000"
        #https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=30&aggregate=3&e=CCCAGG
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        dic = json.loads(soup.prettify())
        for i in range(1, 2001):
            tmp = []
            for e in enumerate(lst):
                x = e[0]
                y = dic['Data'][i][e[1]]
                if (x == 0):
                    tmp.append(str(timestamp2date(y)))
                tmp.append(y)
            if (np.sum(tmp[-4::]) > 0):
                df.loc[len(df)] = np.array(tmp)
        df.index = pd.to_datetime(df.date)
        df.drop('date', axis=1, inplace=True)
        curr_timestamp = int(df.ix[0][0])
        if (j == 0):
            df0 = df.copy()
        else:
            data = pd.concat([df, df0], axis=0)

    return data


if __name__=="__main__":
    CryptoDir = "C:/Users/cesar/Desktop/Projects/BirthDeathPortafolioChoice/data/CryptoHistory/"

    url = "https://www.cryptocompare.com/api/data/coinlist/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    data = json.loads(soup.prettify())
    data = data['Data']

    crypto_lst = sorted(list(data.keys()))

    tsym = 'USD'

    failed = []
    for f in crypto_lst:
        try:
            print(f)
            data = fetchCryptoOHLC(f, tsym)
            pickle.dump(data, open(CryptoDir + "cryptocurrencies_{0}.cpickle".format(f), "w"))
        except:
            print("Failed")
            print(f)
            print(sys.exc_info())

            failed.append(f)
            sleep(int(np.random.uniform(30, 60 * 5)))

    print("Hello World")