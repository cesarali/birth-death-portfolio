import pandas as pd
from pprint import pprint
from bdp.data.crypto.preprocess_summary_crypto import all_coins_summary

def in_date_range(summary_pd,date0,datef,names_only=True):
    """
    """
    if isinstance(date0,str):
        date0 = pd.to_datetime('2015-01-02')
        datef = pd.to_datetime('2021-05-01')
    selected_rows = summary_pd[(summary_pd['date_min_date'] <= date0) & (summary_pd['date_max_date'] >= datef)]
    if names_only:
        return selected_rows["filename"].values
    else:
        return selected_rows

if __name__=="__main__":
    from pathlib import Path
    data_path_string = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortafolioChoice\Codes\birth-death-portfolio\data\raw"
    path_to_data_date = Path(data_path_string) / '2021-05-10'
    summary_pd = all_coins_summary(path_to_data_date,redo=False)
    #in_date_range()
