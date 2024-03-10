import os
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

def save_dataframe(df, filepath, format='csv'):
    """
    Saves a DataFrame to a file.
    
    Args:
    - df: Pandas DataFrame to save.
    - filepath: Path and name of the file to save the DataFrame.
    - format: Format to save the DataFrame in ('csv' or 'pickle'). Defaults to 'csv'.
    """
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'pickle'.")

    """
    Loads a DataFrame from a file.
    
    Args:
    - format: Format of the file to load the DataFrame from ('csv' or 'pickle'). Defaults to 'csv'.
    
    Returns:
    - df: The loaded Pandas DataFrame.
    """
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'pickle'.")

def load_dataframe(filepath, format='csv'):
    """
    Loads a DataFrame from a file.
    
    Args:
    - filepath: Path and name of the file to load the DataFrame from.
    - format: Format of the file to load the DataFrame from ('csv' or 'pickle'). Defaults to 'csv'.
    
    Returns:
    - df: The loaded Pandas DataFrame.
    """
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'pickle'.")
    
def process_file(file_path, filename):
    """
    Function to process each file and update the summary with detailed stats for each column
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df.iloc[:, 0])  # Assuming the date is in the first column
    
    file_summary = {'filename': filename}
    
    # Loop through each column, except the datetime index, to gather stats
    for col in df.columns[1:]:  # Skip the first column assumed to be datetime
        col_data = {
            f'{col}_min_value': df[col].min(),
            f'{col}_max_value': df[col].max(),
            f'{col}_min_date': df.loc[df[col].idxmin(), 'date'],
            f'{col}_max_date': df.loc[df[col].idxmax(), 'date']
        }
        file_summary.update(col_data)
    
    # Additional summary information
    file_summary['num_rows'] = len(df)
    
    return file_summary

def all_coins_summary(folder_path):
    """
    We read all .csv files containing a coin and create a pandas data frame that summarizes how many entries 
    as well as max and min values attained for each coin in terms of date, market cap, volume, and price.

    Parameters
    ----------
    folder path 

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(folder_path,str):
        folder_path = Path(folder_path)
    summary_path = folder_path / "files_summary.csv"
    if not os.path.exists(summary_path):
        files_list = []
        # Iterate over each file in the folder and process it
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith('.csv'):  # Ensure the file is a CSV
                file_path = os.path.join(folder_path, filename)
                try:
                    file_info = process_file(file_path, filename)
                    files_list.append(file_info)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # Create a DataFrame from the collected file summaries
        summary_df = pd.DataFrame(files_list)
        summary_df = summary_df.sort_values(by='market_cap_max_value', ascending=False)
        save_dataframe(summary_df, summary_path, format='csv')
    else:
        summary_df = load_dataframe(summary_path, format='csv')

    return summary_df

if __name__=="__main__":
    from pathlib import Path
    data_path_string = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortafolioChoice\Codes\birth-death-portfolio\data\raw"
    path_to_data_date = Path(data_path_string) / '2021-10-24'


    summary_pd = all_coins_summary(path_to_data_date)
    pprint(summary_pd.head())
