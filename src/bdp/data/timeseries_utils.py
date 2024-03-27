import numpy as np
import pandas as pd


from datetime import timedelta


def extend_time_series(ts):
    """
    """
    max_time = ts.index.max()  # Corrected to reference ts directly for max time
    initial_time = max_time - timedelta(days=90)  # Subtract 90 days from the max_time
    new_index = pd.date_range(initial_time, max_time, freq="h")[1:]
    extended_ts = pd.Series(np.nan,index=new_index)

    # Ensure both series are sorted
    ts1 = ts
    ts2 = extended_ts
    ts1 = ts1.sort_index()
    ts2 = ts2.sort_index()

    # Convert indices to timestamps (if they're not already)
    ts1_indices = ts1.index.view('int64')
    ts2_indices = ts2.index.view('int64')

    # For each index in ts1, find the closest index in ts2
    closest_indices = np.searchsorted(ts2_indices, ts1_indices, side='left')

    # Handle edge cases (searchsorted might return an index equal to the length of ts2_indices)
    closest_indices = np.minimum(closest_indices, len(ts2_indices) - 1)

    # Adjust indices to ensure we choose the closest match (either the found one or the previous one)
    adjust_indices = closest_indices > 0
    adjust_condition = np.abs(ts1_indices - ts2_indices[closest_indices - 1]) < np.abs(ts1_indices - ts2_indices[closest_indices])
    closest_indices[adjust_indices & adjust_condition] -= 1

    # Result: Series with ts1's indices and the corresponding closest ts2's indices as values
    ts2.values[closest_indices] = ts1.values
    return ts2