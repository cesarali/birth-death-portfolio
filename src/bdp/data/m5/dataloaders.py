import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

def covariates_info():
    non_sequential_covariates = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    not_covariate = ['_id', 'id', 'max_day', 'min_day', 'lifetime', 'weird_count', 'count']
    sequential = ["count", "price", "day", "wday", "month", "year", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap"]

    data_dir = "C:/Users/cesar/Desktop/Projects/General/deep_random_fields/src/deep_fields/data/sample_data/M5/"
    json_path = os.path.join(data_dir, "covariates_basic_maps.json")
    basic_covariates_final_maps = json.load(open(json_path, "r"))
    return non_sequential_covariates, sequential,basic_covariates_final_maps, not_covariate

def get_blocks(final_dir):
    path_series = os.path.join(final_dir,"series_covariates.npy")
    path_covariates = os.path.join(final_dir,"covariates.npy")

    FULL_BLOCK_COVARIATES  = np.load(path_covariates)
    FULL_BLOCK_SERIES = np.load(path_series)
    return FULL_BLOCK_COVARIATES, FULL_BLOCK_SERIES


class M5_BLOCK(Dataset):

    def __init__(self, data_dir):

        self.non_sequential_covariates, self.sequential, self.basic_covariates_final_maps, self.not_covariate = covariates_info()
        self.FULL_BLOCK_COVARIATES, self.FULL_BLOCK_SERIES = get_blocks(data_dir)
        self.FULL_BLOCK_COVARIATES = torch.Tensor(self.FULL_BLOCK_COVARIATES).long()
        self.FULL_BLOCK_SERIES = torch.Tensor(self.FULL_BLOCK_SERIES)

        self.number_of_series,_,self.series_lenght = self.FULL_BLOCK_SERIES.size()

    def __len__(self):
        return self.number_of_series

    def __getitem__(self, idx):
        #target = self.FULL_BLOCK_SERIES[idx].unfold(dimension=2,size=28,step=1)
        return self.FULL_BLOCK_COVARIATES[idx],self.FULL_BLOCK_SERIES[idx]

if __name__=="__main__":
    final_dir = "C:/Users/cesar/Desktop/Projects/NeuralProcessesUncertainty/data/preprocessed/"
    dataset = M5_BLOCK(data_dir=final_dir)
    dataloader = DataLoader(dataset, batch_size=4,shuffle=True)
    for databatch in dataloader:
        covariates, series  = databatch
        print(covariates.shape)
        print(series.shape)
        break

