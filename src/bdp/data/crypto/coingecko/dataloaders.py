
from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, TensorDataset,random_split
from pathlib import Path
from bdp.models.crypto.prediction.configuration_classes.prediction_classes import SummaryPredictionConfig

@dataclass
class TimeSeriesTorchForTraining:
    time_series_ids : List[str] = None
    index_to_id :Dict[int,str] = None
    indexes: torch.Tensor = None

    lengths_past: torch.Tensor = None
    lengths_prediction: torch.Tensor = None

    covariates: torch.Tensor = None

    past_padded_sequences: torch.Tensor = None
    prediction_padded_sequences: torch.Tensor = None

    prediction_summary: torch.Tensor = None


def read_data(data_dir:Path|str)->TimeSeriesTorchForTraining:
    """
    data_path dir where all the data (including metadata and .csv files is located)
    """
    if isinstance(data_dir,dict):
        data_dir = Path(data_dir)
    torch_path_dir = data_dir / "preprocess_data_torch.tr"
    torch_data = torch.load(torch_path_dir)
    return torch_data


class TimeSeriesDataLoader:
    """
    Simple first version of a daloader for regression
    """
    def __init__(self,config:SummaryPredictionConfig):
        self.dataloader_config = config.DataLoaderParameters
        data_dir = Path(config.DataLoaderParameters.data_dir)
        torch_data = read_data(data_dir)
        # Create a TensorDataset
        indexes = torch_data.indexes
        covariates = torch_data.covariates
        lengths_past = torch_data.lengths_past
        past_padded_sequences = torch_data.past_padded_sequences
        prediction_summary = torch_data.prediction_summary

        self.check_for_nans(indexes,covariates,lengths_past,past_padded_sequences,prediction_summary)
        
        sequence_dataset = TensorDataset(indexes,covariates,past_padded_sequences,lengths_past,prediction_summary)
        self.batch_values_str = {0:"indexes",1:"covariates",2:"past_added_sequences",3:"lengths_past",4:"prediction_summary"}        

        # Define the sizes of your splits, e.g., 80% for training and 20% for testing
        train_size = int(0.8 * len(sequence_dataset))
        test_size = len(sequence_dataset) - train_size

        # Split the dataset into training and testing
        train_dataset, test_dataset = random_split(sequence_dataset, [train_size, test_size])

        # Assuming your datasets are in the form of TensorDatasets for simplicity
        # train_dataset and test_dataset should be instances of TensorDataset or similar
        # that contain your input features, sequence lengths, and target values
        self.train_loader = DataLoader(train_dataset, batch_size=self.dataloader_config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.dataloader_config.batch_size)

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader
    
    def check_for_nans(self,indexes,covariates,lengths_past,past_padded_sequences,prediction_summary):
        # Check each tensor for NaN values and print if NaN is found
        if torch.isnan(indexes).any():
            print("NaN found in indexes")
        if torch.isnan(covariates).any():
            print("NaN found in covariates")
        if torch.isnan(lengths_past).any():
            print("NaN found in lengths_past")
        if torch.isnan(past_padded_sequences).any():
            print("NaN found in past_padded_sequences")
        if torch.isnan(prediction_summary).any():
            print("NaN found in prediction_summary")