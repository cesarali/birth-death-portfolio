import torch
from torch import nn
from dataclasses import asdict

from bdp.models.crypto.prediction.past_encoders import (
    LSTMModel
)

from bdp.models.crypto.prediction.prediction_heads import (
    MLPRegressionHead,
)

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from bdp.models.crypto.prediction.configuration_classes.prediction_classes import SummaryPredictionConfig
from bdp.models.crypto.prediction.utils import initialize_object

class SummaryPredictionModel(nn.Module):
    
    def __init__(self, config:SummaryPredictionConfig):
        """
        """
        super(SummaryPredictionModel, self).__init__()

        # Encoder (for example)
        class_name = config.PredictionModel.PastEncoder.class_name
        params =  asdict(config.PredictionModel.PastEncoder)
        self.past_encoder = initialize_object(class_name, params)

        # Regression Head
        class_name = config.PredictionModel.PredictionHead.class_name
        params =  asdict(config.PredictionModel.PredictionHead)
        self.prediction_head = initialize_object(class_name, params)

    def forward(self, packed_input):
        # encoding
        past_encoding = self.past_encoder(packed_input)
        # prediction
        output = self.prediction_head(past_encoding)
        return output
    
    def loss_criterion(self,output,target):
        # Mask valid (non-nan) target values
        mask = ~torch.isnan(target)
    
        # Apply the mask to select non-missing values in both output and target
        output_valid = output[mask]
        target_valid = target[mask]
    
        # Compute MSE loss on non-missing values
        loss = (output_valid - target_valid).pow(2).mean()
        return loss