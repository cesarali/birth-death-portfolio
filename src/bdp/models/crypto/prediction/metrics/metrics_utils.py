import os
import sys
import json
import numpy as np
from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment
from torch.nn.utils.rnn import pack_padded_sequence
from bdp.models.crypto.prediction.past_encoders import LSTMModel
from dataclasses import dataclass

@dataclass
class MetricsAvailable:
    test_loss:str


def test_loss(experiment:SummaryPredictionExperiment):
    """
    calculates the loss for the whole test data set
    """
    dataloader = experiment.dataloader
    criterion = experiment.prediction_model.loss_criterion
    model = experiment.prediction_model
    
    if isinstance(model.past_encoder,LSTMModel):
        pack_sentences = True

    losses = []
    for databatch in dataloader.test():
        past_padded = databatch[2]
        lengths = databatch[3]
        y = databatch[4]
        if pack_sentences:
            x = pack_padded_sequence(past_padded, lengths, batch_first=True, enforce_sorted=False)
        x,y = x.float(),y.float()
        output = model(x)  # Forward pass: compute the output
        loss = criterion(output, y)  # Compute the loss
        losses.append(loss.item())
    losses = np.asarray(losses).mean()
    return losses

def log_metrics(experiment:SummaryPredictionExperiment, all_metrics, epoch, writer):
    test_loss_value = test_loss(experiment)

    #=======================================================
    # DUMP METRICS
    all_metrics.update({"test_loss_value":test_loss_value})
    metrics_file_path = experiment.experiment_files.metrics_file.format("test_loss_{0}".format(epoch))
    with open(metrics_file_path,"w") as f:
        json.dump(all_metrics,f)
    return all_metrics
