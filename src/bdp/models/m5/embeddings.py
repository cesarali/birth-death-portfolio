import os
import torch
import json
import numpy as np
import pymongo
from matplotlib import pyplot as plt
import torch

from torch import nn
from pprint import pprint

import torchtext
#from torchtext.data.batch import Batch
#from torchtext.data import Field, Example
#from torchtext.data import BucketIterator,BPTTIterator

from deep_fields.data.m5.dataloaders import covariates_info

#non_sequential_covariates, sequential, basic_covariates_final_maps, _ = covariates_info()

def create_embeddings(embeddings_parameters):
    EMBEDDINGS = {}
    for k,v in basic_covariates_final_maps.items():
        if k != "price":
            try:
                embedding_size = len(v['id_to_str'])
                embedding = nn.Embedding(num_embeddings=embedding_size,embedding_dim=embeddings_parameters[k]["embedding"])
                EMBEDDINGS[k] = embedding
            except:
                embedding_size = len(v['id_to_str'])
                embedding = nn.Embedding(num_embeddings=embedding_size,embedding_dim=10)
                EMBEDDINGS[k] = embedding
        else:
            EMBEDDINGS[k] = nn.Linear(1,embeddings_parameters[k]["embedding"])
    return EMBEDDINGS

if __name__=="__main__":
    create_embeddings()
