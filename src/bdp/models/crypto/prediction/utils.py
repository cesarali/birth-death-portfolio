import os

from bdp.models.crypto.prediction.prediction_heads import *
from bdp.models.crypto.prediction.past_encoders import *

def initialize_object(class_name, params):
    # Dynamically get the class from globals() and instantiate it with **params
    globals_ = globals()
    if class_name in globals_:
        return globals_[class_name](**params)
    else:
        raise ValueError(f"Class {class_name} not found")