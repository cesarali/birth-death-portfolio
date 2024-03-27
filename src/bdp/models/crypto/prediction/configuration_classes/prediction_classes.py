from dataclasses import dataclass, field
from typing import Optional, List
from typing import List, Optional, get_type_hints
import dataclasses

#=========================
# PAST ENCODERS CONFIGS
#=========================

@dataclass
class LSTMModelConfig:
    class_name: str = "LSTMModel"
    input_dim: int = None
    hidden_dim: int = None
    layer_num: int = None
    output_dim: int = None

#==========================
# PREDICTION HEAD CONFIGS
#==========================
    
@dataclass
class MLPRegressionHeadConfig:
    class_name: str = "MLPRegressionHead"
    input_dim: int = None
    hidden_dims: List[int] = None
    output_dim: int = None

#==========================
# PREDICTION MODELS 
@dataclass
class PredictionModel:
    PastEncoder: LSTMModelConfig
    PredictionHead: MLPRegressionHeadConfig

@dataclass
class TrainingParameters:
    learning_rate: float
    num_epochs: int
    device:str

    debug: bool = True
    metric_to_save:List[str] = None
    save_model_epochs: int = 2
    save_model_metrics_stopping: bool = False 
    save_model_metrics_warming:bool = False
    warm_up_best_model_epoch: int = 10
    save_model_test_stopping: bool = True

    clip_grad:bool = True
    clip_max_norm:float = 10.
    
@dataclass
class DataLoaderParameters:
    data_dir: str
    batch_size: int
    shuffle: bool
    num_workers: int
    training_split: float

@dataclass
class ExperimentMetaData:
    name: str
    experiment_dir: Optional[str] = None
    descriptor: Optional[str] = None
    
    experiment_name:str = None
    experiment_type:str= None
    experiment_indentifier: Optional[str] = None
    results_dir:str = None
    
@dataclass
class SummaryPredictionConfig:
    ExperimentMetaData: ExperimentMetaData
    PredictionModel: PredictionModel
    TrainingParameters: TrainingParameters
    DataLoaderParameters: DataLoaderParameters

def load_dataclass_from_dict(dataclass_type, data):
    field_types = get_type_hints(dataclass_type)
    
    # Initialize arguments for the dataclass constructor
    init_args = {}
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            # Check if the field type is a dataclass and the value is a dict
            if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
                # Recursively load the nested dataclass
                init_args[field_name] = load_dataclass_from_dict(field_type, value)
            elif isinstance(field_type, type(List)) and isinstance(value, list) and dataclasses.is_dataclass(field_type.__args__[0]):
                # Handle lists of nested dataclasses
                element_type = field_type.__args__[0]
                init_args[field_name] = [load_dataclass_from_dict(element_type, elem) for elem in value]
            else:
                # Use the value as is for simple fields
                init_args[field_name] = value
        elif hasattr(dataclass_type, field_name):  # Use default if available
            init_args[field_name] = getattr(dataclass_type, field_name)
    
    # Instantiate the dataclass
    return dataclass_type(**init_args)



