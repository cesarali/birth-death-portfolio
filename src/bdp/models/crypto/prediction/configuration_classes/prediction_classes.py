from dataclasses import dataclass, field
from typing import Optional, List


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

    debug: bool = True
    metric_to_save:List[str] = None
    save_model_epochs: int = 2
    save_model_metrics_stopping: bool = False 
    save_model_metrics_warming:bool = False
    warm_up_best_model_epoch: int = 10
    number_of_epochs: int = 100
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
    
@dataclass
class SummaryPredictionConfig:
    ExperimentMetaData: ExperimentMetaData
    PredictionModel: PredictionModel
    TrainingParameters: TrainingParameters
    DataLoaderParameters: DataLoaderParameters



