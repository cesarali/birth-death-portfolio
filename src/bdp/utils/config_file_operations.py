import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List,Dict,Type

from bdp.models.crypto.prediction.configuration_classes.prediction_classes import (
    SummaryPredictionConfig,
    ExperimentMetaData,
    LSTMModelConfig,
    MLPRegressionHeadConfig,
    PredictionModel,
    TrainingParameters,
    DataLoaderParameters
)

# Assuming all your dataclass imports are already here

# Registry for mapping class names to classes
config_class_registry: Dict[str, Type] = {
    "LSTMModel": LSTMModelConfig,
    "MLPRegressionHead": MLPRegressionHeadConfig,
    # Add more mappings as necessary
}

def get_config_class(class_name: str) -> Type:
    """
    Retrieve the class type from the registry based on class_name.
    """
    if class_name in config_class_registry:
        return config_class_registry[class_name]
    else:
        raise ValueError(f"Unknown class name: {class_name}")

def dynamic_load_config_from_yaml(yaml_file_path: str) -> SummaryPredictionConfig:
    with open(yaml_file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Dynamically loading ExperimentMetaData
    experiment_meta_data = ExperimentMetaData(**config_dict['ExperimentMetaData'])
    
    # Dynamically parsing PredictionModel
    past_encoder_class = get_config_class(config_dict['PredictionModel']['PastEncoder']['class_name'])
    past_encoder_config = past_encoder_class(**config_dict['PredictionModel']['PastEncoder'])
    prediction_head_class = get_config_class(config_dict['PredictionModel']['PredictionHead']['class_name'])
    prediction_head_config = prediction_head_class(**config_dict['PredictionModel']['PredictionHead'])
    prediction_model = PredictionModel(PastEncoder=past_encoder_config, PredictionHead=prediction_head_config)
    
    # Assuming TrainingParameters and DataLoaderParameters classes are fixed
    training_parameters = TrainingParameters(**config_dict['TrainingParameters'])
    data_loader_parameters = DataLoaderParameters(**config_dict['DataLoaderParameters'])
    
    # Creating SummaryPredictionConfig with dynamic components
    summary_prediction_config = SummaryPredictionConfig(
        ExperimentMetaData=experiment_meta_data,
        PredictionModel=prediction_model,
        TrainingParameters=training_parameters,
        DataLoaderParameters=data_loader_parameters
    )
    
    return summary_prediction_config

# Example usage
# config = load_config_from_yaml('path_to_your_yaml_file.yaml')
# print(asdict(config))
