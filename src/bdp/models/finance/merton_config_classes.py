from dataclasses import dataclass

from dataclasses import dataclass
import os

@dataclass
class MertonJumpConfig:
    # Default parameter values
    jump_size_scale_prior: float = 1.0
    jump_size_a: float = 0.5
    jump_size_b: float = 1.0
    jump_arrival_alpha: float = 0.5
    jump_arrival_beta: float = 0.5
    returns_mean_a: float = 1.0
    returns_mean_b: float = 1.0
    diffusion_covariance_normalization: float = 0.5
    
    # Parameters to be provided during class instantiation
    number_of_processes: int
    number_of_realizations: int
    model_path: str

    def __post_init__(self):
        # This method is called after the class is instantiated.
        # It's useful for any additional initialization or validation.
        self.model_path = os.path.join(self.model_path, 'results')

@dataclass
class InferenceConfig:
    # Numerical settings
    nmc: int = 1000
    burning: int = 200
    metrics_logs: int = 200

    # Model training configurations
    train_diffusion_covariance: bool = True
    train_expected_returns: bool = True
    train_jumps_arrival: bool = True
    train_jumps_size: bool = True
    train_jumps_intensity: bool = True
    train_jumps_mean: bool = True
    train_jumps_covariance: bool = True