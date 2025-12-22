from dataclasses import dataclass, asdict, fields
from typing import Literal, Optional, Type, TypeVar
from pathlib import Path
import yaml
import torch
import numpy as np

def serialize_scalars(d: dict):
    """Recursively convert NumPy and torch scalars to native Python types."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = serialize_scalars(v)
        elif isinstance(v, np.integer):
            result[k] = int(v)
        elif isinstance(v, (np.floating, torch.Tensor)):
            result[k] = float(v)
        else:
            result[k] = v
    return result

T = TypeVar("T", bound="StrictConfig")

class StrictConfig:
    """Dataclass mixin providing strict from_dict construction."""

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")

        field_names = {f.name for f in fields(cls)}
        unknown = set(data) - field_names
        if unknown:
            raise ValueError(
                f"Unknown fields for {cls.__name__}: {', '.join(sorted(unknown))}"
            )

        return cls(**data)

@dataclass(frozen=True)
class LossWeights(StrictConfig):
    pde: float = 1.0
    data: float = 1.0
    beta: float = 0.0

@dataclass
class SchedulerConfig(StrictConfig):
    type: Optional[Literal['StepLR', 'Plateau']] = None
    step_size: int = 500
    gamma: float = 1 / 3
    patience: int = 20
    factor: float = 0.5

@dataclass
class OptimizerConfig(StrictConfig):
    optimizer_type: Literal['Adam', 'AdamW', 'RMSprop'] = "Adam"
    lr: float = 1e-3

@dataclass
class NFConfig(StrictConfig):
    dim: int = 64
    num_flows: int = 10
    hidden_dim: int = 128
    num_layers: int = 3
    # Go back to 2 layers for original experiment.

# UNIFIED TRAINING CONFIG

@dataclass
class TrainingConfig:
    loss_weights: LossWeights
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    nf_config: NFConfig

    def save(self, path: Path) -> None:
        """Save all configs to a single YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            'loss_weights': asdict(self.loss_weights),
            'optimizer': asdict(self.optimizer),
            'scheduler': asdict(self.scheduler),
            'nf_config': asdict(self.nf_config),
        }

        config_dict = serialize_scalars(config_dict)

        with open(path, 'w') as f:
            yaml.safe_dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping")

        return cls(
            loss_weights=LossWeights.from_dict(data['loss_weights']),
            optimizer=OptimizerConfig.from_dict(data['optimizer']),
            scheduler=SchedulerConfig.from_dict(data['scheduler']),
            nf_config=NFConfig.from_dict(data['nf_config']),
        )
