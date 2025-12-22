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
    pde: float
    data: float


@dataclass
class SchedulerConfig(StrictConfig):
    type: Optional[Literal['StepLR', 'Plateau']]
    step_size: int
    gamma: float
    patience: int
    factor: float

@dataclass
class OptimizerConfig(StrictConfig):
    optimizer_type: Literal['Adam', 'AdamW', 'RMSprop']
    lr: float


@dataclass
class NFConfig(StrictConfig):
    dim: int
    num_flows: int
    hidden_dim: int
    num_layers: int


@dataclass
class SecondStageConfig(StrictConfig):
    nf_config: NFConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    @classmethod
    def from_dict(cls, data: dict) -> 'SecondStageConfig':
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")

        return cls(
            nf_config=NFConfig.from_dict(data['nf_config']),
            optimizer=OptimizerConfig.from_dict(data['optimizer']),
            scheduler=SchedulerConfig.from_dict(data['scheduler'])
        )


# FIRST STAGE CONFIG (for initial model training)
@dataclass
class FirstStageConfig(StrictConfig):
    loss_weights: LossWeights
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    def save(self, path: Path) -> None:
        """Save first stage config to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            'loss_weights': asdict(self.loss_weights),
            'optimizer': asdict(self.optimizer),
            'scheduler': asdict(self.scheduler),
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
    def load(cls, path: Path) -> 'FirstStageConfig':
        """Load first stage configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping")

        return cls(
            loss_weights=LossWeights.from_dict(data['loss_weights']),
            optimizer=OptimizerConfig.from_dict(data['optimizer']),
            scheduler=SchedulerConfig.from_dict(data['scheduler']),
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'FirstStageConfig':
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")

        return cls(
            loss_weights=LossWeights.from_dict(data['loss_weights']),
            optimizer=OptimizerConfig.from_dict(data['optimizer']),
            scheduler=SchedulerConfig.from_dict(data['scheduler'])
        )


# UNIFIED TRAINING CONFIG (for complete pipeline with both stages)
@dataclass
class TrainingConfig:
    first_stage: FirstStageConfig
    second_stage: SecondStageConfig

    def save(self, path: Path) -> None:
        """Save all configs to a single YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            'first_stage': {
                'loss_weights': asdict(self.first_stage.loss_weights),
                'optimizer': asdict(self.first_stage.optimizer),
                'scheduler': asdict(self.first_stage.scheduler),
            },
            'second_stage': {
                'nf_config': asdict(self.second_stage.nf_config),
                'optimizer': asdict(self.second_stage.optimizer),
                'scheduler': asdict(self.second_stage.scheduler),
            }
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
            first_stage=FirstStageConfig.from_dict(data['first_stage']),
            second_stage=SecondStageConfig.from_dict(data['second_stage']),
        )
