from dataclasses import dataclass, asdict
from typing import Literal, Optional
import yaml
from pathlib import Path

@dataclass(frozen=True)
class LossWeights:
    pde: float = 1.0
    data: float = 1.0
    beta: float = 0.0

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(asdict(self), f)


@dataclass
class SchedulerConfig:
    type: Optional[Literal['StepLR', 'Plateau']] = None
    step_size: int = 500
    gamma: float = 1 / 3
    patience: int = 20
    factor: float = 0.5

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(asdict(self), f)


@dataclass
class OptimizerConfig:
    optimizer_type: Literal['Adam', 'AdamW', 'RMSprop'] = "Adam"
    lr: float = 1e-3

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(asdict(self), f)
