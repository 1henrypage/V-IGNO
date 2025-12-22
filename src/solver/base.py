"""
Base trainer class that handles all logging, saving, and loading logic.
Both Solver and IGNOInverter inherit from this.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class BaseTrainer(ABC):
    """
    Abstract base class that handles all the boilerplate for:
    - Directory management (shared across stages via run_dir)
    - Model saving/loading
    - TensorBoard logging
    - Train/eval mode switching
    """

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
            run_dir: Optional[Path] = None
    ):
        """
        Args:
            device: Device to use for training
            dtype: Data type for tensors
            run_dir: Shared run directory for all stages of this experiment
        """
        self.device = device
        self.dtype = dtype
        self._shared_run_dir = run_dir  # Shared across all stages

        # Forward declare
        self.model_dict: Optional[Dict[str, nn.Module]] = None
        self.optimizer = None
        self.scheduler = None
        self.writer: Optional[SummaryWriter] = None

        # Stage-specific directories (set during setup)
        self.stage_dir: Optional[Path] = None
        self.weights_dir: Optional[Path] = None
        self.tb_dir: Optional[Path] = None

    @property
    def run_dir(self) -> Optional[Path]:
        """Get the shared run directory for this experiment."""
        return self._shared_run_dir

    @run_dir.setter
    def run_dir(self, path: Path) -> None:
        """Set the shared run directory."""
        self._shared_run_dir = path

    def _setup_stage_directories(
            self,
            stage_name: str
    ) -> None:
        """
        Setup directories for this specific training stage.

        Args:
            stage_name: Name of training stage (e.g., 'solver', 'igno_stage2')
        """
        if self._shared_run_dir is None:
            raise RuntimeError(
                "run_dir must be set before calling _setup_stage_directories. "
                "This should be set by the ProblemInstance or Solver."
            )

        # Create stage-specific subdirectory
        self.stage_dir = self._shared_run_dir / stage_name
        self.weights_dir = self.stage_dir / "weights"
        self.tb_dir = self.stage_dir / "tensorboard"

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def _setup_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        if self.tb_dir is None:
            raise RuntimeError("Must call _setup_stage_directories before _setup_tensorboard")
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def save_models(
            self,
            filename: str,
            additional_state: Optional[Dict] = None
    ) -> None:
        """
        Save model weights and optional additional state.

        Args:
            filename: Name of file to save (e.g., 'best.pt', 'checkpoint_epoch_100.pt')
            additional_state: Optional dict of additional things to save (epoch, optimizer state, etc.)
        """
        if self.model_dict is None:
            raise RuntimeError("No models to save. model_dict is None.")

        save_path = self.weights_dir / filename

        state = {
            'models': {name: model.state_dict() for name, model in self.model_dict.items()}
        }

        if additional_state:
            state.update(additional_state)

        torch.save(state, save_path)

    def load_models(
            self,
            filename: str,
            load_optimizer: bool = False
    ) -> Dict:
        """
        Load model weights and return any additional state.

        Args:
            filename: Name of file to load
            load_optimizer: Whether to load optimizer state if available

        Returns:
            Dict containing any additional state that was saved
        """
        if self.model_dict is None:
            raise RuntimeError("model_dict must be initialized before loading")

        load_path = self.weights_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        # Load model weights
        if 'models' in checkpoint:
            for name, model in self.model_dict.items():
                if name in checkpoint['models']:
                    model.load_state_dict(checkpoint['models'][name])
                else:
                    raise KeyError(f"Model '{name}' not found in checkpoint")
        else:
            # Legacy format: assume checkpoint is just the state dicts
            for name, model in self.model_dict.items():
                if name in checkpoint:
                    model.load_state_dict(checkpoint[name])

        # Optionally load optimizer
        if load_optimizer and 'optimizer' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        return {k: v for k, v in checkpoint.items() if k != 'models'}

    def _activate_train(self) -> None:
        """Set all models to training mode."""
        if self.model_dict:
            for model in self.model_dict.values():
                model.train()

    def _activate_eval(self) -> None:
        """Set all models to evaluation mode."""
        if self.model_dict:
            for model in self.model_dict.values():
                model.eval()

    def close(self) -> None:
        """Close TensorBoard writer and cleanup."""
        if self.writer:
            self.writer.close()

    @abstractmethod
    def setup(self, config, **kwargs) -> None:
        """Setup models, optimizers, and logging. Must be implemented by subclass."""
        raise NotImplementedError

    @abstractmethod
    def train(self, **kwargs) -> None:
        """Training loop. Must be implemented by subclass."""
        raise NotImplementedError
