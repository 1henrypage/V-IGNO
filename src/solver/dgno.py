"""
Complete Solver implementation with ProblemInstance.
Inherits from BaseTrainer and manages the shared run directory.
"""
import torch
import torch.nn as nn
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict
from tqdm import trange
from datetime import datetime
from pathlib import Path

from src.solver.base import BaseTrainer
from src.solver.config import TrainingConfig
from src.utils.Losses import MyError, MyLoss
from src.utils.misc_utils import get_default_device, get_project_root
from src.utils.solver_utils import get_optimizer, get_scheduler, data_loader


# Loss classes
#############################
class ProblemInstance(ABC):
    """Abstract base class for PDE losses"""

    def __init__(
            self,
            device: torch.device | str = get_default_device(),
            dtype: Optional[torch.dtype] = torch.float32,
            artifact_root: Optional[Path] = None
    ) -> None:
        self.device = device
        self.dtype = dtype

        # Set artifact root (can be overridden)
        if artifact_root is None:
            self.artifact_root = get_project_root() / "runs"
        else:
            self.artifact_root = Path(artifact_root)

        # Run directory will be set by Solver
        self.run_dir: Optional[Path] = None

        # Forward declare
        self.get_loss = None
        self.get_error = None

        # INIT
        self.init_error()
        self.init_loss()

    @abstractmethod
    def loss_pde(
            self,
            a: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_data(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
            u: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_model_dict(self) -> Dict[str, nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def error(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
            u: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def init_error(
            self,
            err_type: str='lp_rel',
            d: int = 2,
            p: int = 2,
            size_average: bool = True,
            reduction: bool = True,
    ) -> None:
        self.get_error = MyError(
            d=d,
            p=p,
            size_average=size_average,
            reduction=reduction,
        )(err_type)

    def init_loss(
            self,
            loss_type: str = 'mse_org',
            size_average=True,
            reduction=True
    ):
        self.get_loss = MyLoss(
            size_average=size_average,
            reduction=reduction
        )(loss_type)

    def pre_train_check(self) -> None:
        """
        This is useful if we don't do any custom losses.
        The solver will initialise the default settings.

        Logic:
        - If both get_loss and get_error are None, call init_loss() and init_error()
        - If exactly one of them is None, raise an error
        - Otherwise, do nothing
        """
        loss_none = self.get_loss is None
        error_none = self.get_error is None

        if loss_none and error_none:
            # Initialize both
            self.init_loss()
            self.init_error()
        elif loss_none != error_none:  # XOR: exactly one is None
            raise ValueError("Both get_loss and get_error must be set, or both None to auto-initialize.")
        # else: both are already set, do nothing


class Solver(BaseTrainer):
    """
    Solver for PDE problems. Handles the first stage of training.
    Manages the shared run directory for the entire experiment.
    """

    def __init__(
            self,
            problem_instance: ProblemInstance,
    ):
        # Initialize with run_dir that will be set during setup
        super().__init__(
            device=problem_instance.device,
            dtype=problem_instance.dtype,
            run_dir=None  # Will be set in setup
        )
        self.problem_instance = problem_instance

    def _create_run_directory(
            self,
            custom_run_tag: Optional[str] = None
    ) -> Path:
        """
        Create the main run directory for this experiment.
        This directory will be shared across all training stages.

        Args:
            custom_run_tag: Optional custom tag for the run name

        Returns:
            Path to the created run directory
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if custom_run_tag:
            run_name = f"{timestamp}_{custom_run_tag}"
        else:
            run_name = timestamp

        run_dir = self.problem_instance.artifact_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def setup(
            self,
            config: TrainingConfig,
            custom_run_tag: Optional[str] = None
    ) -> None:
        """
        Setup models, optimizer, scheduler, and logging.
        Creates the shared run directory for the entire experiment.

        Args:
            config: Training configuration
            custom_run_tag: Optional tag for run directory name
        """
        # Pre-training checks
        self.problem_instance.pre_train_check()

        # Create shared run directory
        self.run_dir = self._create_run_directory(custom_run_tag)
        self.problem_instance.run_dir = self.run_dir  # Share with problem instance

        # Setup stage-specific directories
        self._setup_stage_directories(stage_name='dgno')

        # Send to device & collect parameters
        param_list = []
        for model in self.model_dict.values():
            model.to(self.device)
            param_list += list(model.parameters())

        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(
            optimizer_config=config.first_stage.optimizer,
            param_list=param_list,
        )

        self.scheduler = get_scheduler(
            scheduler_config=config.first_stage.scheduler,
            optimizer=self.optimizer,
        )

        # Setup TensorBoard
        self._setup_tensorboard()

        # Save config to shared run directory (not stage directory)
        config.save(self.run_dir / "config.yaml")

    def _log_epoch(
            self,
            loss_train: torch.Tensor,
            loss_data: torch.Tensor,
            loss_pde: torch.Tensor,
            loss_test: torch.Tensor,
            error_test: torch.Tensor,
            epoch: int
    ) -> None:
        """Log metrics to TensorBoard."""
        self.writer.add_scalar("train/loss_train", loss_train.item(), epoch)
        self.writer.add_scalar("train/loss_data", loss_data.item(), epoch)
        self.writer.add_scalar("train/loss_pde", loss_pde.item(), epoch)
        self.writer.add_scalar("test/loss", loss_test.item(), epoch)

        if error_test.numel() > 1:
            for i, err in enumerate(error_test):
                self.writer.add_scalar(f"test/error_{i}", err.item(), epoch)
        else:
            self.writer.add_scalar("test/error", error_test.item(), epoch)

    def _scheduler_step(
            self,
            error_test: torch.Tensor,
            config: TrainingConfig
    ) -> None:
        """Update learning rate scheduler."""
        if self.scheduler is None:
            return

        scheduler_type = config.first_stage.scheduler.type
        if scheduler_type == 'Plateau':
            self.scheduler.step(error_test.item())
        elif scheduler_type is not None:
            self.scheduler.step()

    def train(
            self,
            a_train, u_train, x_train,
            a_test, u_test, x_test,
            config: TrainingConfig,
            batch_size: int = 100,
            epochs: int = 1,
            epoch_show: int = 10,
            shuffle: bool = True,
            custom_run_tag: str = '',
            **kwargs
    ):
        """
        Train the solver using mini-batch gradient descent.

        Args:
            a_train, u_train, x_train: Training data
            a_test, u_test, x_test: Test data
            config: Training configuration
            batch_size: Batch size for training
            epochs: Number of training epochs
            epoch_show: Print frequency
            shuffle: Whether to shuffle training data
            custom_run_tag: Optional tag for logging
        """
        # Setup if not already done
        if self.model_dict is None:
            self.setup(config=config, custom_run_tag=custom_run_tag)

        # Create data loaders
        train_loader = data_loader(
            a=a_train, u=u_train, x=x_train,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        test_loader = data_loader(
            a=a_test, u=u_test, x=x_test,
            batch_size=batch_size
        )

        t_start = time.time()
        best_err_test = float('inf')
        training_weights = config.first_stage.loss_weights

        # Training loop
        for epoch in trange(epochs):
            self._activate_train()
            loss_train_sum, loss_data_sum, loss_pde_sum = 0., 0., 0.

            # Training
            for a, u, x in train_loader:
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)

                loss_pde = self.problem_instance.loss_pde(a)
                loss_data = self.problem_instance.loss_data(x, a, u)
                loss_train = (loss_pde * training_weights.pde +
                              loss_data * training_weights.data)

                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

                loss_train_sum += loss_train
                loss_data_sum += loss_data
                loss_pde_sum += loss_pde

            # Validation
            self._activate_eval()
            loss_test_sum = 0
            error_test_sum = 0

            for a, u, x in test_loader:
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                with torch.no_grad():
                    loss_test = self.problem_instance.loss_data(x, a, u)
                    error_test = self.problem_instance.error(x, a, u)
                loss_test_sum += loss_test
                error_test_sum += error_test

            # Compute averages
            avg_loss_train = loss_train_sum / len(train_loader)
            avg_loss_data = loss_data_sum / len(train_loader)
            avg_loss_pde = loss_pde_sum / len(train_loader)
            avg_loss_test = loss_test_sum / len(test_loader)
            avg_error_test = error_test_sum / len(test_loader)

            # Log metrics
            self._log_epoch(
                loss_train=avg_loss_train,
                loss_data=avg_loss_data,
                loss_pde=avg_loss_pde,
                loss_test=avg_loss_test,
                error_test=avg_error_test,
                epoch=epoch,
            )

            # Save best model
            error_test_scalar = torch.mean(avg_error_test).item()
            if error_test_scalar < best_err_test:
                best_err_test = error_test_scalar
                self.save_models(
                    filename='best.pt',
                    additional_state={'epoch': epoch, 'error': best_err_test}
                )

            # Update learning rate
            self._scheduler_step(
                error_test=torch.mean(avg_error_test),
                config=config
            )

            # Print progress
            if (epoch + 1) % epoch_show == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch + 1}, Time: {time.time() - t_start:.4f}s')
                print(f'  Loss: {avg_loss_train.item():.4f}, '
                      f'PDE: {avg_loss_pde.item():.4f}, '
                      f'Data: {avg_loss_data.item():.4f}')
                print(f'  Test Error: {error_test_scalar:.4f}, LR: {lr:.6f}')

        # Save final model
        self.save_models(
            filename='last.pt',
            additional_state={'epoch': epochs - 1}
        )

        print(f'Total training time: {time.time() - t_start:.4f}s')
        print(f'Best test error: {best_err_test:.4f}')