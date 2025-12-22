"""
Refactored IGNOInverter that inherits from BaseTrainer.
Handles the second stage of IGNO training (latent to Gaussian mapping).
Uses the shared run directory from the Solver.
"""
import torch
import time
from tqdm import trange

from src.solver.base import BaseTrainer
from src.solver.dgno import Solver
from src.solver.config import TrainingConfig
from src.utils.solver_utils import get_optimizer, get_scheduler, var_data_loader
from src.components.nf import RealNVP


class IGNOInverter(BaseTrainer):
    """
    Learns bijective mapping from latent space to Gaussian noise using normalizing flows.
    This is the second stage of IGNO training.

    Automatically uses the same run directory as the Solver for unified experiment tracking.
    """

    def __init__(
            self,
            solver: Solver,
    ):
        """
        Args:
            solver: Trained solver instance (must have run_dir set)
        """
        # Get shared run directory from solver
        if solver.run_dir is None:
            raise RuntimeError(
                "Solver must be set up (have a run_dir) before creating IGNOInverter. "
                "Call solver.setup() or solver.train() first."
            )

        super().__init__(
            device=solver.device,
            dtype=solver.dtype,
            run_dir=solver.run_dir  # Share the same run directory!
        )
        self.solver = solver
        self.nf = None  # Normalizing flow model

    def setup(
            self,
            config: TrainingConfig,
    ) -> None:
        """
        Setup normalizing flow, optimizer, scheduler, and logging.
        Uses the solver's run directory automatically.

        Args:
            config: Training configuration
        """

        # Verify solver is set up
        if self.solver.run_dir is None:
            raise RuntimeError(
                "Solver must have a run_dir. Call solver.setup() or solver.train() first."
            )

        # Initialize normalizing flow
        self.nf = RealNVP(
            config=config.second_stage.nf_config
        ).to(self.device)

        # Store as model_dict for compatibility with BaseTrainer
        self.model_dict = {'nf': self.nf}

        # Setup optimizer
        self.optimizer = get_optimizer(
            optimizer_config=config.second_stage.optimizer,
            param_list=list(self.nf.parameters())
        )

        # Setup scheduler
        self.scheduler = get_scheduler(
            scheduler_config=config.second_stage.scheduler,
            optimizer=self.optimizer
        )

        # Setup stage-specific directories under shared run_dir
        self._setup_stage_directories(stage_name='igno')

        # Setup TensorBoard
        self._setup_tensorboard()

        # Update config in shared run directory (append second stage info if needed)
        # The config should already exist from solver.setup()
        if not (self.run_dir / "config.yaml").exists():
            config.save(self.run_dir / "config.yaml")

    def _freeze_solver_models(self) -> None:
        """Freeze all pretrained solver models."""
        if self.solver.model_dict is None:
            raise RuntimeError(
                "Solver models not initialized. Run solver.setup() or solver.train() first."
            )

        for model in self.solver.model_dict.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def _extract_latents(
            self,
            a: torch.Tensor,
            batch_size: int = 100
    ) -> torch.Tensor:
        """
        Extract latent representations using frozen encoder.

        Args:
            a: Input functions
            batch_size: Batch size for extraction (to avoid OOM)

        Returns:
            Latent representations
        """
        if "enc" not in self.solver.model_dict:
            raise KeyError("Encoder 'enc' not found in solver.model_dict")

        encoder = self.solver.model_dict["enc"]
        encoder.eval()

        latents_list = []
        num_samples = a.shape[0]

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = a[i:i + batch_size].to(self.device)
                latents_batch = encoder(batch)
                latents_list.append(latents_batch.cpu())

        return torch.cat(latents_list, dim=0)

    def train(
            self,
            a_train: torch.Tensor,
            a_test: torch.Tensor,
            config: TrainingConfig,
            batch_size: int = 100,
            epochs: int = 1000,
            epoch_show: int = 100,
            shuffle: bool = True,
    ):
        """
        Train normalizing flow to map latent space to Gaussian noise.

        Args:
            a_train: Training input functions
            a_test: Test input functions
            config: Training configuration
            batch_size: Batch size for training
            epochs: Number of training epochs
            epoch_show: Print frequency
            shuffle: Whether to shuffle training data
        """
        # Setup if not already done
        if self.nf is None:
            self.setup(config=config)

        # Freeze solver and extract latents
        print("Freezing solver models...")
        self._freeze_solver_models()

        print("Extracting training latents...")
        latents_train = self._extract_latents(a_train, batch_size=batch_size)

        print("Extracting test latents...")


        train_loader = var_data_loader(
            latents_train,
            batch_size=batch_size,
            shuffle=shuffle
        )

        test_loader = var_data_loader(
            latents_test,
            batch_size=batch_size,
            shuffle=False
        )

        t_start = time.time()
        best_loss_test = float('inf')

        # Training loop
        print(f"Starting normalizing flow training for {epochs} epochs...")
        for epoch in trange(epochs):
            self._activate_train()
            loss_train_sum = 0.0

            for (z_batch,) in train_loader:
                z_batch = z_batch.to(self.device)

                loss = self.nf.loss(z_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_train_sum += loss.item()

            # Validation
            self._activate_eval()
            loss_test_sum = 0.0

            with torch.no_grad():
                for (z_batch,) in test_loader:
                    z_batch = z_batch.to(self.device)
                    loss = self.nf.loss(z_batch)
                    loss_test_sum += loss.item()

            # Compute averages
            avg_loss_train = loss_train_sum / len(train_loader)
            avg_loss_test = loss_test_sum / len(test_loader)

            # Log to TensorBoard
            self.writer.add_scalar("train/nll_loss", avg_loss_train, epoch)
            self.writer.add_scalar("test/nll_loss", avg_loss_test, epoch)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], epoch)

            # Save best model
            if avg_loss_test < best_loss_test:
                best_loss_test = avg_loss_test
                self.save_models(
                    filename='best_nf.pt',
                    additional_state={
                        'epoch': epoch,
                        'nll_loss': best_loss_test
                    }
                )

            # Scheduler step
            if self.scheduler is not None:
                if config.second_stage.scheduler.type == 'Plateau':
                    self.scheduler.step(avg_loss_test)
                elif config.second_stage.scheduler.type is not None:
                    self.scheduler.step()

            # Print progress
            if (epoch + 1) % epoch_show == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch + 1}, Time: {time.time() - t_start:.4f}s')
                print(f'  Train NLL: {avg_loss_train:.4f}, '
                      f'Test NLL: {avg_loss_test:.4f}, LR: {lr:.6f}')

        # Save final model
        self.save_models(
            filename='last_nf.pt',
            additional_state={'epoch': epochs - 1}
        )

        print(f'Total training time: {time.time() - t_start:.4f}s')
        print(f'Best test NLL: {best_loss_test:.4f}')

    def sample_latents(
            self,
            num_samples: int,
            temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample from Gaussian noise and map back to latent space.

        Args:
            num_samples: Number of samples to generate
            temperature: Temperature for sampling (default 1.0 = standard Gaussian)

        Returns:
            Latent vectors that can be decoded by the solver's decoder
        """
        if self.nf is None:
            raise RuntimeError("Normalizing flow not initialized. Call setup() first.")

        self.nf.eval()
        with torch.no_grad():
            # Sample from Gaussian with temperature
            latent_dim = self.nf.dim
            noise = torch.randn(num_samples, latent_dim, device=self.device) * temperature

            # Inverse transform: noise -> latent space
            latents = self.nf.inverse(noise)

        return latents

    # def generate_samples(
    #         self,
    #         num_samples: int,
    #         temperature: float = 1.0
    # ) -> torch.Tensor:
    #     """
    #     Generate new input functions by sampling latents and decoding.
    #
    #     Args:
    #         num_samples: Number of samples to generate
    #         temperature: Temperature for sampling
    #
    #     Returns:
    #         Generated input functions
    #     """
    #     if "dec" not in self.solver.model_dict:
    #         raise KeyError("Decoder 'dec' not found in solver.model_dict")
    #
    #     # Sample latents
    #     latents = self.sample_latents(num_samples, temperature)
    #
    #     # Decode to input space
    #     decoder = self.solver.model_dict["dec"]
    #     decoder.eval()
    #
    #     with torch.no_grad():
    #         samples = decoder(latents.to(self.device))
    #
    #     return samples
