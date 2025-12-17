
import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt

from src.utils.misc_utils import get_default_device


class ScaleTranslateNet(nn.Module):
    """
    Neural network for computing scale and translation params.
    Used in RealNVP coupling layers.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2
    ):
        super(ScaleTranslateNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim,hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)
        self.scale_layer = nn.Linear(hidden_dim, input_dim)
        self.translate_layer = nn.Linear(hidden_dim, input_dim)

        nn.init.zeros_(self.scale_layer.weight)
        nn.init.zeros_(self.scale_layer.bias)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        scale = torch.clamp(self.scale_layer(h), min=-5.0, max=5.0)
        translation = self.translate_layer(h)
        return scale, translation

class CouplingLayer(nn.Module):
    """
    RealNVP coupling layer that splits input and applies affine transformation.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, num_layers: int = 2, mask_type: str = 'checkerboard'):
        super(CouplingLayer, self).__init__()
        self.dim = dim

        if mask_type == 'checkerboard':
            self.register_buffer('mask', self._create_checkerboard_mask(dim))
        elif mask_type == 'channel':
            self.register_buffer('mask', self._create_channel_mask(dim))
        else:
            raise ValueError(f'Unknown mask type {mask_type}')

        self.split_dim = int(self.mask.sum().item())

        self.st_net = ScaleTranslateNet(self.split_dim, hidden_dim, num_layers)

    def _create_checkerboard_mask(self, dim: int) -> torch.Tensor:
        mask = torch.zeros(dim)
        mask[::2] = 1
        return mask

    def _create_channel_mask(self, dim: int) -> torch.Tensor:
        mask = torch.zeros(dim)
        mask[:dim // 2] = 1
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = x * self.mask
        x2 = x * (1 - self.mask)
        x1_active = x1[:, self.mask.bool()]

        scale, translation = self.st_net(x1_active)
        x2_active = x2[:, (~self.mask.bool())]
        y2_active = x2_active * torch.exp(scale) + translation

        y = x1.clone()
        y[:, (~self.mask.bool())] = y2_active
        log_det = scale.sum(dim = 1)
        return y, log_det

    def inverse(self, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1 = y * self.mask
        y2 = y * (1 - self.mask)

        y1_active = y1[:, self.mask.bool()]

        scale, translation = self.st_net(y1_active)

        y2_active = y2[:, (~self.mask.bool())]
        x2_active = (y2_active - translation) * torch.exp(-scale)

        x = y1.clone()
        x[:, (~self.mask.bool())] = x2_active

        log_det = -scale.sum(dim = 1)

        return x, log_det

class RealNVP(nn.Module):
    """
    RealNVP normalizing flow model.
    Maps between latent beta distributiona nd a standard normal distribution.

    """

    def __init__(
            self,
            latent_dim: int,
            num_flows: int = 3,
            hidden_dim: int = 64,
            num_layers: int = 2
    ):
        super(RealNVP, self).__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows

        self.flows = nn.ModuleList()
        for i in range(num_flows):
            mask_type = 'checkerboard' if i % 2 == 0 else 'channel'
            self.flows.append(
                CouplingLayer(latent_dim, hidden_dim, num_layers, mask_type)
            )

        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))


    def forward(self, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # data to latent
        z = beta
        log_det_total = 0
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # latent to data
        beta = z
        log_det_total = 0

        for flow in reversed(self.flows):
            beta, log_det = flow.inverse(beta)
            log_det_total += log_det

        return beta, log_det_total

    def log_prob(self, beta: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(beta)
        log_pz = -0.5 * (z ** 2 + self.log_2pi).sum(dim=1)
        log_prob = log_pz + log_det

        return log_prob

    def sample(self, num_samples: int, device: torch.device = get_default_device()):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        beta, _ = self.inverse(z)
        return beta

    def loss(self, beta: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(beta).mean()


# ====================
# TOY PROBLEM TEST
# ====================

def create_2d_mixture_data(n_samples=1000):
    """Create a 2D mixture of Gaussians as toy data"""
    n_per_component = n_samples // 3

    # Component 1: centered at (-2, -2)
    data1 = torch.randn(n_per_component, 2) * 0.5 + torch.tensor([-2.0, -2.0])

    # Component 2: centered at (2, 2)
    data2 = torch.randn(n_per_component, 2) * 0.5 + torch.tensor([2.0, 2.0])

    # Component 3: centered at (2, -2)
    data3 = torch.randn(n_samples - 2 * n_per_component, 2) * 0.5 + torch.tensor([2.0, -2.0])

    data = torch.cat([data1, data2, data3], dim=0)
    data = data[torch.randperm(data.size(0))]

    return data


def plot_results(model, data, epoch, device):
    """Visualize the learned distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original data
    axes[0, 0].scatter(data[:, 0].cpu(), data[:, 1].cpu(), alpha=0.5, s=1)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)

    # Samples from model
    with torch.no_grad():
        samples = model.sample(1000, device=device)
    axes[0, 1].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5, s=1, color='red')
    axes[0, 1].set_title(f'Generated Samples (Epoch {epoch})')
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)

    # Latent space (data transformed to z)
    with torch.no_grad():
        z, _ = model.forward(data[:1000].to(device))
    axes[1, 0].scatter(z[:, 0].cpu(), z[:, 1].cpu(), alpha=0.5, s=1, color='green')
    axes[1, 0].set_title('Data in Latent Space (should be Gaussian)')
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)

    # Log probability heatmap
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    with torch.no_grad():
        log_probs = model.log_prob(points).cpu().reshape(100, 100)

    im = axes[1, 1].imshow(log_probs.T, extent=[-4, 4, -4, 4], origin='lower', cmap='viridis')
    axes[1, 1].set_title('Log Probability Heatmap')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Set device
    device = get_default_device()
    print(f"Using device: {device}")

    # Create toy data
    print("Creating toy dataset...")
    data = create_2d_mixture_data(n_samples=5000)
    train_data = data.to(device)

    # Initialize model
    print("Initializing RealNVP model...")
    model = RealNVP(
        latent_dim=2,
        num_flows=6,
        hidden_dim=128,
        num_layers=3
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # Training loop
    print("Starting training...")
    batch_size = 256
    num_epochs = 2000

    losses = []

    for epoch in range(num_epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(train_data.size(0))
        epoch_loss = 0
        num_batches = 0

        for i in range(0, train_data.size(0), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = train_data[batch_indices]

            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Visualize progress
        if (epoch + 1) % 500 == 0 or epoch == 0:
            model.eval()
            fig = plot_results(model, data, epoch + 1, device)
            plt.savefig(f'realnvp_epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
            plt.close()

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Final visualization
    model.eval()
    fig = plot_results(model, data, num_epochs, device)
    plt.savefig('realnvp_final.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")

    # Test forward and inverse consistency
    print("\nTesting forward-inverse consistency...")
    with torch.no_grad():
        test_samples = data[:100].to(device)
        z, log_det_fwd = model.forward(test_samples)
        reconstructed, log_det_inv = model.inverse(z)

        reconstruction_error = (test_samples - reconstructed).abs().mean()
        log_det_error = (log_det_fwd + log_det_inv).abs().mean()

        print(f"Reconstruction error: {reconstruction_error:.6f}")
        print(f"Log-det consistency: {log_det_error:.6f}")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print("=" * 50)






