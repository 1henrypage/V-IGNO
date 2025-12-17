
import torch
import torch.nn as nn
from typing import Tuple

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

        # TODO Init
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






