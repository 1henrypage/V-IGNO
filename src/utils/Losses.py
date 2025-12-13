import torch

class MyError(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(MyError, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.size_average = size_average
        self.reduction = reduction
        self.eps = 1e-8

    def _compute_norm(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Compute batch-wise Lp norm of difference WITH mesh scaling.
        Used ONLY for LP_abs; NOT for relative error.
        """
        assert y_pred.shape == y_true.shape
        batch_size = y_true.shape[0]
        diff = y_pred.reshape(batch_size, -1) - y_true.reshape(batch_size, -1)
        h = 1.0 / (y_true.shape[1] - 1.0)  # uniform mesh
        total_norm = (h ** (self.d / self.p)) * torch.norm(diff, self.p, dim=1)
        return total_norm, batch_size

    def _reduce(self, values: torch.Tensor):
        if not self.reduction:
            return values
        return torch.mean(values) if self.size_average else torch.sum(values)

    def LP_abs(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        total_norm, _ = self._compute_norm(y_pred, y_true)
        return self._reduce(total_norm)

    def Lp_rel(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Relative Lp error — NO mesh scaling."""
        assert y_pred.shape == y_true.shape
        batch_size = y_true.shape[0]

        diff_norm = torch.norm(
            y_true.reshape(batch_size, -1) - y_pred.reshape(batch_size, -1),
            self.p, dim=1
        )
        y_norm = torch.norm(
            y_true.reshape(batch_size, -1),
            self.p, dim=1
        ) + self.eps

        return self._reduce(diff_norm / y_norm)

    def __call__(self, err_type: str):
        if err_type == 'lp_abs':
            return self.LP_abs
        elif err_type == 'lp_rel':
            return self.Lp_rel
        else:
            raise NotImplementedError(f"Error type '{err_type}' is not defined.")


class MyLoss(object):
    def __init__(self, size_average=True, reduction=True):
        super(MyLoss, self).__init__()
        self.size_average = size_average
        self.reduction = reduction
        self.eps = 1e-6

    def _compute_diff(self, y_pred: torch.Tensor, y_true: torch.Tensor, p=2):
        """Compute batch-wise Lp norm of difference"""
        assert y_pred.shape == y_true.shape
        batch_size = y_true.shape[0]
        diff = y_pred.reshape(batch_size, -1) - y_true.reshape(batch_size, -1)
        diff_norm = torch.norm(diff, p, dim=1)
        return diff_norm, batch_size

    def _reduce(self, values: torch.Tensor):
        """Apply reduction according to settings"""
        if not self.reduction:
            return values
        return torch.mean(values) if self.size_average else torch.sum(values)

    def mse_org(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MSE loss without relative scaling"""
        diff_norm, _ = self._compute_diff(y_pred, y_true, p=2)
        return self._reduce(diff_norm)

    def mse_rel(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MSE loss with relative scaling"""
        diff_norm, batch_size = self._compute_diff(y_pred, y_true, p=2)
        y_norm = torch.norm(y_true.reshape(batch_size, -1), 2, dim=1) + self.eps
        return self._reduce(diff_norm / y_norm)

    def __call__(self, loss_type: str):
        """Return the selected loss function"""
        if loss_type == 'mse_org':
            return self.mse_org
        elif loss_type == 'mse_rel':
            return self.mse_rel
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' is not defined.")
