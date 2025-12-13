import torch
import pytest

# -----------------------------------------------------------
# PLACEHOLDER IMPORTS — REPLACE WITH YOUR ACTUAL MODULE NAMES
# -----------------------------------------------------------


from DGenNO.Utils.Losses import MyError as OldError
from DGenNO.Utils.Losses import MyLoss as OldLoss

from src.utils.Losses import MyError as NewError
from src.utils.Losses import MyLoss as NewLoss


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def random_batch(batch=4, mesh=50):
    y_true = torch.rand(batch, mesh, 1)
    y_pred = torch.rand(batch, mesh, 1)
    return y_pred, y_true


def equal(a, b, tol=1e-6):
    return torch.allclose(a, b, atol=tol, rtol=tol)


# -----------------------------------------------------------
# MyError Tests
# -----------------------------------------------------------

@pytest.mark.parametrize("err_type", ["lp_abs", "lp_rel"])
@pytest.mark.parametrize("d,p", [(1,1), (2,2), (3,2)])
@pytest.mark.parametrize("size_average", [True, False])
@pytest.mark.parametrize("reduction", [True, False])
def test_myerror_equivalence(err_type, d, p, size_average, reduction):
    y_pred, y_true = random_batch()

    old = OldError(d=d, p=p, size_average=size_average, reduction=reduction)
    new = NewError(d=d, p=p, size_average=size_average, reduction=reduction)

    old_fn = old.LP_abs if err_type == 'lp_abs' else old.Lp_rel
    new_fn = new(err_type)          # tests __call__

    out_old = old_fn(y_pred, y_true)
    out_new = new_fn(y_pred, y_true)

    assert equal(out_old, out_new), f"MyError mismatch for '{err_type}'"


# -----------------------------------------------------------
# MyLoss Tests
# -----------------------------------------------------------

@pytest.mark.parametrize("loss_type", ["mse_org", "mse_rel"])
@pytest.mark.parametrize("size_average", [True, False])
@pytest.mark.parametrize("reduction", [True, False])
def test_myloss_equivalence(loss_type, size_average, reduction):
    y_pred, y_true = random_batch()

    old = OldLoss(size_average=size_average, reduction=reduction)
    new = NewLoss(size_average=size_average, reduction=reduction)

    old_fn = old.mse_org if loss_type == "mse_org" else old.mse_rel
    new_fn = new(loss_type)         # tests __call__

    out_old = old_fn(y_pred, y_true)
    out_new = new_fn(y_pred, y_true)

    assert equal(out_old, out_new), f"MyLoss mismatch for '{loss_type}'"


# -----------------------------------------------------------
# Interface Tests
# -----------------------------------------------------------

def test_callable_dispatch():
    new = NewError()
    assert new('lp_abs').__name__ == 'LP_abs'
    assert new('lp_rel').__name__ == 'Lp_rel'

    loss = NewLoss()
    assert loss('mse_org').__name__ == 'mse_org'
    assert loss('mse_rel').__name__ == 'mse_rel'


def test_shapes_preserved():
    y_pred, y_true = random_batch()
    new = NewError()
    out = new('lp_rel')(y_pred, y_true)
    assert isinstance(out, torch.Tensor)


# -----------------------------------------------------------
# End of test suite
# -----------------------------------------------------------
