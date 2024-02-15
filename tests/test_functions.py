from numpyGPT.engine import Parameter
import torch
import numpy as np
from numpyGPT.functions import softmax, einsum, log, cross_entropy_loss
from tests.utils import make_values, is_close


def test_einsum():
    shapes = [(2, 3), (3, 4)]
    t1, t2, x1, x2 = make_values(shapes)
    out = einsum("ij,jk->", x1, x2)

    torch_out = torch.einsum("ij,jk->", t1, t2)

    out.backward()
    torch_out.backward()
    assert is_close(out.data, torch_out)
    assert is_close(x1.grad, t1.grad)
    assert is_close(x2.grad, t2.grad)


def test_softmax():
    shapes = [(5, 2, 3), (3, 4)]
    t, k, x, y = make_values(shapes)
    atol = 1e-6
    for dim in range(-t.dim(), t.dim(), 1):
        x_pred = einsum("bij,jk->bik", x, y)
        t_pred = torch.einsum("bij,jk->bik", t, k)

        x_soft = softmax(x_pred, dim=dim)
        t_soft = torch.softmax(t_pred, dim=dim)

        torch.log(t_soft).sum().backward()
        log(x_soft).sum().backward()

        assert is_close(x_soft.data, t_soft)
        assert is_close(x.grad, t.grad, atol=atol)
        assert is_close(y.grad, k.grad, atol=atol)
        t.grad.zero_()
        x.grad = np.zeros_like(x.grad)


def test_cross_entropy():
    batch_size = 64
    n_class = 4
    x = torch.randn(batch_size, n_class, requires_grad=True)
    t, x = make_values([(batch_size, n_class)])

    k = torch.randint(0, n_class - 1, size=(batch_size,))
    k = torch.nn.functional.one_hot(k, num_classes=n_class).double()

    y = Parameter(k.detach().numpy())

    t_loss = torch.nn.CrossEntropyLoss()(t, k)
    x_loss = cross_entropy_loss(x, y)
    t_loss.backward()
    x_loss.backward()
    assert is_close(x_loss.data, t_loss)
    assert is_close(x.grad, t.grad)
