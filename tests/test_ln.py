import torch
from numpyGPT.functions import layer_norm
from tests.utils import make_values, is_close


def torch_layer_norm(t, normalized_shape, eps=1e-5):
    return torch.nn.LayerNorm(
        normalized_shape, eps, elementwise_affine=False, bias=False
    )(t)


def test_ln():
    shape, normalized_shape = (64, 3, 15, 14), (3, 15, 14)
    t, x = make_values([shape])
    eps = 1e-5

    ln_x = layer_norm(x, normalized_shape, eps=eps)
    ln_t = torch_layer_norm(t, normalized_shape, eps=eps)

    f = lambda x: (x**2 - x**-2).sum().backward()
    f(ln_x)
    f(ln_t)

    assert is_close(ln_x.data, ln_t)
    assert is_close(x.grad, t.grad)
