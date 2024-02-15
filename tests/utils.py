import torch
import numpy as np
from numpyGPT.engine import Parameter
from itertools import combinations

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)


def make_values(shapes, high=1.0, low=0.0, seed=42):
    torch.manual_seed(42)
    t_values = [(torch.randn(s) * (high - low) + low).requires_grad_() for s in shapes]
    p_values = [Parameter(t.detach().numpy()) for t in t_values]
    return *t_values, *p_values


def is_close(*args, atol=1e-8):
    tensors = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in args]
    return torch.allclose(tensors[0], tensors[1], atol=atol)
    combs = list(combinations(tensors, 2))
    close = [torch.allclose(t1, t2, atol=atol) for (t1, t2) in combs]
    # print(*args, sep=2 * "\n")
    # print(50 * "=", "\n")
    return (*close,)


def zero_grad(*args):
    for a in args:
        if isinstance(a, Parameter):
            a.grad = np.zeros_like(a.grad)
        if isinstance(a, torch.Tensor):
            a.grad.zero_()


def param_grad_is_close(l1, l2):
    close = [is_close(t1.grad, t2.grad) for (t1, t2) in zip(l1, l2)]
    if not all(close):
        # if all(close):
        for _, __ in enumerate(close):
            print(f"param {_} grad close : {__}")
    return all(close)
