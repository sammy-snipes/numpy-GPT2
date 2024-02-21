import numpy as np
from einops import repeat
from einops import rearrange as erearrange
import string
from .engine import Parameter
from scipy.stats import norm


def exp(x: Parameter) -> Parameter:
    out = Parameter(np.exp(x.data), _children=(x,))

    def _backward():
        x.grad += out.data * out.grad

    out._backward = _backward
    return out


def softmax(x: Parameter, dim=-1) -> Parameter:
    e = exp(x)
    out = e * (e.sum(dim, keepdim=True) ** -1)
    return out


def einsum(ptrn: str, *args: Parameter) -> Parameter:
    out = Parameter(np.einsum(ptrn, *[_.data for _ in args]), _children=tuple(args))

    def _backward():
        in_ptrn, out_ptrn = ptrn.split("->")
        in_ptrns = in_ptrn.split(",")
        if not out_ptrn:
            out_ptrn = "".join(list(set(string.ascii_lowercase) - set(in_ptrn))[0])
            temp_out_grad = np.expand_dims(out.grad, 0)
        else:
            temp_out_grad = out.grad

        def calc_grad(idx):
            op_ptrn, op = in_ptrns[idx], args[idx]
            other_op_ptrns = in_ptrns[:idx] + in_ptrns[idx + 1 :]
            known_dims = "".join(
                [c for c in op_ptrn if c in "".join(other_op_ptrns) + out_ptrn]
            )
            grad_string = f"{out_ptrn},{','.join(other_op_ptrns)}->{known_dims}"
            if not other_op_ptrns:
                grad_string = grad_string.replace(",", "")
            grad = np.einsum(
                grad_string, temp_out_grad, *[_.data for _ in args if _ != op]
            )
            if known_dims != op_ptrn:
                expand_dims = tuple(
                    _ for _, __ in enumerate(op_ptrn) if __ not in known_dims
                )
                grad = np.expand_dims(grad, expand_dims)
            return grad

        for idx, arg in enumerate(args):
            arg.grad += calc_grad(idx)

    out._backward = _backward
    return out


def log(x: Parameter) -> Parameter:
    out = Parameter(np.log(x.data), _children=(x,))

    def _backward():
        x.grad += out.grad * (1 / x.data)

    out._backward = _backward
    return out


def cross_entropy_loss(x: Parameter, y: Parameter, dim=-1) -> Parameter:
    if any([_.data.dtype != np.float64 for _ in (x, y)]):
        raise TypeError("cross entropy takes float64")
    soft = softmax(x, dim=dim)
    log_soft = log(soft)
    ptrn = string.ascii_lowercase[: len(x.data.shape)]
    return (float(-x.data.shape[0]) ** -1) * einsum(f"{ptrn},{ptrn}->", log_soft, y)


def rearrange(x: Parameter, ptrn, **kwargs) -> Parameter:
    out = Parameter(erearrange(x.data, ptrn, **kwargs), _children=(x,))

    def _backward():
        in_ptrn, out_ptrn = ptrn.split("->")
        rvrs_ptrn = f"{out_ptrn}->{in_ptrn}"
        x.grad += erearrange(out.grad, rvrs_ptrn, **kwargs)

    out._backward = _backward
    return out


def layer_norm(x: Parameter, normalized_shape, eps: float = 1e-5):
    dims = tuple([x.shape.index(_) for _ in normalized_shape])
    mean = x.mean(dims, keepdim=True)
    variance = (x**2).mean(dims, keepdim=True) - mean**2
    return (variance + eps) ** -0.5 * (x - mean)


def relu(x: Parameter):
    out = Parameter(np.where(x.data > 0, x.data, 0), _children=(x,))

    def _backward():
        x.grad += np.where(x.data > 0, 1, 0) * out.grad

    out._backward = _backward
    return out


def gelu(x: Parameter) -> Parameter:
    out = Parameter(x.data * norm.cdf(x.data), _children=(x,))

    def _backward():
        d_gelu = x.data * norm.pdf(x.data) + norm.cdf(x.data)

        x.grad += d_gelu * out.grad

    out._backward = _backward
    return out


def dropout(x: Parameter, p: float) -> Parameter:
    mask = np.random.binomial(1, 1 - p, x.shape) * (1 / (1 - p))
    out = Parameter(x.data * mask, _children=(x,))

    def _backward():
        x.grad += mask * out.grad

    out._backward = _backward
    return out


def embed(idx: np.ndarray, w: Parameter) -> Parameter:
    if not idx.dtype == np.int64:
        raise TypeError("idx gotta be ints")
    out = Parameter(w.data[idx], _children=(w,))

    def _backward():
        local_grad = np.zeros_like(w.data)
        np.add.at(local_grad, idx, out.grad)
        w.grad += local_grad

    out._backward = _backward
    return out
