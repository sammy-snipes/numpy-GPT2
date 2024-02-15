from numpyGPT.engine import Parameter
import torch
import numpy as np
from tests.utils import is_close, make_values, zero_grad


def test_multiplication():
    shapes = [(2, 3), (2, 3)]
    t, k, x, y = make_values(shapes)
    z = x * y
    l = t * k

    l.sum().backward()
    z.sum().backward()

    assert is_close(z.data, l), "output inequality"
    assert is_close(x.grad, t.grad), "grad inequality"
    assert is_close(y.grad, k.grad), "grad inequality"


def test_multiplication_broadcasting():
    shapes = [(2, 3, 4), (3, 4), (4)]
    t, k, l, x, y, z = make_values(shapes)
    w = x * y * z
    m = t * k * l
    w = 5 * w
    m = 5 * m
    m.sum().backward()
    w.sum().backward()

    assert is_close(w.data, m)
    assert is_close(x.grad, t.grad)
    assert is_close(z.grad, l.grad)
    assert is_close(y.grad, k.grad)


def test_subtraction_broadcasting():
    shapes = [(2, 3, 4), (3, 4), (4)]
    t, k, l, x, y, z = make_values(shapes)
    w = x - y - z
    m = t - k - l
    w = 5 * w
    m = 5 * m
    m.sum().backward()
    w.sum().backward()

    assert is_close(w.data, m)
    assert is_close(x.grad, t.grad)
    assert is_close(z.grad, l.grad)
    assert is_close(y.grad, k.grad)


def test_pow():
    shapes = (1, 2, 3, 4, 5)
    t, x = make_values([shapes], high=20, low=19)

    z = (x**2) - (3 * x) - (x**-1.5)
    k = (t**2) - (3 * t) - (t**-1.5)
    z.sum().backward()
    k.sum().backward()
    assert is_close(z.data, k)
    assert is_close(x.grad, t.grad)


def test_mean():
    shapes = (2, 3, 4)
    t, x = make_values([shapes], high=10, low=1)
    for dims in (-1,), (-2,), (-3,), (-1, -2), (-1, -3):

        x_m = x.mean(dims, keepdim=True)
        t_m = t.mean(dims, keepdim=True)

        z = (x_m) ** 2 - (x_m) ** 3
        k = (t_m) ** 2 - (t_m) ** 3

        k.sum().backward()
        z.sum().backward()

        assert is_close(x_m.data, t_m.data)
        assert is_close(x.grad, t.grad)
        zero_grad(x, t)


def test_mean_children():
    shapes = (2, 3, 4)
    t, x = make_values([shapes], high=3, low=2)
    for dim1, dim2 in [((0,), (1,)), ((0,), (2,)), ((1,), (2,))]:

        x_m1, x_m2 = x.mean(dim1, keepdim=True), x.mean(dim2, keepdim=True)
        t_m1, t_m2 = t.mean(dim1, keepdim=True), t.mean(dim2, keepdim=True)

        ((x_m1) ** 3 - (x_m1) ** 2).sum().backward()
        ((x_m2) ** 0.25 - (x_m2) ** -1).sum().backward()
        ((t_m1) ** 3 - (t_m1) ** 2).sum().backward(retain_graph=True)
        ((t_m2) ** 0.25 - (t_m2) ** -1).sum().backward(retain_graph=True)

        assert is_close(x.grad, t.grad)
        zero_grad(x, t)


def test_split():
    shapes = (2, 3, 4)
    t, x = make_values([shapes], high=3, low=2)
    x1, x2 = x.split(0)
    t1, t2 = t

    (x1**2 - x1**-1).sum().backward()
    (t1**2 - t1**-1).sum().backward()
    assert is_close(x.grad, t.grad)

    zero_grad(x1, x2, x, t)

    (x1**2 - x2**2).sum().backward()
    (t1**2 - t2**2).sum().backward()
    assert is_close(x.grad, t.grad)


def test_sum():
    shapes = (1, 2, 3, 4, 5)
    t, x = make_values([shapes], high=3, low=2)

    f = lambda x: (x**2 - x).sum().backward()
    keep = [True, False]
    dims = [(0, -1), (0, 3, 4)]
    for keepdim in keep:
        for dim in dims:
            k = t.sum(dim, keepdim=keepdim)
            y = x.sum(dim, keepdim=keepdim)
            f(k)
            f(y)
            assert is_close(x.grad, t.grad)
            zero_grad(x, t)
