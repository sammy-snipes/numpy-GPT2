import numpy as np
from typing import List, Tuple


class Parameter:
    def __init__(self, data=None, shape=None, _children=()) -> None:
        if data is not None:
            self.data = data if isinstance(data, np.ndarray) else np.array(data)
        elif shape is not None:
            self.data = self._init_normal(shape)
        else:
            raise NotImplementedError

        self.grad = np.zeros_like(self.data)
        self._children = _children
        self.shape = self.data.shape
        self.dim = len(self.shape) if self.shape else 0
        self._backward = lambda: None

    def backward(self):
        assert self.grad.shape == ()
        self.grad = 1.0
        visited, stack = set(), []

        def dfs(node):
            visited.add(node)
            for child in node._children:
                if child not in visited:
                    dfs(child)
            stack.append(node)

        dfs(self)

        for node in stack[::-1]:
            node._backward()

    @staticmethod
    def _init_normal(shape):
        limit = np.sqrt(6 / np.prod(np.array(shape)))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def broadcast_helper(grad: np.ndarray, a: np.ndarray) -> np.ndarray:
        if grad.shape == a.shape:
            return grad
        else:
            sum_dims = tuple(range(len(grad.shape) - len(a.shape)))
            sum_to_one = tuple(_ for _, __ in enumerate(a.shape) if __ == 1)
            return grad.sum(sum_dims).sum(sum_to_one, keepdims=True)

    def __add__(self, other) -> "Parameter":
        other = other if isinstance(other, Parameter) else Parameter(other)
        out = Parameter(self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += self.broadcast_helper(out.grad, self.grad)
            other.grad += self.broadcast_helper(out.grad, other.grad)

        out._backward = _backward
        return out

    def __sub__(self, other: "Parameter") -> "Parameter":
        out = Parameter(self.data - other.data, _children=(self, other))

        def _backward():
            self.grad += self.broadcast_helper(out.grad, self.grad)
            other.grad -= self.broadcast_helper(out.grad, other.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: "Parameter") -> "Parameter":
        out = Parameter(self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += self.broadcast_helper(out.grad * other.data, self.grad)
            other.grad += self.broadcast_helper(out.grad * self.data, other.grad)

        out._backward = _backward
        return out

    def __pow__(self, pow: float) -> "Parameter":
        out = Parameter(self.data**pow, _children=(self,))

        def _backward():
            self.grad += pow * (self.data) ** (pow - 1) * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> "Parameter":
        out = Parameter(self.data * other, _children=(self,))

        def _backward():
            self.grad += other * out.grad

        out._backward = _backward
        return out

    def split(self, dim=0) -> List["Parameter"]:
        data = np.moveaxis(self.data, dim, 0)
        kids = []
        for idx, slice in enumerate(data):
            kid = Parameter(slice, _children=(self,))

            def _undo_split(idx=idx, kid=kid):
                np.moveaxis(self.grad, dim, 0)[idx] += kid.grad

            kid._backward = _undo_split
            kids.append(kid)
        return kids

    def masked_fill(self, mask: np.ndarray, value: float) -> "Parameter":
        out_data = np.copy(self.data)
        out_data[mask] = value
        out = Parameter(out_data, _children=(self,))

        def _backward():
            masked_grad = np.copy(out.grad)
            masked_grad[mask] = 0
            self.grad += masked_grad

        out._backward = _backward
        return out

    def sum(self, dim=None, keepdim=False) -> "Parameter":
        out = Parameter(self.data.sum(axis=dim, keepdims=keepdim), _children=(self,))

        def _backward():
            self.grad += (
                np.expand_dims(out.grad, dim)
                if (dim is not None and not keepdim)
                else out.grad
            )

        out._backward = _backward
        return out

    def mean(self, dim: Tuple[int], keepdim=True) -> "Parameter":
        m = np.mean(self.data, dim, keepdims=keepdim)
        out = Parameter(m, _children=(self,))

        def _backward():
            original_shape = [int(_) for _ in self.data.shape]
            new_shape = [original_shape[d] for d in dim]
            out_grad = out.grad if keepdim else np.expand_dims(out.grad, dim)
            self.grad += out_grad / np.prod(new_shape)

        out._backward = _backward
        return out
