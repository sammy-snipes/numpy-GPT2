from typing import Any, Union, List, Tuple
import numpy as np
import string
import torch
import torch.nn as nn
from numpyGPT.engine import Parameter
from numpyGPT.functions import (
    einsum,
    softmax,
    rearrange,
    relu,
    layer_norm,
    gelu,
    cross_entropy_loss,
    dropout,
    embed,
)
import numpyGPT.pytorch_reference as r


class Module:
    def __init__(self) -> None:
        pass

    def forward(self, *args) -> Any:
        pass

    def __call__(self, *args) -> Any:
        return self.forward(*args)

    @staticmethod
    def set_attrs(x: nn.Module, a: "Module", attrs: List[Tuple]) -> "Module":
        for attr, conv_func in attrs:
            setattr(a, attr, conv_func(getattr(x, attr)))
        return a

    @staticmethod
    def _weight_to_param(x: Union[None, torch.Tensor]):
        return 1 if x is None else Parameter(x.detach().numpy())

    @staticmethod
    def _bias_to_param(x: Union[None, torch.Tensor]):
        return 0 if x is None else Parameter(x.detach().numpy())

    @staticmethod
    def _do_nothing(x):
        return x

    @classmethod
    def _from_torch(cls, x: torch.nn.Module) -> "Module":
        return cls.__new__(cls)

    def parameters(self):
        params, learnable_attrs = [], ["weight", "bias"]
        for attr in learnable_attrs:
            if hasattr(self, attr):
                params.append(getattr(self, attr))
        return params


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.weight = Parameter(shape=(out_features, in_features))
        self.bias = (
            Parameter(shape=(out_features,))
            if bias
            else Parameter(np.zeros((out_features,)))
        )

    @classmethod
    def _from_torch(cls, x: nn.Module):
        self = cls.__new__(cls)
        attrs = [("weight", self._weight_to_param), ("bias", self._bias_to_param)]
        return self.set_attrs(x, self, attrs)

    def forward(self, x: Parameter) -> Any:
        x_ptrn = string.ascii_lowercase[: x.dim + 1]
        x_ptrn, new_dim = x_ptrn[:-1], x_ptrn[-1]
        ptrn = f"{x_ptrn},{new_dim + x_ptrn[-1]} -> {x_ptrn[:-1] + new_dim}"
        return einsum(ptrn, x, self.weight) + self.bias


class Softmax(Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim

    @classmethod
    def _from_torch(cls, x: torch.nn.Module):
        self = cls.__new__(cls)
        attrs = [("dim", self._do_nothing)]
        return self.set_attrs(x, self, attrs)

    def forward(self, x: Parameter):
        return softmax(x, dim=self.dim)


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Parameter):
        return relu(x)


class GELU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Parameter):
        return gelu(x)


class Dropout(Module):
    def __init__(self, p) -> None:
        super().__init__()
        self.active = True
        self.p = p

    @classmethod
    def _from_torch(cls, x: torch.nn.Module):
        self = cls.__new__(cls)
        attrs = [("p", self._do_nothing)]
        self.active = True
        return self.set_attrs(x, self, attrs)

    def forward(self, x: Parameter):
        return dropout(x, self.p) if self.active else x

    def deactivate(self):
        self.active = False

    def activate(self):
        self.active = True


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.weight = Parameter(shape=(num_embeddings, embedding_dim))

    @classmethod
    def _from_torch(cls, x: torch.nn.Module) -> "Module":
        self = cls.__new__(cls)
        attrs = [("weight", self._weight_to_param)]
        return self.set_attrs(x, self, attrs)

    def forward(self, x: Parameter):
        return embed(x.data, self.weight)


class CrossEntropyLoss(Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim

    @classmethod
    def _from_torch(cls, x: torch.nn.Module):
        self = cls.__new__(cls)
        attrs = [("dim", self._do_nothing)]
        return self.set_attrs(x, self, attrs)

    def forward(self, x: Parameter, y: Parameter):
        return cross_entropy_loss(x, y, dim=self.dim)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = normalized_shape
        self.weight = Parameter(np.ones(normalized_shape) if elementwise_affine else 1)
        self.bias = Parameter(np.zeros(normalized_shape) if bias else 0)
        self.eps = eps

    @classmethod
    def _from_torch(cls, x: nn.Module):
        self = cls.__new__(cls)
        attrs = [
            ("weight", self._weight_to_param),
            ("bias", self._bias_to_param),
            ("normalized_shape", self._do_nothing),
            ("elementwise_affine", self._do_nothing),
            ("eps", self._do_nothing),
        ]
        return self.set_attrs(x, self, attrs)

    def forward(self, x: Parameter):
        return self.weight * layer_norm(x, self.normalized_shape, self.eps) + self.bias


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self.layers = [a for a in args]

    @classmethod
    def _from_torch(cls, x: nn.Sequential):
        self = cls.__new__(cls)
        self.layers = []
        for layer in x:
            self.layers.append(convert_nn_module(layer))
        return self

    def forward(self, x: Parameter):
        x_out = x
        for l in self.layers:
            x_out = l(x_out)
        return x_out

    def parameters(self):
        params = [l.parameters() for l in self.layers]
        return sum(params, [])


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.in_proj = Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

    @classmethod
    def _from_torch(cls, x: nn.Module):
        self = cls.__new__(cls)
        attrs = [
            ("in_proj", convert_nn_module),
            ("out_proj", convert_nn_module),
            ("embed_dim", self._do_nothing),
            ("num_heads", self._do_nothing),
            ("head_dim", self._do_nothing),
        ]
        return self.set_attrs(x, self, attrs)

    def forward(self, x: Parameter):
        qvk = self.in_proj(x)
        qvk = rearrange(qvk, "b t (d k h) -> k b h t d", k=3, h=self.num_heads)
        q, v, k = qvk.split(0)

        scaled_product = (self.head_dim**-0.5) * einsum("bhid,bhjd->bhij", q, k)

        mask = np.tril(np.ones_like(scaled_product.data))
        scaled_product = scaled_product.masked_fill(mask == 0, -np.inf)

        attention = softmax(scaled_product, dim=-1)
        out = einsum("bhij,bhjd->bhid", attention, v)
        out = rearrange(out, "b h t d -> b t (h d)", h=self.num_heads, d=self.head_dim)
        return self.out_proj(out)

    def parameters(self):
        return self.in_proj.parameters() + self.out_proj.parameters()


class Block(Module):
    def __init__(self, embed_dim, num_heads, p=0.0):
        super().__init__()
        self.ln1, self.ln2 = [
            LayerNorm(normalized_shape=(embed_dim,)) for _ in range(2)
        ]
        self.attn = MultiheadAttention(embed_dim, num_heads)

        self.mlp = Sequential(
            Linear(embed_dim, embed_dim * 4),
            GELU(),
            Linear(embed_dim * 4, embed_dim),
            Dropout(p),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    @classmethod
    def _from_torch(cls, x: nn.Module):
        self = cls.__new__(cls)
        attrs = [
            ("ln1", convert_nn_module),
            ("ln2", convert_nn_module),
            ("attn", convert_nn_module),
            ("mlp", convert_nn_module),
        ]
        return self.set_attrs(x, self, attrs)

    def parameters(self):
        return (
            self.ln1.parameters()
            + self.ln2.parameters()
            + self.attn.parameters()
            + self.mlp.parameters()
        )


class GPT(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, seq_length, n_blocks) -> None:
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.position_embedding = Embedding(seq_length, embed_dim)
        self.blocks = Sequential(
            *[Block(embed_dim, num_heads) for _ in range(n_blocks)]
        )
        self.ln_f = LayerNorm((embed_dim,))
        self.lm_head = Linear(embed_dim, vocab_size)

    def forward(self, idx: Parameter):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(Parameter(np.arange(T)))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def _from_torch(cls, x: nn.Module):
        self = cls.__new__(cls)
        attrs = [
            ("token_embedding", convert_nn_module),
            ("position_embedding", convert_nn_module),
            ("blocks", convert_nn_module),
            ("ln_f", convert_nn_module),
            ("lm_head", convert_nn_module),
        ]
        return self.set_attrs(x, self, attrs)

    def parameters(self):
        return sum(
            [
                _.parameters()
                for _ in (
                    self.token_embedding,
                    self.position_embedding,
                    self.blocks,
                    self.ln_f,
                    self.lm_head,
                )
            ],
            [],
        )


CONVERSION_DICT = {
    nn.Linear: Linear,
    nn.Softmax: Softmax,
    nn.CrossEntropyLoss: CrossEntropyLoss,
    nn.ReLU: ReLU,
    nn.GELU: GELU,
    nn.Embedding: Embedding,
    nn.Dropout: Dropout,
    nn.LayerNorm: LayerNorm,
    nn.Sequential: Sequential,
    r.MultiHeadSelfAttention: MultiheadAttention,
    r.Block: Block,
}


def convert_nn_module(x: nn.Module):
    return CONVERSION_DICT[type(x)]._from_torch(x)
