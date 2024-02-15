import torch.nn as nn
import torch
import torch.nn.functional as F
from numpyGPT.models import (
    Softmax,
    Linear,
    CrossEntropyLoss,
    LayerNorm,
    Module,
    Sequential,
    MultiheadAttention,
    Dropout,
    Embedding,
)
from numpyGPT.engine import Parameter
from tests.utils import make_values, is_close, param_grad_is_close


def test_linear():
    t, p = make_values([(2, 3)])
    t_linears = [nn.Linear(_, _ + 1) for _ in range(3, 6)]
    p_linears = [Linear._from_torch(_) for _ in t_linears]

    # Check weights
    for p_linear, t_linear in zip(p_linears, t_linears):
        p_linear._from_torch(t_linear)
        assert is_close(p_linear.weight.data, t_linear.weight)
        assert is_close(p_linear.bias.data, t_linear.bias)
    # Check forward
    t_out, p_out = t, p
    for p_linear, t_linear in zip(p_linears, t_linears):
        t_out, p_out = t_linear(t_out), p_linear(p_out)
        assert is_close(t_out, p_out.data)

    p_out.sum().backward()
    t_out.sum().backward()

    # Check grad
    for p_linear, t_linear in zip(p_linears, t_linears):
        p_weight, p_bias = p_linear.weight, p_linear.bias
        t_weight, t_bias = t_linear.weight, t_linear.bias
        assert is_close(p_weight.grad, t_weight.grad)
        assert is_close(p_bias.grad, t_bias.grad)


def test_net():

    batch_size = 64
    n_class = 4

    t, p = make_values([(batch_size, 5)])
    k = torch.randint(0, n_class - 1, size=(batch_size,))
    q = Parameter(F.one_hot(k, n_class).detach().double().numpy())

    t_model = nn.Sequential(
        nn.Linear(5, 5),
        nn.LayerNorm(normalized_shape=(5,)),
        nn.GELU(),
        nn.Dropout(p=0),
        nn.Linear(5, 4),
        nn.ReLU(),
        nn.LayerNorm(normalized_shape=(4,)),
    )
    p_model = Sequential._from_torch(t_model)
    t_out, p_out = t_model(t), p_model(p)

    t_loss = nn.CrossEntropyLoss()(t_out, k)
    p_loss = CrossEntropyLoss()(p_out, q)

    t_loss.backward()
    p_loss.backward()
    assert is_close(p_loss.data, t_loss)
    assert is_close(t.grad, p.grad)
    assert param_grad_is_close(p_model.parameters(), list(t_model.parameters()))


def test_embed():
    batch_size = 64
    n_tokens = 8
    embed_dim = 128

    t = torch.randint(0, n_tokens, size=(batch_size,))
    p = Parameter(t.detach().numpy())

    t_target = torch.randint(0, n_tokens, size=(batch_size,))
    p_target = Parameter(F.one_hot(t_target, n_tokens).detach().double().numpy())

    t_model = nn.Sequential(
        nn.Embedding(n_tokens, embed_dim),
        nn.Linear(embed_dim, embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim, n_tokens),
    )
    p_model = Sequential._from_torch(t_model)

    t_out, p_out = t_model(t), p_model(p)

    t_loss = nn.CrossEntropyLoss()(t_out, t_target)
    p_loss = CrossEntropyLoss()(p_out, p_target)

    p_loss.backward()
    t_loss.backward()

    p_embed = p_model.layers[0]
    t_embed = t_model[0]

    assert is_close(p_embed.weight.grad, t_embed.weight.grad)
    assert is_close(p_loss.data, t_loss)
    assert param_grad_is_close(p_model.parameters(), t_model.parameters())
    assert is_close(t_out, p_out.data)
