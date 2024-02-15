import torch.nn as nn
import torch
import torch.nn.functional as F
from numpyGPT.models import (
    CrossEntropyLoss,
    MultiheadAttention,
    Block,
    GPT,
)
from numpyGPT.engine import Parameter
from tests.utils import make_values, is_close, param_grad_is_close
import numpyGPT.pytorch_reference as r
import numpy as np
import einops
from numpyGPT.functions import rearrange


def test_mha_naive():
    batch_size = 8
    seq_length = 32
    embed_dim = 64
    num_heads = 4
    t, p = make_values([(batch_size, seq_length, embed_dim)])

    t_mha = r.MultiHeadSelfAttention(embed_dim, num_heads)
    p_mha = MultiheadAttention._from_torch(t_mha)

    t_out, p_out = t_mha(t), p_mha(p)
    t_out.sum().backward()
    p_out.sum().backward()

    assert param_grad_is_close(t_mha.parameters(), p_mha.parameters())
    assert is_close(t.grad, p.grad)
    assert is_close(t_out, p_out.data)


def test_block_naive():
    batch_size = 8
    seq_length = 32
    embed_dim = 64
    num_heads = 4
    t, p = make_values([(batch_size, seq_length, embed_dim)])

    t_block = r.Block(embed_dim, num_heads, p=0)
    p_block = Block._from_torch(t_block)

    t_out, p_out = t_block(t), p_block(p)
    t_out.sum().backward()
    p_out.sum().backward()

    assert param_grad_is_close(t_block.parameters(), p_block.parameters())
    assert is_close(t.grad, p.grad)
    assert is_close(t_out, p_out.data)


def test_gpt():
    vocab_size = 10
    embed_dim = 128
    num_heads = 4
    batch_size = 4
    n_blocks = 2
    seq_length = 16

    t_model = r.GPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_length=seq_length,
        n_blocks=n_blocks,
    )
    p_model = GPT._from_torch(t_model)

    t = torch.randint(0, vocab_size, size=(batch_size, seq_length))
    t_target = torch.randint(0, vocab_size, size=(batch_size * seq_length,))

    p = Parameter(t.detach().numpy())
    p_target = Parameter(F.one_hot(t_target, vocab_size).detach().double().numpy())

    t_out = t_model(t)
    t_out = einops.rearrange(t_out, "a b c -> (a b) c")
    t_loss = nn.CrossEntropyLoss()(t_out, t_target)

    p_out = p_model(p)
    p_out = rearrange(p_out, "a b c -> (a b) c", a=batch_size, b=seq_length)
    p_loss = CrossEntropyLoss()(p_out, p_target)

    p_loss.backward()
    t_loss.backward()

    for child in t_model.children():
        print(50 * "=")
        print(child)
    assert is_close(t_out, p_out.data)
    assert is_close(t_loss, p_loss.data)
    assert param_grad_is_close(p_model.parameters(), t_model.parameters())
