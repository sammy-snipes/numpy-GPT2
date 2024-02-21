import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(self.head_dim * num_heads, embed_dim, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        qvk = self.in_proj(x)
        q, v, k = tuple(
            rearrange(qvk, "b t (d k h) -> k b h t d", k=3, h=self.num_heads)
        )

        scaled_prod = einsum("bhid,bhjd->bhij", q, k) * (self.head_dim) ** -0.5

        mask = torch.tril(torch.ones_like(scaled_prod))
        scaled_prod = scaled_prod.masked_fill(mask == 0, -float("inf"))

        attention = torch.softmax(scaled_prod, dim=-1)
        out = einsum("bhij,bhjd->bhid", attention, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, p=0.0):
        super().__init__()
        self.ln1, self.ln2 = [nn.LayerNorm(embed_dim) for _ in range(2)]
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, seq_length, n_blocks, device="cpu"
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_length, embed_dim)
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.device = device

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, new_tokens):
        for _ in range(new_tokens):
            idx_cond = idx[:, -seq_length:]
            logits = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
