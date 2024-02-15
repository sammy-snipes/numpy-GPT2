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
        x = x + self.attn(self.ln1(x))  # Resnet connection
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    batch_size = 16
    seq_length = 64
    max_iters = 10000

    with open("test.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    s_to_i = {ch: i for i, ch in enumerate(chars)}
    i_to_s = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [s_to_i[c] for c in s]
    decode = lambda l: "".join([i_to_s[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    def get_batch():
        ix = torch.randint(len(data) - seq_length, (batch_size,))
        x = torch.stack([data[i : i + seq_length] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + seq_length] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    model = GPT(
        embed_dim=256,
        num_heads=8,
        seq_length=seq_length,
        n_blocks=4,
        vocab_size=vocab_size,
        device=device,
    )
    optim = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    print_every = 100
    for iter in range(max_iters):
        x, y = get_batch()
        optim.zero_grad()
        logits = model(x)
        logits = rearrange(logits, "B T C -> (B T) C")
        targets = rearrange(y, "B T -> (B T)")
        err = loss(logits, targets)
        err.backward()
        optim.step()

        if iter % print_every == 0:
            print(25 * "=", f"iter : {iter}", 25 * "=")
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(decode(model.generate(context, new_tokens=500)[0].tolist()))
