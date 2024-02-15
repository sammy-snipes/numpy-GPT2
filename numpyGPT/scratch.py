import sys

sys.path.append("..")
import torch
import numpy as np

# x = torch.tensor([0, 0, 2, 3, 2])
#
# torch.manual_seed(42)
# embed = torch.nn.Embedding(4, 2)
# y = embed(x)
# y.retain_grad()
#
# (y**2).sum().backward()
#
# print   (y.grad, embed.weight.grad, embed.weight, sep=2 * "\n")

np.random.seed(42)

x = np.array([0, 0, 0, 0])

embed = np.random.randn(2, 2)

y = embed[x]
grad = 2 * y

local_grad = np.zeros_like(embed)
np.add.at(local_grad, x, grad)
print(local_grad, grad, sep=2 * "\n")
