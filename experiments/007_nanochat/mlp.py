import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.out = nn.Linear(10, 3)
    def forward(self, x):
        logits = self.l1(x)
        activations = nn.functional.relu(logits)
        out_logits = self.out(activations)
        return nn.functional.softmax(out_logits)


if __name__ == "__main__":
    mlp = MLP()
    x = torch.randn(10)
    results = mlp(x)
    print(results)
    best = torch.argmax(results)
    print(best)