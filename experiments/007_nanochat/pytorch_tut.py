import torch
from torch import nn
from torch.utils.data import DataLoader

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNet(nn.module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNet().to(device)
print(model)

"""
Pytorch Cheat sheet:
torch.rand_like(x_data, dtype=torch.float) # create a tensor of the same shape and data type as x_data
torch.zeros_like(x_data, dtype=torch.float) # create a tensor of the same shape and data type as x_data
torch.ones_like(x_data, dtype=torch.float) # create a tensor of the same shape and data type as x_data
t = torch.tensor([1, 2, 3])
t.shape, t.dtype, t.device

all do mat mul
y1 = t @ t2
y2 = t.matmul(t2)
torch.matmul(t, t2, out=y3)

element-wise multiplication
z1 = t1 * t2
z2 = t1.mul(t2)
z3 = torch.mul(t1, t2)

single element tensor can convert back to a scalar by calling .item()


"""
