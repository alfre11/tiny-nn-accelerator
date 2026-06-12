import torch

state = torch.load("data/tiny_nn_float.pth")

for name, tensor in state.items():
    print(name, tensor.shape)