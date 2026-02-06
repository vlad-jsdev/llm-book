import torch

tensor2d = torch.tensor([[1,2,3],[4,5,6,]])

print(tensor2d)
print(tensor2d.matmul(tensor2d.T))
