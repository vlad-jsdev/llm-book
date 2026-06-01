import torch
from torch.utils.data import Dataset

# class NeuralNetwork(torch.nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()
#
#         self.layers = torch.nn.Sequential(
# 			torch.nn.Linear(num_inputs, 30),
# 			torch.nn.ReLU(),
#
# 			torch.nn.Linear(30, 20),
# 			torch.nn.ReLU(),
#
# 			torch.nn.Linear(20, num_outputs),
# 		)
#
#     def forward(self, x):
#         logits = self.layers(x)
#         return logits
#
# # model = NeuralNetwork(50, 3)
# # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# torch.manual_seed(123)
# model = NeuralNetwork(50, 3)
# X = torch.randn(1, 50)
# # output = model(X)
# # print(output)
# with torch.no_grad():
# 	out = torch.softmax(model(X), dim=-1)
# print(out)
# print(f"Number of trainable parameters: {num_params}")
# print(model)
#
X_train = torch.tensor([
	[-1.2, 3.1],
	[-0.9, 2.9],
	[-0.5, 2.6],
	[2.3, -1.1],
	[2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
	[-0.8, 2.8],
	[2.6, -1.6]
])

y_test = torch.tensor([0, 1])



class ToyDataset(Dataset):

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))
