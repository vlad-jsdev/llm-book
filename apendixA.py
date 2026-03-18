import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
			torch.nn.Linear(num_inputs, 30),
			torch.nn.ReLU(),

			torch.nn.Linear(30, 20),
			torch.nn.ReLU(),

			torch.nn.Linear(20, num_outputs),
		)

    def forward(self, x):
        logits = self.layers(x)
        return logits

# model = NeuralNetwork(50, 3)
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.manual_seed(123)
model = NeuralNetwork(50, 3)
X = torch.randn(1, 50)
output = model(X)
print(output)
# print(f"Number of trainable parameters: {num_params}")
# print(model)
