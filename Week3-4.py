import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Perceptron for AND gate
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# AND gate training
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
perceptron = Perceptron(2)
perceptron.train(X, y)
print("AND gate predictions:", [perceptron.predict(xi) for xi in X])

# PyTorch Neural Network for XOR
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# XOR problem
xor_inputs = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
xor_targets = torch.FloatTensor([[0],[1],[1],[0]])

model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training
for epoch in range(1000):
    outputs = model(xor_inputs)
    loss = criterion(outputs, xor_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test XOR
with torch.no_grad():
    predictions = model(xor_inputs)
    print("XOR predictions:", predictions.round().int().flatten().tolist())
