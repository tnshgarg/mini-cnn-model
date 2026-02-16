import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 3 x 32 x 32
        # Conv1: 3 -> 16 filters, 3x3 kernel, padding=1 (to keep size)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Pool: 2x2, stride 2 -> 16 x 16
        self.pool = nn.MaxPool2d(2, 2)
        # Conv2: 16 -> 32 filters, 3x3 kernel, padding=1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Pool: 2x2, stride 2 -> 8 x 8
        
        # Fully connected layers
        # Input size: 32 * 8 * 8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10) # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test the model with a random input
    model = SimpleCNN()
    test_input = torch.randn(1, 3, 32, 32)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
