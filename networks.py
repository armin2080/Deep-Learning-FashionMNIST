import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # First layer

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # Second layer

        # Fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 6) # Output layer for six classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        
        x = x.view(x.size(0), -1) # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        
        return self.fc2(x)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)      
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6) # Output layer for six classes

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flatten the tensor
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)