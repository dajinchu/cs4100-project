import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    NN to approximate a Q-table
    """

    def __init__(self):
        """
        Initialize the neural network
        """

        super().__init__()

        # 8 x 8 states, 3 one-hot encoded vectors
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 2, 3, padding=1)
        self.fc1 = nn.Linear(128, 64)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): tensor of shape (N, 8, 8, 3) where N is the batch size
        """

        x = torch.unsqueeze(x, 0)
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x

class QNetworkFC(nn.Module):
    """
    NN to approximate a Q-table
    """

    def __init__(self):
        """
        Initialize the neural network
        """

        super().__init__()

        # 8 x 8 states, 3 one-hot encoded vectors
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): tensor of shape (N, 8, 8, 3) where N is the batch size
        """

        x = torch.flatten(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
