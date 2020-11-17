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
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): tensor of shape (N, 8, 8, 3) where N is the batch size
        """

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
