"""The CNN that estimates a Q-value for every action from a stack of frames."""

import torch
from torch import nn


class DQN(nn.Module):
    """Convolutional Q-network (same layer sizes as the 2015 DQN Nature paper).

    Input:  a batch of states, shape (batch, 4, 84, 84), pixel values 0-255.
    Output: one Q-value per action, shape (batch, n_actions).
    """

    def __init__(self, n_actions):
        super().__init__()

        # Convolutional layers find visual features (paddles, ball, movement).
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 7, 7)
            nn.ReLU(),
        )

        # Fully connected layers turn those features into one Q-value per action.
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, state):
        # Frames are stored as uint8 (0-255) to save memory; scale them to 0-1 here.
        x = state.float() / 255.0
        x = self.conv_layers(x)
        q_values = self.fc_layers(x)
        return q_values
