import torch
import torch.nn.functional as F
from torch import nn, flatten


class Mnih2013(nn.Module):
    """The model used by MNIH 2013 paper of DQN."""

    def __init__(self,
                 conv1_in_channels,
                 conv1_out_channels,
                 conv1_kernel_size,
                 conv1_stride,
                 conv2_out_channels,
                 conv2_kernel_size,
                 conv2_stride,
                 fc1_in_features,
                 fc1_out_features,
                 num_logits,
                 add_value,
                 output_features):
        super().__init__()
        self.add_value = add_value
        self.output_features = output_features

        self.conv1 = nn.Conv2d(in_channels=conv1_in_channels,
                               out_channels=conv1_out_channels,
                               kernel_size=conv1_kernel_size,
                               stride=conv1_stride)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,
                               out_channels=conv2_out_channels,
                               kernel_size=conv2_kernel_size,
                               stride=conv2_stride)

        self.fc1 = nn.Linear(fc1_in_features,
                             fc1_out_features)

        self.output = nn.Linear(fc1_out_features,
                                num_logits)

        if self.add_value:
            self.value_head = nn.Linear(fc1_out_features, 1)

    def forward(self, x: torch.Tensor):

        # Make sure we're feeding in the correct number of channels
        assert(x.shape[1] == self.conv1.in_channels)

        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)

        # Apply the dense layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply the final policy head
        logits = self.output(x)

        if self.add_value:
            value = flatten(self.value_head(x))  # Flatten the value output
            if self.output_features:
                return logits, value, x
            else:
                return logits, value
        else:
            if self.output_features:
                return logits, x
            else:
                return logits
