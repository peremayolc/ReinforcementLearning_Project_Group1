import torch  # For tensor operations
import torch.nn as nn  # For defining the neural network and layers


class DQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        """
        Initializes a convolutional DQN for processing visual input.
        Args:
            input_shape (tuple): The shape of the input observation).
            action_dim (int): The number of possible actions in the environment.
        """
        super(DQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        # Calculate the flattened size after the convolutional layers
        conv_output_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

        # Apply Xavier initialization
        self._initialize_weights()

    def _get_conv_output_size(self, input_shape):
        """
        Computes the size of the output after passing the input through the convolutional layers.
        Args:
            input_shape (tuple): The shape of the input observation).
        Returns:
            int: Flattened size after convolutions.
        """
        # Create a dummy tensor to pass through the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Batch size of 1
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.numel()  # Total number of elements in the tensor

    def _initialize_weights(self):
        """
        Initializes weights of the network using Xavier initialization.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Forward pass of the DQN.
        Args:
            x (torch.Tensor): Input tensor representing the state.
        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Pass through convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output Q-values
        return x  # Ensure this line is present