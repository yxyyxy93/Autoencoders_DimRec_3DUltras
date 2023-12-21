import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_channels=300):
        super(Encoder, self).__init__()
        self.enc_conv1 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(num_channels // 2, num_channels // 4, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(num_channels // 4, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, num_channels=300):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(num_channels)
        self.dec_conv1 = nn.Conv2d(64, num_channels // 4, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(num_channels // 4, num_channels // 2, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(num_channels // 2, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.sigmoid(self.dec_conv3(x))
        return x


if __name__ == "__main__":
    # Define the input dimensions
    batch_size = 4
    num_channels = 256  # This can now be modified
    height = 21
    width = 21

    # Create a dummy input tensor with the specified dimensions
    x = torch.randn(batch_size, num_channels, height, width)

    # Initialize the Autoencoder model
    model = Autoencoder(num_channels=num_channels)

    print("Input shape:", x.shape)

    # Pass the input tensor through the model
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)
    print("Output max value:", output.max().item())
    print("Output min value:", output.min().item())
