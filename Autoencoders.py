import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, num_channels=300):
        super(Autoencoder, self).__init__()
        self.num_channels = num_channels

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(self.num_channels, 150, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(150, 75, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(75, 64, kernel_size=3, padding=1)

        # Decoder layers
        self.dec_conv1 = nn.Conv2d(64, 75, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(75, 150, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(150, self.num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoding
        x = F.relu(self.enc_conv1(x))
        print("After enc_conv1:", x.shape)
        x = F.relu(self.enc_conv2(x))
        print("After enc_conv2:", x.shape)
        x = F.relu(self.enc_conv3(x))
        print("After enc_conv3:", x.shape)

        # Decoding
        x = F.relu(self.dec_conv1(x))
        print("After dec_conv1:", x.shape)
        x = F.relu(self.dec_conv2(x))
        print("After dec_conv2:", x.shape)
        x = F.sigmoid(self.dec_conv3(x))
        print("After dec_conv3:", x.shape)
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
