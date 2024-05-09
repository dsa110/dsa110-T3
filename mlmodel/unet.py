import torch 
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, kernel_size=3):
        '''
        Double convolution block for U-Net, includes two convolutions with ReLU activation in between

        Args:

        `in_channels`: number of input channels

        `out_channels`: number of output channels

        `kernel_size`: size of convolution kernel

        `dropout`: dropout probability
        '''
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        # x = self.batchnorm2(x)
        out = self.relu2(x)
        out = self.dropout2(out)
        return out


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, kernel_size=3, pool_kernel=2, pool_stride=2):
        '''
        Down convolution block for U-Net, includes DoubleConv block and max pooling

        Args:

        `in_channels`: number of input channels

        `out_channels`: number of output channels

        `kernel_size`: size of convolution kernel

        `pool_kernel`: size of max pooling kernel

        `pool_stride`: stride used for max pooling

        `dropout`: dropout probability
        '''

        super(DownConvBlock, self).__init__()
        self.double_conv = DoubleConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dropout=dropout
            )
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        double_conv = self.double_conv(x)
        out = self.pool(double_conv)
        return double_conv, out


class DownSample(nn.Module):
    def __init__(self, depth, channel_multiplier, initial_channel_size, pool_kernel=2, pool_stride=2, double_conv_kernel=3, dropout=0.5):
        '''
        Downsample block for U-Net, includes multiple DownConvBlocks

        Args:

        `depth`: number of down convolution blocks

        `channel_multiplier`: factor to scale channels by

        `initial_channel_size`: number of channels in first convolution

        `pool_kernel`: size of max pooling kernel

        `pool_stride`: stride used for max pooling

        `double_conv_kernel`: size of convolution kernel

        `dropout`: dropout probability
        '''
        super(DownSample, self).__init__()
        self.down_convs = nn.ModuleList([DownConvBlock(
            1, initial_channel_size,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            kernel_size=double_conv_kernel,
            dropout=dropout,
            )])
        self.down_convs.extend([
            DownConvBlock(
                initial_channel_size * (channel_multiplier ** i), initial_channel_size * (channel_multiplier ** (i + 1)), pool_kernel=pool_kernel,
                pool_stride=pool_stride,
                kernel_size=double_conv_kernel,
                dropout=dropout)
            for i in range(depth)])


    def forward(self, x):
        '''
        Args:

        `x`: output of last down convolution block

        Returns:

        `down_conv_tensors`: list of tensors from each down convolution block in downsampling layer order, i.e. decreasing resolution (used for upsampling)

        `x` (tensor): resulting mapping at bottom of U-Net after applying down convolutions
        '''

        down_conv_tensors = []
        for down_conv in self.down_convs:
            copy, x = down_conv(x)
            down_conv_tensors.append(copy)


        return down_conv_tensors, x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 conv_stride=2, dropout=0.5):
        '''
        Up convolution block for U-Net, includes up convolution, concatenation, and DoubleConv block

        Args:

        `in_channels`: number of input channels

        `out_channels`: number of output channels

        `kernel_size`: size of convolution kernel

        `conv_stride`: stride used for convolution

        `dropout`: dropout probability
        '''

        super(UpConvBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            )
        self.dropout = nn.Dropout(dropout)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)


    def forward(self, x, down_tensor, output_size=None):
        '''

        Args:

        `x` (tensor): input mapping

        `down_tensor`: corresponding mapping from downsampling section of U-Net

        `output_size`: the size to target to match the down_tensor skip connection size

        Returns:

        `out` (tensor): resulting mapping after one layer of upsampling
        '''
        out = self.up_conv(x, output_size)
        out = self.dropout(out)

        if down_tensor.shape != out.shape:
          print(f'tensors of different shapes on upsample: down => ({down_tensor.shape}) vs up => ({out.shape})')
        assert(down_tensor.shape == out.shape)

        out = torch.cat([down_tensor, out], dim=1)
        out = self.double_conv(out)

        return out


class UpSample(nn.Module):
    def __init__(self, depth, channel_multiplier, initial_channel_size, up_conv_kernel=2, up_conv_stride=2, dropout=0.5):
        '''
        Upsample block for U-Net, includes multiple UpConvBlocks

        Args:

        `depth`: number of up convolution blocks

        `channel_multiplier`: factor to scale channels by

        `initial_channel_size`: number of channels in first convolution

        `up_conv_kernel`: size of up convolution kernel

        `up_conv_stride`: stride used for up convolution

        `dropout`: dropout probability
        '''

        super(UpSample, self).__init__()
        self.up_convs = nn.ModuleList([UpConvBlock(
            initial_channel_size * (channel_multiplier ** (depth + 1)), initial_channel_size * (channel_multiplier ** depth),
            conv_stride=up_conv_stride,
            kernel_size=up_conv_kernel,
            dropout=dropout,
            )])
        self.up_convs.extend([
            UpConvBlock(
                initial_channel_size * (channel_multiplier ** (i + 1)), initial_channel_size * (channel_multiplier ** (i)),
                conv_stride=up_conv_stride,
                kernel_size=up_conv_kernel,
                dropout=dropout)
            for i in reversed(range(depth))])

    def forward(self, x, down_conv_tensors):
        '''
        Args:
        `x` (tensor): input mapping

        `down_conv_tensors`: list of tensors resulting from downsampling in top-down order, i.e. decreasing resolution
        '''

        for i, up_conv in enumerate(self.up_convs):
            skip = down_conv_tensors[-(i+1)]
            x = up_conv(x, skip, output_size=skip.shape)

        return x

class UNet(nn.Module):
    def __init__(self, dropout=0.2, depth=2, channel_multiplier=2, initial_channel_size=128, pool_kernel=2, pool_stride=2, double_conv_kernel=3, up_conv_kernel=2, up_conv_stride=2) -> None:
        '''
        Model for U-Net implementation for RFI segmentation

        Args:
        `image_size`: size of input image
        `depth`: number of down and up convolutions in the model (after the first one)
        `channel_multiplier`: factor to scale channels by
        `initial_channel_size`: number of channels in first convolution
        `pool_kernel`: size of max pooling kernel
        `pool_stride`: stride used for max pooling
        `double_conv_kernel`: size of convolution kernel
        `up_conv_kernel`: size of up convolution kernel
        `up_conv_stride`: stride used for up convolution
        `dropout`: dropout probability
        '''
        super(UNet, self).__init__()
        self.down_sample = DownSample(
            depth,
            channel_multiplier,
            initial_channel_size,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            double_conv_kernel=double_conv_kernel,
            dropout=dropout
            )

        self.bottom_conv = DoubleConv(initial_channel_size * (channel_multiplier ** depth), initial_channel_size * (channel_multiplier ** (depth + 1)), kernel_size=double_conv_kernel)

        self.up_sample = UpSample(
            depth,
            channel_multiplier,
            initial_channel_size,
            up_conv_kernel=up_conv_kernel,
            up_conv_stride=up_conv_stride,
            dropout=dropout
            )

        self.end_conv = nn.Conv2d(
            initial_channel_size,
            1,
            kernel_size=1) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        '''
        Returns a mask for the image (flattened) where each pixel represents the probability that that pixel is RFI.
        0 means not RFI and 1 means definitely RFI.

        Args:

        `image` (tensor): input image in format (B, N, H, W) where B is batch size, N is number of channels, H is height, and W is width

        '''

        # Encoding information
        residual_convs, out = self.down_sample(image)
        # Bottom of U-Net
        out = self.bottom_conv(out)
        # Decoding information
        out = self.up_sample(out, residual_convs)
        # Final convolution
        out = self.end_conv(out)
        # out = self.sigmoid(out)

        return out
    
    def forward_probs(self, image):
        '''
        Returns a mask for the image (flattened) where each pixel represents the probability that that pixel is RFI.
        0 means not RFI and 1 means definitely RFI.

        Args:

        `image` (tensor): input image in format (B, N, H, W) where B is batch size, N is number of channels, H is height, and W is width

        '''

        # Encoding information
        residual_convs, out = self.down_sample(image)
        # Bottom of U-Net
        out = self.bottom_conv(out)
        # Decoding information
        out = self.up_sample(out, residual_convs)
        # Final convolution
        out = self.end_conv(out)
        out = self.sigmoid(out)

        return out