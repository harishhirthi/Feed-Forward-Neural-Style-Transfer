import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""Class to create Residual block."""
class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()

        self.conv_1 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect')
        self.inst_1 = nn.InstanceNorm2d(channels, affine = True)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect')
        self.inst_2 = nn.InstanceNorm2d(channels, affine = True)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        out = self.relu(self.inst_1(self.conv_1(x)))
        out = self.inst_2(self.conv_2(out))
        return residue + out

"""_____________________________________________________________________________________________________________________________________________________________"""    

"""Class to create Upsampling of image."""
class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(UpConv, self).__init__()
        self.factor = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                              stride = 1, padding = padding, padding_mode = 'reflect'
                             )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor > 1:
            x = F.interpolate(x, scale_factor = self.factor)
        return self.conv(x)
    
"""_____________________________________________________________________________________________________________________________________________________________"""

"""Class to create Transformer Net."""
class TransFormerNet(nn.Module):

    """
    Leon A. Gatys, Alexander S. Ecker, Matthias Bethge paper [1].
    Justin Johnson, Alexandre Alahi, Li Fei-Fei paper [2].
    It follows Original Johnson's Architecture [3].
    Instance normalization is used in the place of Batch Normalization. It prevents instance-specific mean and covariance shift simplifying the learning process [4].

    """
    def __init__(self):
        super().__init__()

        # Down Conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 9, stride = 1, padding = 9 // 2, padding_mode = 'reflect')
        self.inst1 = nn.InstanceNorm2d(32, affine = True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect')
        self.inst2 = nn.InstanceNorm2d(64, affine = True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect')
        self.inst3 = nn.InstanceNorm2d(128, affine = True)
        

        # Residual Blocks
        self.resblock1 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(128)
        self.resblock3 = ResidualBlock(128)
        self.resblock4 = ResidualBlock(128)
        self.resblock5 = ResidualBlock(128)

        # Up conv
        self.up_conv1 = UpConv(128, 64, kernel_size = 3, stride = 2, padding = 1)
        self.inst4 = nn.InstanceNorm2d(64, affine = True)
        self.up_conv2 = UpConv(64, 32, kernel_size = 3, stride = 2, padding = 1)
        self.inst5 = nn.InstanceNorm2d(32, affine = True)
        self.up_conv3 = UpConv(32, 3, kernel_size = 9, stride = 1, padding = 9 // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = F.relu(self.inst1(self.conv1(x)))
        out = F.relu(self.inst2(self.conv2(out)))
        out = F.relu(self.inst3(self.conv3(out)))
        
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)

        out = F.relu(self.inst4(self.up_conv1(out)))
        out = F.relu(self.inst5(self.up_conv2(out)))
        out = self.up_conv3(out)

        return out
    


if __name__ == '__main__':

    net = TransFormerNet()
    Num_of_parameters = sum(p.numel() for p in net.parameters())
    print("Model Parameters : {:.3f} M".format(Num_of_parameters / 1e6))


"""
References:
[1] https://arxiv.org/abs/1508.06576
[2] https://arxiv.org/abs/1603.08155
[3] https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
[4] https://arxiv.org/pdf/1607.08022.pdf

"""