# -*- coding: utf-8 -*-
"""
    lite generator version
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from AnimeGANv2.tools.init_net import weights_init

class ResidualBlock(nn.Module):

    def __init__(self,ch,kernel_size=3,stride=1,padding=1):
        """
        Args:
        Description:
            This model do not change the size of feature map
        """

        super(ResidualBlock,self).__init__()

        # this model do not change the size of feature map
        self.conv_1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_1 = nn.InstanceNorm2d(ch)
        self.norm_2 = nn.InstanceNorm2d(ch)

    def forward(self,x):
        output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
        return output + x #ES

class Generator(nn.Module):
    """
    Generator of AnimeGAN
    """

    def __init__(self,in_nc=3,out_nc=3,nf=32,nb=6):
        super(Generator,self).__init__()

        self.down_convs = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=nf*2, out_channels=nf*2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=nf*4, out_channels=nf*4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(nf*4),
            nn.ReLU(True)
        )        

        # residual blocks #
        residualBlocks = []
        for l in range(nb):
            residualBlocks.append(ResidualBlock(nf*4))
        self.res = nn.Sequential(*residualBlocks)

        # up-convolution #
        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf*4, out_channels=nf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(nf, nf, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3), #k7n3s1
            nn.Tanh()
        )

        weights_init(self)

    def forward(self,x):
        x = self.down_convs(x)
        x = self.res(x)
        x = self.up_convs(x)

        return x

def test():
    """
    Test Generator
    """
    netG = Generator()
    print(netG)
    img = torch.rand([1,3,360,640])
    print(img)
    res = netG(img)
    print(res.shape)

if __name__ == "__main__":
    test()
