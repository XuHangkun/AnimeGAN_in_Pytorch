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

class Conv2DNormaLReLU(nn.Module):

    def __init__(self,in_ch,out_ch,kernel_size=3,stride = 1,padding=1,bias=False):
        """
        Args:
            in_ch (int) : input channel
            out_ch(int) : output channel
            kernel_size, stride, padding, bias are same with nn.Conv2d
        """
        super(Conv2DNormaLReLU,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2,inplace=True)
        )

    def forward(self,input):
        return self.model(input)

class ResBlock(nn.Module):
    
    def __init__(self,ch):
        """
        Args:
            ch(int) : input and output channel
        Description:
            This model do not change the size of feature map
        """

        super(ResBlock,self).__init__()

        # this model do not change the size of feature map
        self.model = nn.Sequential(
            Conv2DNormaLReLU(ch,ch*2,kernel_size=1,padding=0),
            Conv2DNormaLReLU(ch*2,ch*2,kernel_size=3,padding=1),
            nn.Conv2d(ch*2,ch,kernel_size=1,padding=0)
        )

    def forward(self,x):
        return self.model(x) + x

class Unsample(nn.Module):

    def __init__(self,in_ch,out_ch,kernel_size=3,stride = 1,padding=1):
        """
        Unsample the feature map
        [H,W] --> [2*H,2*W]
        """
        super(Unsample,self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_ch,in_ch,4,2,1,bias=False),
            Conv2DNormaLReLU(in_ch,out_ch,kernel_size=kernel_size,padding=padding,stride=stride)
        )

    def forward(self,x):
        return self.main(x)

class Generator(nn.Module):
    """
    Generator of AnimeGAN
    """

    def __init__(self):
        super(Generator,self).__init__()

        self.main = nn.Sequential(
            # A
            Conv2DNormaLReLU(3,32,kernel_size=7,padding=3),
            Conv2DNormaLReLU(32,32,stride=2),
            Conv2DNormaLReLU(32,32),

            #B
            Conv2DNormaLReLU(32,64,stride=2),
            Conv2DNormaLReLU(64,64),
            Conv2DNormaLReLU(64,64),

            #C
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),

            #D Unsample, [H,W] --> [2*H,2*W]
            Unsample(64,64),
            Conv2DNormaLReLU(64,64),
            Conv2DNormaLReLU(64,64),

            #E Unsample, [H,W] --> [2*H,2*W]
            Unsample(64,32),
            Conv2DNormaLReLU(32,32),
            Conv2DNormaLReLU(32,32,kernel_size=7,padding=3),

            #F
            nn.Conv2d(32,3,kernel_size=1,stride=1,padding=0),
            nn.Tanh()
        )


    def forward(self,x):
        return self.main(x)

def test():
    """
    Test Generator
    """
    netG = Generator()
    img = torch.rand([12,3,720,1280])
    res = netG(img)
    print(res.shape)

if __name__ == "__main__":
    test()
