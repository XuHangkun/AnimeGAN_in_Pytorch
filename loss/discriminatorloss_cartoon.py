# -*- coding: utf-8 -*-
"""
    discriminator loss
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

import torch
import torch.nn as nn
from AnimeGANv2.config.config import device
class DiscriminatorLoss(nn.Module):
    """
    Module of discriminator loss
    """
    def __init__(self,w_adv=1.0):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self,
        discriminator_output_of_cartoon_input,
        discriminator_output_of_cartoon_smooth_input,
        discriminator_output_of_generated_image_input
        ):
        
        d_loss_carton = self.bce_loss(
                discriminator_output_of_cartoon_input,
                torch.ones(discriminator_output_of_cartoon_input.shape).to(device, torch.float)
            )

        d_loss_generated = self.bce_loss(
                discriminator_output_of_generated_image_input,
                torch.zeros(discriminator_output_of_generated_image_input.shape).to(device, torch.float)
            )
        
        d_loss_carton_smooth = self.bce_loss(
                discriminator_output_of_cartoon_smooth_input,
                torch.zeros(discriminator_output_of_cartoon_smooth_input.shape).to(device, torch.float)
            )
        
        d_loss = 1.0*d_loss_carton + d_loss_generated + 1.0*d_loss_carton_smooth
        return d_loss

        

def test():
    print("Test DiscriminatorLoss")
    DLoss = DiscriminatorLoss()   
    a = torch.full([4,3,255,255],0.75).to(device, torch.float)
    loss = DLoss(a,a,a)
    print(loss)

if __name__ == "__main__":
    test()