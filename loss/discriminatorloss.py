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
from config.config import device
class DiscriminatorLoss(nn.Module):
    """
    Module of discriminator loss
    """
    def __init__(self,w_adv=300):
        super(DiscriminatorLoss, self).__init__()
        self.huber_loss = nn.SmoothL1Loss().to(device)
        self.mse = nn.MSELoss().to(device)
        self.l1_loss = nn.L1Loss().to(device)
        self.w_adv=w_adv

    def forward(self,
        discriminator_output_of_cartoon_input,
        discriminator_output_of_cartoon_grey_input,
        discriminator_output_of_cartoon_smooth_grey_input,
        discriminator_output_of_generated_image_input
        ):

        d_loss_carton = self.mse(
                discriminator_output_of_cartoon_input,
                torch.ones(discriminator_output_of_cartoon_input.shape).to(device, torch.float)
            )

        d_loss_generated = self.mse(
                discriminator_output_of_generated_image_input,
                torch.zeros(discriminator_output_of_generated_image_input.shape).to(device, torch.float)
            )

        d_loss_carton_grey = self.mse(
                discriminator_output_of_cartoon_grey_input,
                torch.zeros(discriminator_output_of_cartoon_grey_input.shape).to(device, torch.float)
            )

        d_loss_carton_smooth_grey = self.mse(
                discriminator_output_of_cartoon_smooth_grey_input,
                torch.zeros(discriminator_output_of_cartoon_smooth_grey_input.shape).to(device, torch.float)
            )

        d_loss = self.w_adv*d_loss_carton + d_loss_generated + d_loss_carton_grey + 0.1*d_loss_carton_smooth_grey
        return d_loss



def test():
    print("Test DiscriminatorLoss")
    DLoss = DiscriminatorLoss()
    a = torch.full([4,3,255,255],0.75).to(device, torch.float)
    loss = DLoss(a,a,a,a)
    print(loss)

if __name__ == "__main__":
    test()

