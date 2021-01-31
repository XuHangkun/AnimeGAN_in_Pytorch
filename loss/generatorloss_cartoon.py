# -*- coding: utf-8 -*-
"""
    generator loss
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config.config import device,dataset_mean,dataset_std
import copy

def gram_matrix(input):
    """
    calculate the input feature map to gram matrix
    """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def rgb2yuv(rgb):
    rgb = (rgb + 1)/2.0
    rgb_ = rgb.transpose(1,3)                              # input is batch_size*3*n*n   default
    A = torch.tensor([[0.299, -0.14714119,0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]).to(device, torch.float)   # from  Wikipedia
    yuv = torch.tensordot(rgb_,A,1).transpose(1,3)
    return yuv


class GeneratorLoss(nn.Module):
    """
    Module of generator loss
    """
    def __init__(self,w_adv=300.,w_con = 1.5,w_tv=1.0,w_gra=3.,w_col=10.0,is_init_phase=False):
        super(GeneratorLoss,self).__init__()
        #load vgg as feature extractor
        self.vgg19 = models.vgg19(pretrained=True).to(device)
        self.vgg19.eval()
        self.feature_extractor = self.vgg19.features[:24]
        for param in self.feature_extractor.parameters():
            param.require_grad = False
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.w_adv = w_adv
        self.w_con = w_con
        self.w_gra = w_gra
        self.w_col = w_col
        self.w_tv = w_tv
        self.is_init_phase = is_init_phase

    def forward(self,
        discriminator_output_of_generated_image_input,
        real_image,
        generated_image,
        grey_image
        ):
        if self.is_init_phase:
            loss = self.w_con*self._con_loss(real_image,generated_image)
        else:
            loss = self.w_adv*self._adv_loss(discriminator_output_of_generated_image_input)
            loss += self.w_con*self._con_loss(real_image,generated_image)
            loss += self.w_col*self._col_loss(real_image,generated_image)
            if True:
                loss += self.w_gra*self._gra_loss(generated_image,grey_image)
        return loss


    def _adv_loss(self,discriminator_output_of_generated_image_input):
        """
        compute adversial loss
        """
        g_loss_generated = self.mse_loss(
                discriminator_output_of_generated_image_input,
                torch.ones(discriminator_output_of_generated_image_input.shape).to(device, torch.float)
            )
        return g_loss_generated

    def _con_loss(self,real_img,generated_image):
        """
        compute content loss
        """
        output_fvgg_real_in = self.feature_extractor((real_img+1)/2)
        output_fvgg_generated_in = self.feature_extractor((generated_image+1)/2)
        g_con_loss = self.l1_loss(output_fvgg_real_in,output_fvgg_generated_in)
        return g_con_loss

    def _gra_loss(self,generated_image,grey_image):
        """
        compute gram loss
        """
        output_fvgg_generated_in = gram_matrix(self.feature_extractor(generated_image))
        output_fvgg_grey_in = gram_matrix(self.feature_extractor(grey_image))
        g_gra_loss = self.l1_loss(output_fvgg_generated_in,output_fvgg_grey_in)
        return g_gra_loss

    def _col_loss(self,real_image,generated_image):
        """
        compute color loss
        """
        # size [batch,YUV,H,W]
        real_image = rgb2yuv(real_image)
        generated_image = rgb2yuv(generated_image)

        col_loss  = self.l1_loss(real_image[:,0,:,:],generated_image[:,0,:,:])
        col_loss += self.huber_loss(real_image[:,1,:,:],generated_image[:,1,:,:])
        col_loss += self.huber_loss(real_image[:,2,:,:],generated_image[:,2,:,:])
        return col_loss

    def _tv_loss(self,generated_image):
        """
        A smooth loss in fact. Like the smooth prior in MRF.
        V(y) = || y_{n+1} - y_n ||_2
        """
        dw = self.mse_loss(generated_image[..., :-1],generated_image[...,1:])
        dh = self.mse_loss(generated_image[...,:-1,:],generated_image[..., 1:, :])
        return dh + dw


def test():
    """
    test
    """
    img = torch.rand([4,3,32,32])
    img = img.to(device, torch.float)
    grey_img = torch.rand([4,3,32,32])
    grey_img = grey_img.to(device, torch.float)
    generated_img = torch.rand([4,3,32,32])
    generated_img = generated_img.to(device, torch.float)
    d_img = torch.rand([4,3,16,16])
    d_img = d_img.to(device, torch.float)
    yuv = rgb2yuv(img)
    print(yuv.shape)
    g_loss = GeneratorLoss(is_init_phase=True)
    print(g_loss(d_img,img,generated_img,img))

    # test by feature
    feature_net = g_loss.feature_extractor
    f_1 = feature_net(img)
    f_2 = feature_net(generated_img)
    print(f_1.shape)
    print(f_2.shape)
    l1loss = nn.L1Loss()
    print(l1loss(f_1,f_2))
    print(g_loss)

if __name__ == "__main__":
    test()
