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
from AnimeGANv2.config.config import device,dataset_mean,dataset_std
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
    def __init__(self,w_adv=300.,w_con = 1.5,w_gra = 3.,w_col = 10,w_tv=1.0):
        super(GeneratorLoss,self).__init__()
        #load vgg as feature extractor
        self.feature_extractor = models.vgg16(pretrained=True).to(device).features[:24]
        for param in self.feature_extractor.parameters():
            param.require_grad = False
        self.l1_loss = nn.L1Loss().to(device)
        self.huber_loss = nn.SmoothL1Loss().to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.w_adv = w_adv
        self.w_con = w_con
        self.w_gra = w_gra
        self.w_col = w_col
        self.w_tv = w_tv
    
    def _cut_vgg(self,vgg):
        cnn = copy.deepcopy(vgg)
        # normalization module
        #normalization = Normalization(dataset_mean,dataset_std).to(device)

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential()

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in ['conv_4']:
                # add style loss:
                break
        model = model.to(device=device)
        for param in model.parameters():
            param.require_grad = False
        return model

    def forward(self,
        discriminator_output_of_generated_image_input,
        real_image,
        generated_image,
        grey_image,
        ):
        loss = self.w_adv*self._adv_loss(discriminator_output_of_generated_image_input)
        loss += self.w_con*self._con_loss(real_image,generated_image)
        loss += self.w_gra*self._gra_loss(generated_image,grey_image)
        loss += self.w_col*self._col_loss(real_image,generated_image)
        loss += self.w_tv*self._tv_loss(generated_image)

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
        output_fvgg_real_in = self.feature_extractor(real_img)
        output_fvgg_generated_in = self.feature_extractor(generated_image)
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
    g_loss = GeneratorLoss()
    print(g_loss(d_img,img,generated_img,grey_img))

if __name__ == "__main__":
    test()