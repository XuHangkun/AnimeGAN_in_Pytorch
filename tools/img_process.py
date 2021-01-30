# -*- coding: utf-8 -*-
"""
    process img data
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

from __future__ import print_function
import torch
from PIL import Image
import torchvision.transforms as transforms
from config.config import device

#turn torch.Tensor to PIL image
unloader = transforms.ToPILImage()  # reconvert into PIL image

def image_loader(image_name,imsize=(1080,1920)):
    """
    open image and transform it to tensor.
    """
    loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def saveimg(tensor,fp,title=None):
    """save tensor to fp as img
    """
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save(fp)

def img_ud_concat(imgs,imsize=(720,1280)):
    """
    resize each image to imsize and concat them up and down
    pars:
        imgs: paths of images
    return:
        PIL image
    """
    bkg = Image.new("RGBA",(imsize[1],len(imgs)*imsize[0]),(0,0,0))
    for index,img in enumerate(imgs):
        #img = Image.open(img)
        img = img.resize((imsize[1],imsize[0]))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        location = (0,index*imsize[0])
        bkg.paste(img,location)
    return bkg
