# -*- coding: utf-8 -*-
"""
    Dataset class for image
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

import os
import cv2,random
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):

    def __init__(self,image_dir,transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        color_img,grey_img = self.read_image(self.paths[idx])

        sample = {"color_img":color_img,"grey_img":grey_img}

        if self.transform:
            sample["color_img"] = self.transform(sample["color_img"])
            sample["grey_img"] = self.transform(sample["grey_img"])
        return sample


    def get_image_paths_train(self, image_dir):

        image_dir = os.path.join(image_dir)

        paths = []

        for path in os.listdir(image_dir):
            # Check extensions of filename
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            # Construct complete path to anime image
            path_full = os.path.join(image_dir, path)

            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue

            paths.append(path_full)

        return paths

    def read_image(self, img_path1):

        if 'style' in img_path1.split('/') or 'smooth' in img_path1.split('/'):
            # color image1
            image1 = Image.open(img_path1)

            # gray image2
            image2 = np.asarray(image1.convert("L"))
            image2 = image2[:,:,None]
            image2 = np.concatenate((image2,image2,image2),2)
            image2 = Image.fromarray(np.uint8(image2))

        else:
            # color image1
            image1 = Image.open(img_path1)

            # gray image2
            image2 = np.asarray(image1.convert("L"))
            image2 = image2[:,:,None]
            image2 = np.concatenate((image2,image2,image2),2)
            image2 = Image.fromarray(np.uint8(image2))

        return image1, image2

def test():
    """
    Test ImageDataset
    """
    loader = transforms.Compose([
            transforms.Resize((256,int(256*1.7777))),
            transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    unloader = transforms.Compose([
                transforms.ToPILImage()
            ])
    img_dir = os.path.join(os.getcwd(),"dataset/FHFE/style")
    print(img_dir)
    dataset = ImageDataset(img_dir,transform=loader)
    print("Image Numbers :",len(dataset))
    sample = dataset[1]
    print(sample["color_img"].shape,sample["grey_img"].shape)

    #test in dataloader
    dataloader = DataLoader(dataset, batch_size=4,shuffle=True,num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch >= 1:
            break
        print(i_batch, sample_batched['color_img'].size(),sample_batched['grey_img'].size())
        print(sample_batched['color_img'][0,0,:,:])
        print(sample_batched['grey_img'][0,0,:,:])
    color_img = sample_batched["grey_img"][0]
    color_img = color_img.squeeze(0)
    color_img = (color_img + 1 )/2
    color_img = unloader(color_img)
    color_img.show()

if __name__ == "__main__":
    test()
