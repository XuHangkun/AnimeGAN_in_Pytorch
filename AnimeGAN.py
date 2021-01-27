# -*- coding: utf-8 -*-
"""
    AnimeGAN
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

from tools.data_loader import ImageDataset
from torch.utils.data import Dataset,DataLoader
from net.discriminator_cartoon import Discriminator
from net.generator_cartoon import Generator
from loss.discriminatorloss_cartoon import DiscriminatorLoss
from loss.generatorloss_cartoon import GeneratorLoss
from config.config import device
import torch.optim as optim
import torch
import time
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tools.init_net import weights_init
import torchvision.models as models

class AnimeGAN(object) :
    """
    AnimeGAN
    """
    def __init__(self, args):
        """
        AnimeGAN Initialization
        """
        self.model_name = 'AnimeGANv2'
        self.phase = args.phase
        self.contain_init_phase = args.contain_init_phase       # for training
        self.init_epoch = args.init_epoch       # for training
        self.load_model_from_file = args.load_model
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.style = args.style

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq

        self.init_lr = args.init_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.g_adv_weight = args.g_adv_weight
        self.d_adv_weight = args.d_adv_weight
        self.con_weight = args.con_weight
        self.sty_weight = args.sty_weight
        self.color_weight = args.color_weight
        self.tv_weight = args.tv_weight

        self.print_every = 10

        self.img_loader = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        self.img_unloader = transforms.Compose([
                transforms.ToPILImage()
            ])
        self.realimg_dataset = ImageDataset("./dataset/train_photo",transform=self.img_loader)
        self.anime_dataset = ImageDataset('./dataset/{}'.format(self.style + '/style'),transform=self.img_loader)
        self.anime_smooth_dataset = ImageDataset('./dataset/{}'.format(self.style + '/smooth'),transform=self.img_loader)
        self.realimg_dataloader = DataLoader(self.realimg_dataset, batch_size=self.batch_size,shuffle=True,num_workers=4)
        self.anime_dataloader = DataLoader(self.anime_dataset, batch_size=self.batch_size,shuffle=True,num_workers=4)
        self.anime_smooth_dataloader = DataLoader(self.anime_smooth_dataset, batch_size=self.batch_size,shuffle=True,num_workers=4)

        #load vgg model
        self.generator, self.discriminator = self.load_model(self.load_model_from_file)

        #loss
        self.d_loss = DiscriminatorLoss().to(device)
        self.g_loss = GeneratorLoss().to(device)

        #optim
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),self.d_lr, [self.beta1, self.beta2])
        self.g_optimizer = optim.Adam(self.generator.parameters(),self.g_lr, [self.beta1, self.beta2])
        self.G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.g_optimizer, milestones=[self.epoch // 2, self.epoch // 4 * 3], gamma=0.1)
        self.D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.d_optimizer, milestones=[self.epoch // 2, self.epoch // 4 * 3], gamma=0.1)

    def train(self):
        """
        Training the generator and discriminator
        """
        start_time = time.time()
        train_info = {"epoch":[],"batch":[],"g_loss":[],"d_loss":[]}
        log_path = os.path.join(self.log_dir,self.style+'_train_log.npy')
        min_loss = [1.e10,1.e10]
        for epoch in range(self.epoch):
            g_losses = []
            d_losses = []
            self.generator.train()
            ## if we are in pretrain epoch
            if (self.contain_init_phase and (epoch < self.init_epoch)):
                print("Init train epoch......")
            self.g_loss.is_init_phase = (self.contain_init_phase and (epoch < self.init_epoch))

            for index, (photo_images,smoothed_cartoon_images,cartoon_images) in enumerate(zip(self.realimg_dataloader,self.anime_smooth_dataloader, self.anime_dataloader)):
                batch_size = photo_images["color_img"].size(0)
                photo_images = photo_images["color_img"].to(device)
                cartoon_grey_images = cartoon_images["grey_img"].to(device)
                cartoon_smooth_grey_images = smoothed_cartoon_images["grey_img"].to(device)
                smoothed_cartoon_images = smoothed_cartoon_images["color_img"].to(device)
                cartoon_images = cartoon_images["color_img"].to(device)

                #train the discriminator
                if not (self.contain_init_phase and (epoch < self.init_epoch)):
                    self.d_optimizer.zero_grad()
                    discriminator_output_of_cartoon_input = self.discriminator(cartoon_images)
                    discriminator_output_of_cartoon_smooth_input = self.discriminator(smoothed_cartoon_images)
                    discriminator_output_of_generated_cartoon_input = self.discriminator(self.generator(cartoon_images))
                    discriminator_output_of_cartoon_grey_input = self.discriminator(cartoon_grey_images)
                    discriminator_output_of_cartoon_smooth_grey_input = self.discriminator(cartoon_smooth_grey_images)
                    dloss = self.d_loss(
                        discriminator_output_of_cartoon_input,
                        #discriminator_output_of_cartoon_grey_input,
                        discriminator_output_of_cartoon_smooth_input,
                        discriminator_output_of_generated_cartoon_input
                    )
                    dloss.backward()
                    self.d_optimizer.step()
                else:
                    dloss = torch.Tensor([0.0])
                d_losses.append(dloss.item())


                #train the generator
                self.g_optimizer.zero_grad()
                generated_cartoon = self.generator(photo_images)
                discriminator_output_of_generated_cartoon_input = self.discriminator(generated_cartoon)
                gloss = self.g_loss(
                    discriminator_output_of_generated_cartoon_input,
                    photo_images,
                    generated_cartoon
                )
                gloss.backward()
                self.g_optimizer.step()
                g_losses.append(gloss.item())

            # tune learning rate
            self.G_scheduler.step()
            self.D_scheduler.step()

            # Print the train info here
            now = time.time()
            current_run_time = now - start_time
            start_time = now
            print("Epoch {} | d_loss {:6.4f} | g_loss {:6.4f} | time {:2.0f}s".format(epoch+1,np.sum(d_losses)/len(d_losses),np.sum(g_losses)/len(g_losses),current_run_time))
            train_info["epoch"].append(epoch+1)
            train_info["batch"].append(index+1)
            train_info["g_loss"].append(gloss.item())
            train_info["d_loss"].append(dloss.item())

            # do some test here
            self.test(epoch=epoch)

            # write csv here
            np.save(log_path,train_info)

            #save model every epoch
            self.save_discriminator()
            self.save_generator()

    def save_generator(self):
        """
        save the model to checkpoint_dir
        """
        dir = os.path.join(self.checkpoint_dir,self.style)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        g_path = os.path.join(dir,self.style+'_generator.pth')
        torch.save(self.generator,g_path)

    def save_discriminator(self):
        """
        save the model to checkpoint_dir
        """
        dir = os.path.join(self.checkpoint_dir,self.style)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        d_path = os.path.join(dir,self.style+'_discriminator.pth')
        torch.save(self.discriminator,d_path)

    def load_model(self,new = False):
        """
        load the model
        """
        dir = os.path.join(self.checkpoint_dir,self.style)
        g_path = os.path.join(dir,self.style+'_generator.pth')
        d_path = os.path.join(dir,self.style+'_discriminator.pth')
        pth_exists = os.path.exists(g_path) and os.path.exists(d_path)
        if new and pth_exists:
            discriminator = torch.load(d_path).to(device)
            print("Load model from ",d_path)
            generator = torch.load(g_path).to(device)
            print("Load model from ",g_path)
        else:
            discriminator = Discriminator().to(device)
            #discriminator.apply(weights_init)
            print("Create discriminator from class!")
            generator = Generator().to(device)
            #generator.apply(weights_init)
            print("Create generator from class!")

        if self.phase == "train":
            generator.train()
            discriminator.train()
        else:
            generator.eval()
            discriminator.eval()

        return (generator,discriminator)

    def transfer(self,img):
        """
        Args:
            img (PIL image)

        transfer RGB real img to cartoon img with same size
        """
        img = self.img_loader(img)
        img = img.to(device,torch.float)
        img = img.unsqueeze(0)
        #print(img.shape)

        g_img = self.generator(img)

        g_img = g_img.cpu().clone()
        g_img = g_img.squeeze(0)
        g_img = (g_img + 1) / 2
        g_img = self.img_unloader(g_img)

        return g_img

    def test(self,n=5,epoch=None):
        """
        Test image in test
        """
        self.generator.eval()
        test_dir = os.path.join(os.getcwd(),"dataset/test/test_photo")
        res_dir = os.path.join(os.getcwd(),self.result_dir,self.style)
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir)
        print("Test AnimeGAN model")
        print("Read  imgs dir : ",test_dir)
        print("Gene. imgs dir : ",res_dir)
        count=0
        for path in os.listdir(test_dir):
            # Check extensions of filename
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue
            path_full = os.path.join(test_dir, path)

            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue

            img = Image.open(path_full)
            g_img = self.transfer(img)
            if epoch:
                save_name = self.style + "_epoch%d"%(epoch)+"_%d.png"%(count+1)
            else:
                save_name = self.style + "_%d.png"%(count+1)
            g_img.save(os.path.join(res_dir,save_name))
            count += 1
            if count > n and self.phase == "train":
                # in the test model, we'd better to transfer all image
                break
        print("Test finished!")

def test():
    pass

if __name__ == "__main__":
    test()
