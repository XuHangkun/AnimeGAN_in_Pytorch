import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    Descriminator network
    """

    def __init__(self,nc=3,b_ch=32,ngpu=1):
        """
        Args:
            nc    (int) : Number of channels in the training images, default =3
            n_dis (int) : the number of discriminator layer, defualt = 3
            b_ch  (int) : base channel number per layer, defualt = 64
            n_gpu (int) : number of GPU, default = 1
        """
        super(Discriminator,self).__init__()
        self.ngpu = 1
        self.nc = nc
        self.b_ch = b_ch//2

        self.main = nn.Sequential(
            # input channel is nc
            nn.Conv2d(self.nc,self.b_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            # input channel b_ch, output channel 4 * b_ch
            nn.Conv2d(self.b_ch,self.b_ch*2,kernel_size=3,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.b_ch*2,self.b_ch*4,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(self.b_ch * 4),
            nn.LeakyReLU(0.2,inplace=True),

            # input channel b_ch, output channel 4 * b_ch
            nn.Conv2d(self.b_ch * 4,self.b_ch * 4,kernel_size=3,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.b_ch * 4,self.b_ch*8,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(self.b_ch * 8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(self.b_ch * 8,self.b_ch * 8,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(self.b_ch*8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.b_ch * 8,1,kernel_size=3,stride=1,padding=1,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(x)

def test():
    """
    Test Discriminator
    """
    netD = Discriminator()
    img = torch.rand([4,3,512,512])
    res = netD(img)
    print(res.shape)

if __name__ == "__main__":
    test()
