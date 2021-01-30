import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.init_net import weights_init

class Discriminator(nn.Module):
    """
    Descriminator network
    """
    # initializers
    def __init__(self, in_nc=3, out_nc=3, nf=32):
        super(Discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1)
        )

        #weights_init(self)

    # forward method
    def forward(self, x):
        output = self.convs(x)
        return output

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
