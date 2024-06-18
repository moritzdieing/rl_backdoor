# new AE part move later

import torch.nn as nn
import torch.nn.functional as F


class CNNAutoEncoderL10(nn.Module):

    def __init__(self):

        super(CNNAutoEncoderL10,self).__init__()

        self.eL1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.eL2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.eL3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.eL4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.eL5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.eL6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.eL7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.eL8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.eL9 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.eL10 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.activation = nn.ReLU()

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices = True)

        self.dL1 = nn.ConvTranspose2d(1, 16, kernel_size=3, padding=1)
        self.dL2 = nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1)
        self.dL3 =nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1)
        self.dL4 =nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1)
        self.dL5 =nn.ConvTranspose2d(128, 256, kernel_size=3, padding=1)
        self.dL6 =nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.dL7 =nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.dL8 =nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.dL9 =nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.dL10 =nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)

        self.maxUnpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.tan = nn.Tanh()

    def encode(self, x):
        x = self.activation(self.eL1(x))
        x = self.activation(self.eL2(x))
        x = self.activation(self.eL3(x))
        x,i2 = self.maxPool(x)
        x = self.activation(self.eL4(x))
        x = self.activation(self.eL5(x))
        x = self.activation(self.eL6(x))
        x,i6 = self.maxPool(x)
        x = self.activation(self.eL7(x))
        x = self.activation(self.eL8(x))
        x = self.activation(self.eL9(x))
        x = self.activation(self.eL10(x))
        return x, (i2, i6)# encoded version

    def decode(self, x, inds):
        x = self.activation(self.dL1(x))
        x = self.activation(self.dL2(x))
        x = self.activation(self.dL3(x))
        x = self.activation(self.dL4(x))
        x = self.maxUnpool(x,inds[1])
        x = self.activation(self.dL5(x))
        x = self.activation(self.dL6(x))
        x = self.activation(self.dL7(x))
        x = self.maxUnpool(x,inds[0])
        x = self.activation(self.dL8(x))
        x = self.activation(self.dL9(x))
        x = self.dL10(x)
        x = self.tan(x)

        return x

    def forward(self, x):
        x, inds = self.encode(x)
        x = self.decode(x, inds)
        return x