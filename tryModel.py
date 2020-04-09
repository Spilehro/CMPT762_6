import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
TOTAL_CLASSES = 100


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        
        # <<TODO#3>> Add more conv layers with increasing 
        # output channels
        # <<TODO#4>> Add normalization layers after conv
        # layers (nn.BatchNorm2d)

        # Also experiment with kernel size in conv2d layers (say 3
        # inspired from VGGNet)
        # To keep it simple, keep the same kernel size
        # (right now set to 5) in all conv layers.
        # Do not have a maxpool layer after every conv layer in your
        # deeper network as it leads to too much loss of information.

        self.conv1 = nn.Conv2d(3, 6, 5,padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.poolb = nn.MaxPool2d(2,2)
        self.pooln = nn.MaxPool2d(1,1)

        self.conv2 = nn.Conv2d(6, 16,5,padding=2 )
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 5,padding=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 5,padding=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 5,padding=2)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 256, 5,padding=2)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 5,padding=2)
        self.bn7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512,1024,5,padding=2)
        self.bn8 = nn.BatchNorm2d(1024)
        

        # <<TODO#3>> Add more linear (fc) layers
        # <<TODO#4>> Add normalization layers after linear and
        # experiment inserting them before or after ReLU (nn.BatchNorm1d)
        # More on nn.sequential:
        # http://pytorch.org/docs/master/nn.html#torch.nn.Sequential
        
        self.fc_net = nn.Sequential(
            nn.Linear( 1024* 4 * 4, 512),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(TOTAL_CLASSES//4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(TOTAL_CLASSES//2),
            # nn.Linear(TOTAL_CLASSES//4, TOTAL_CLASSES//2),
            # nn.ReLU(inplace= True),
            nn.Linear(128, TOTAL_CLASSES),
            nn.ReLU(inplace= True),
            # nn.BatchNorm1d(TOTAL_CLASSES)

        )

    def forward(self, x):

        # <<TODO#3&#4>> Based on the above edits, you'll have
        # to edit the forward pass description here.

        x = F.relu(self.bn1(self.conv1(x)))
        # Output size = 28//2 x 28//2 = 14 x 14

        x = F.relu(self.bn2(self.conv2(x)))
        # Output size = 10//2 x 10//2 = 5 x 5
        x=self.poolb(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x = self.pooln(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.poolb(x)

        x = F.relu(self.bn7(self.conv7(x)))
        # x = self.pooln(x)

        x=F.relu(self.bn8(self.conv8(x)))
        x= self.poolb(x)

        channels = x.shape[1]
        w_h = x.shape[2]

        # See the CS231 link to understand why this is 16*5*5!
        # This will help you design your own deeper network
        x = x.view(-1, channels * w_h * w_h)
        x = self.fc_net(x)

        # No softmax is needed as the loss function in step 3
        # takes care of that
        
        return x

