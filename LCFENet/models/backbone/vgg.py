import torch
import torch.nn as nn
import os

class VGG(nn.Module):
    # pooling layer at the front of block
    def __init__(self, in_channels=[64,128,256,512,512],mode = 'rgb'):
        super(VGG, self).__init__()

        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, in_channels[0], 7, 1, 3))
        conv1.add_module('bn1_1', nn.BatchNorm2d(in_channels[0]))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        # conv1.add_module('conv1_2', nn.Conv2d(in_channels[0], in_channels[0], 3, 1, 1))
        # conv1.add_module('bn1_2', nn.BatchNorm2d(in_channels[0]))
        # conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1
        
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(in_channels[0], in_channels[1], 3, 1, 1))
        conv2.add_module('bn2_1', nn.BatchNorm2d(in_channels[1]))
        conv2.add_module('relu2_1', nn.ReLU())
        # conv2.add_module('conv2_2', nn.Conv2d(in_channels[1], in_channels[1], 3, 1, 1))
        # conv2.add_module('bn2_2', nn.BatchNorm2d(in_channels[1]))
        # conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(in_channels[1], in_channels[2], 3, 1, 1))
        conv3.add_module('bn3_1', nn.BatchNorm2d(in_channels[2]))
        conv3.add_module('relu3_1', nn.ReLU())
        # conv3.add_module('conv3_2', nn.Conv2d(in_channels[2], in_channels[2], 3, 1, 1))
        # conv3.add_module('bn3_2', nn.BatchNorm2d(in_channels[2]))
        # conv3.add_module('relu3_2', nn.ReLU())
        # conv3.add_module('conv3_3', nn.Conv2d(in_channels[2], in_channels[2], 3, 1, 1))
        # conv3.add_module('bn3_3', nn.BatchNorm2d(in_channels[2]))
        # conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3_1', nn.MaxPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(in_channels[2], in_channels[3], 3, 1, 1))
        conv4.add_module('bn4_1', nn.BatchNorm2d(in_channels[3]))
        conv4.add_module('relu4_1', nn.ReLU())
        # conv4.add_module('conv4_2', nn.Conv2d(in_channels[3], in_channels[3], 3, 1, 1))
        # conv4.add_module('bn4_2', nn.BatchNorm2d(in_channels[3]))
        # conv4.add_module('relu4_2', nn.ReLU())
        # conv4.add_module('conv4_3', nn.Conv2d(in_channels[3], in_channels[3], 3, 1, 1))
        # conv4.add_module('bn4_3', nn.BatchNorm2d(in_channels[3]))
        # conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.MaxPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(in_channels[3], in_channels[4], 3, 1, 1))
        conv5.add_module('bn5_1', nn.BatchNorm2d(in_channels[4]))
        conv5.add_module('relu5_1', nn.ReLU())
        # conv5.add_module('conv5_2', nn.Conv2d(in_channels[4], in_channels[4], 3, 1, 1))
        # conv5.add_module('bn5_2', nn.BatchNorm2d(in_channels[4]))
        # conv5.add_module('relu5_2', nn.ReLU())
        # conv5.add_module('conv5_3', nn.Conv2d(in_channels[4], in_channels[4], 3, 1, 1))
        # conv5.add_module('bn5_2', nn.BatchNorm2d(in_channels[4]))
        # conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5

        conv6 = nn.Sequential()
        conv6.add_module('pool5', nn.MaxPool2d(2, stride=2))
        conv6.add_module('conv6_1', nn.Conv2d(in_channels[4], in_channels[5], 3, 1, 1))
        conv6.add_module('bn6_1', nn.BatchNorm2d(in_channels[5]))
        conv6.add_module('relu6_1', nn.ReLU())
        # conv5.add_module('conv6_2', nn.Conv2d(in_channels[4], in_channels[4], 3, 1, 1))
        # conv5.add_module('bn6_2', nn.BatchNorm2d(in_channels[4]))
        # conv5.add_module('relu6_2', nn.ReLU())
        # conv5.add_module('conv5_3', nn.Conv2d(in_channels[4], in_channels[4], 3, 1, 1))
        # conv5.add_module('bn5_2', nn.BatchNorm2d(in_channels[4]))
        # conv5.add_module('relu5_3', nn.ReLU())
        self.conv6 = conv6

        # pre_train = torch.load(os.path.join(os.getcwd(),'vgg16-397923af.pth'),map_location='cpu')
        # self._initialize_weights(pre_train)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x=[x1,x2,x3,x4,x5,x6]
        return x

    def _initialize_weights(self, pre_train):
        keys = pre_train.keys()
        
        self.conv1.conv1_1.weight.data.copy_(pre_train[list(keys)[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[list(keys)[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[list(keys)[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[list(keys)[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[list(keys)[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[list(keys)[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[list(keys)[12]])
        self.conv4.conv4_1.weight.data.copy_(pre_train[list(keys)[14]])
        self.conv4.conv4_2.weight.data.copy_(pre_train[list(keys)[16]])
        self.conv4.conv4_3.weight.data.copy_(pre_train[list(keys)[18]])
        self.conv5.conv5_1.weight.data.copy_(pre_train[list(keys)[20]])
        self.conv5.conv5_2.weight.data.copy_(pre_train[list(keys)[22]])
        self.conv5.conv5_3.weight.data.copy_(pre_train[list(keys)[24]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[list(keys)[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[list(keys)[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[list(keys)[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[list(keys)[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[list(keys)[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[list(keys)[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[list(keys)[13]])
        self.conv4.conv4_1.bias.data.copy_(pre_train[list(keys)[15]])
        self.conv4.conv4_2.bias.data.copy_(pre_train[list(keys)[17]])
        self.conv4.conv4_3.bias.data.copy_(pre_train[list(keys)[19]])
        self.conv5.conv5_1.bias.data.copy_(pre_train[list(keys)[21]])
        self.conv5.conv5_2.bias.data.copy_(pre_train[list(keys)[23]])
        self.conv5.conv5_3.bias.data.copy_(pre_train[list(keys)[25]])
        
        self.conv5=self.conv5.cuda()
        self.conv4=self.conv4.cuda()
        self.conv3=self.conv3.cuda()
        self.conv2=self.conv2.cuda()
        self.conv1=self.conv1.cuda()




