# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        chans = 32 if in_channels > 16 else 16
        self.initial_block = DownsamplerBlock(in_channels, chans)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(chans, 64))

        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = UpsamplerBlock(128, 64)
        self.layer2 = non_bottleneck_1d(64, 0, 1)
        self.layer3 = non_bottleneck_1d(64, 0, 1) # 64x64x304

        self.layer4 = UpsamplerBlock(64, 32)
        self.layer5 = non_bottleneck_1d(32, 0, 1)
        self.layer6 = non_bottleneck_1d(32, 0, 1) # 32x128x608

        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        em2 = output
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        em1 = output

        output = self.output_conv(output)

        return output, em1, em2


class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):  #use encoder to pass pretrained encoder
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels)
        print('erfnet--------------------')

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
            return self.decoder.forward(output)

'''



# liuhaojie resnet43-unet 
# 2019 12.23


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import numpy as np

import importlib
import sys
from torchvision.utils import make_grid



def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )  
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)

        # x = self.conv2(x)
        # x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
      
        x += residual
        x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):  #use encoder to pass pretrained encoder
        super().__init__()
        # F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0
        # self.d1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=1)
        # self.d3 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=1)
        # self.edge = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=1)
        # self.rgb = conv_bn_relu(3, 8, kernel_size=3, stride=1, padding=1)
      
          
        self.conv1= conv_bn_relu(in_channels, 64, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet34'](pretrained=True)
        # # pretrained_model = resnet.__dict__['resnet18'](pretrained=args.pretrained)
        # if not args.pretrained:
        #     pretrained_model.apply(init_weights)

        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model # clear memory

        # define number of intermediate channels
        num_channels = 512
       
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)
        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768, out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64,
            kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=out_channels, kernel_size=1, stride=1, bn=False, relu=False)
        self.conv_em2= conv_bn_relu(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv_em1= conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1)



    def forward(self, in_put): 
        
        # F_0_1 = self.edge(F_0_1)
        # F_1_0 = self.F_1_0(F_1_0)
        # g_I1_F_t_1 = self.g_I1_F_t_1(g_I1_F_t_1)
        # g_I0_F_t_0 = self.g_I0_F_t_0(g_I0_F_t_0)
        # d = self.d1(d1)
        # d3 = self.d3(d3)
        # in_put = torch.cat((d,d3,F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0),1)

        conv1 = self.conv1(in_put)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2) # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3) # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4) # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5) # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        em2 = convt4
        em2 = self.conv_em2(em2)

        y = torch.cat((convt4, conv4), 1)


        convt3 = self.convt3(y)
        em1 = convt3
        em1 = self.conv_em1(em1)

        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
       
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1,conv1), 1)

        y = self.convtf(y)    #coarse depth 1,1,240,1216

        # print(em1.shape,em2.shape)

        return  y, em1, em2
