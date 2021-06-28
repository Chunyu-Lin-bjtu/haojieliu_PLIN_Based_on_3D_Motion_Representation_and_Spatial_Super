import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import numpy as np

import importlib
import sys
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter('logs')
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        # nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # nn.ReLU(inplace=True)
    ) 

def down_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3,stride=2,padding=1),
        nn.ReLU(inplace=True)
    ) 
def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
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

class ResNet_refine(nn.Module):

    def __init__(self,args, block, layers):
        
        super(ResNet_refine, self).__init__()

        self.inplanes = 64
        self.modality = args.input  
        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            # self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
            self.conv1_d = conv_bn_relu(1, channels , kernel_size=3, stride=1, padding=1)
        if 'rgb' in self.modality:
            # channels = 64 * 3 // len(self.modality)
            # self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
            pass
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(3, channels , kernel_size=3, stride=1, padding=1)
            # self.conv1_rgb = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)


        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, 1)
        self.layer2 = self._make_layer(block, 128, 1, stride=2)
        self.layer3 = self._make_layer(block, 256, 1, stride=2)
        self.layer4 = self._make_layer(block, 512, 1, stride=2)

        self.conv6 = conv_bn_relu(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = conv_bn_relu(256, 256, kernel_size=3, stride=2, padding=1)
        # decoding layers
        kernel_size = 3
        stride = 2
        self.deconvt5 = convt_bn_relu(in_channels=512, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.deconvt4 = convt_bn_relu(in_channels=768, out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.deconvt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.deconvt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.deconvt1 = convt_bn_relu(in_channels=128, out_channels=64,
            kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

     
        # self.deconvt4 = convt_bn_relu(in_channels=256, out_channels=128,
        #     kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        # self.deconvt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
        #     kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        # self.deconvt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
        #     kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        # self.deconvt1 = convt_bn_relu(in_channels=128, out_channels=64,
        #     kernel_size=kernel_size, stride=1, padding=1)
        # self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # print(self.inplanes,planes,downsample)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
     
    def forward(self, x1,x2):
        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(x1)
        if 'rgb' in self.modality:
            # conv1_img = self.conv1_img(inputx['rgb'])
            pass
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x2)
            # conv1_rgb = self.conv1_rgb(inputx['rgb'])
        if self.modality=='rgbd' or self.modality=='gd':
            conv1 = torch.cat((conv1_d, conv1_img),1)
            # conv1 = torch.cat((conv1, conv1_rgb),1)    
        else:
            conv1 = conv1_d if (self.modality=='d') else conv1_img
        # #encoder 256
        # x2 = self.layer1(conv1)    # 64
        # x3 = self.layer2(x2)       #128  
        # x4 = self.layer3(x3)       #256
        # # x5 = self.layer4(x4)       #512

        # x5 = self.conv5(x4)     #256   
        # # # decoder
        # # convt5 = self.deconvt5(x6)     #256
        # # y = torch.cat((convt5, x5), 1) #512

        # convt4 = self.deconvt4(x5)      #128  
        # y = torch.cat((convt4, x4), 1) #128+256
       
        # convt3 = self.deconvt3(y)
        # y = torch.cat((convt3, x3), 1)

        # convt2 = self.deconvt2(y)
        # y = torch.cat((convt2, x2), 1)
       
        # convt1 = self.deconvt1(y)
        # y = torch.cat((convt1,conv1), 1)

        #encoder 512
        x2 = self.layer1(conv1)    # 64
        x3 = self.layer2(x2)       #128  
        x4 = self.layer3(x3)       #256
        x5 = self.layer4(x4)       #512

        x6 = self.conv6(x5)        
        # decoder
        convt5 = self.deconvt5(x6)     #256
        y = torch.cat((convt5, x5), 1) #512

        convt4 = self.deconvt4(y)   
        y = torch.cat((convt4, x4), 1)
       
        convt3 = self.deconvt3(y)
        y = torch.cat((convt3, x3), 1)

        convt2 = self.deconvt2(y)
        y = torch.cat((convt2, x2), 1)
       
        convt1 = self.deconvt1(y)
        y = torch.cat((convt1,conv1), 1)

        y = self.convtf(y)    #depth 1,1,240,1216 
        return y



# class UNet(nn.Module):
class refineNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        # # self.modality = args.input
        
        # if 'd' in self.modality:
        #     channels = 64 // len(self.modality)
        #     # self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        #     self.conv1_d = conv_bn_relu(1, channels , kernel_size=3, stride=1, padding=1)
        # if 'rgb' in self.modality:
        #     # channels = 64 * 3 // len(self.modality)
        #     # self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        #     pass
        # elif 'g' in self.modality:
        #     channels = 64 // len(self.modality)
        #     self.conv1_img = conv_bn_relu(3, channels , kernel_size=3, stride=1, padding=1)
        #     # self.conv1_rgb = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)

        self.conv1_d = conv_bn_relu(1, 8 , kernel_size=3, stride=1, padding=1)
        self.conv1_img = conv_bn_relu(3, 8 , kernel_size=3, stride=1, padding=1)
        self.input= conv_bn_relu(16, 64, kernel_size=3, stride=1, padding=1)

        self.dconv_down1 = double_conv(64, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_down5 = double_conv(512, 1024)         

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        # # self.upsample = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)        
        # # nn.functional.interpolate
        self.conv1=conv(1024,512)
        self.conv2=conv(512,256)
        self.conv3=conv(256,128)
        self.conv4=conv(128,64)

        self.dconv_up4 = double_conv(1024, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x1,x2):
        # # first layer
        # if 'd' in self.modality:
            # conv1_d = self.conv1_d(x1)
        # if 'rgb' in self.modality:
        #     # conv1_img = self.conv1_img(inputx['rgb'])
        #     pass
        # elif 'g' in self.modality:
        #     conv1_img = self.conv1_img(x2)

        #     # conv1_rgb = self.conv1_rgb(inputx['rgb'])
        # if self.modality=='rgbd' or self.modality=='gd':
        #     inputx = torch.cat((conv1_d, conv1_img),1)
        #     # conv1 = torch.cat((conv1, conv1_rgb),1)    
        # else:
        #     conv1 = conv1_d if (self.modality=='d') else conv1_img
        conv1_d = self.conv1_d(x1)
        conv1_img = self.conv1_img(x2)
        in_put = torch.cat((conv1_d, conv1_img),1)
        in_put=self.input(in_put)

        conv1 = self.dconv_down1(in_put)
        x = self.maxpool(conv1)
       
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
      
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
      
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
      
        x = self.dconv_down5(x)
       
        x = self.upsample(x)
        x = self.conv1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample(x) 
        x = self.conv2(x)       
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)   
        x = self.conv3(x)     
        x = torch.cat([x, conv2], dim=1)       
        x = self.dconv_up2(x)

        x = self.upsample(x)  
        x = self.conv4(x)      
        x = torch.cat([x, conv1], dim=1)   
        x = self.dconv_up1(x)
        
        
        out = self.conv_last(x)
        # print('out',out.size())
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        # proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        # out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,C,width,height)
        
        # out = self.gamma*out + x
        # return out,attention
        return attention

def getFlowCoeff (device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C00, C01, C10, C11.
    """
    # Convert indices tensor to numpy array
   
    C11 = C00 = -0.25
    C01 = 0.25
    C10 = 0.25
    return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)

class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """
    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """
        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        self.device=device
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """
    
        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


class DepthCompletionNet(nn.Module):
    def __init__(self, args):
        assert (args.layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(DepthCompletionNet, self).__init__()
        # self.modality = args.input
       
        # if 'd' in self.modality:
        #     channels = 64 // len(self.modality)
        #     # self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        #     self.conv1_d = conv_bn_relu(1, channels // 2, kernel_size=3, stride=1, padding=1)
        # if 'rgb' in self.modality:
        #     channels = 64 * 3 // len(self.modality)
        #     self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        # elif 'g' in self.modality:
        #     channels = 64 // len(self.modality)
        #     self.conv1_img = conv_bn_relu(1, channels // 2, kernel_size=3, stride=1, padding=1)
        #     self.conv1_rgb = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        #     self.conv1_flow = conv_bn_relu(2, channels, kernel_size=3, stride=1, padding=1)


        # F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0
        self.d1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=1)
        self.d3 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=1)
        self.F_0_1 = conv_bn_relu(2, 8, kernel_size=3, stride=1, padding=1)
        self.F_1_0 = conv_bn_relu(2, 8, kernel_size=3, stride=1, padding=1)
        self.g_I1_F_t_1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=1)
        self.g_I0_F_t_0 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=1)
            
            #  # F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0
            # self.d1 = conv_bn_relu(1, 16, kernel_size=3, stride=1, padding=1)
            # self.d3 = conv_bn_relu(1, 16, kernel_size=3, stride=1, padding=1)
            # self.F_0_1 = conv_bn_relu(2, 16, kernel_size=3, stride=1, padding=1)
            # self.F_1_0 = conv_bn_relu(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv1= conv_bn_relu(6*8, 64, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        # pretrained_model = resnet.__dict__['resnet18'](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)


        '''
        sys.path.append('liteflownet')
        f = importlib.import_module('run')
        self.flow_net = f.Network()
        checkpoint_dict = torch.load('liteflownet/network-kitti.pytorch')
        self.flow_net.load_state_dict(checkpoint_dict)
        
        # Disable Training for the unguided module
        for p in self.flow_net.parameters():            
            p.requires_grad=False
        '''

        # # self.maxpool = pretrained_model._modules['maxpool']
        # self.after_conv1 = conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1)
        # self.before_convtf = conv_bn_relu(128, 64, kernel_size=3, stride=1, padding=1)

        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
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
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)
        # self.convtf = conv_bn_relu(in_channels=128, out_channels=2, kernel_size=1, stride=1, bn=False, relu=False)



        # self.conv_conf = conv_bn_relu(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        # self.selfatt = Self_Attn(1,None)
        # self.convtf = conv_bn_relu(in_channels=32, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

       

        # self.attention=conv_bn_relu(in_channels=1, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)
        # self.softmax = torch.nn.Softmax(dim=2)

        self.refinenet = refineNet(args)
        # self.Res_refine = ResNet_refine(args,BasicBlock, [1, 1, 1, 1])


    def forward(self, d1,d3,F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0,rgb): 
    # def forward(self, x,flow): 
        # # first layer
        # if 'd' in self.modality:
        #     conv1_d = self.conv1_d(x['d'])
        # if 'rgb' in self.modality:
        #     conv1_img = self.conv1_img(x['rgb'])
        # elif 'g' in self.modality:
        #     conv1_img = self.conv1_img(x['d3'])
        #     # conv1_rgb = self.conv1_rgb(x['rgb'])
        #     conv1_rgb = self.conv1_flow(flow)
        # if self.modality=='rgbd' or self.modality=='gd':
        #     conv1 = torch.cat((conv1_d, conv1_img),1)
        #     conv1 = torch.cat((conv1, conv1_rgb),1)
            
        # else:
        #     conv1 = conv1_d if (self.modality=='d') else conv1_img

        
        F_0_1 = self.F_0_1(F_0_1)
        F_1_0 = self.F_1_0(F_1_0)
        g_I1_F_t_1 = self.g_I1_F_t_1(g_I1_F_t_1)
        g_I0_F_t_0 = self.g_I0_F_t_0(g_I0_F_t_0)
        d = self.d1(d1)
        d3 = self.d3(d3)
        in_put = torch.cat((d,d3,F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0),1)
        # # conv1 = torch.cat((d,d3,F_0_1),1)

        # in_put = torch.cat((d1,d3,F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0),1)
        # in_put = torch.cat((x['d'], x['d3']),1)
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
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1,conv1), 1)
#//////////////////////////////////////

        # y = self.conv_conf(y)     #1,64,240,1216
        # conf_feature = y
        # attention_mask = self.selfatt(conf_feature)
        # out of memory

#//////////////////////////////////////


        # y = self.before_convtf(y)
        y = self.convtf(y)    #coarse depth 1,1,240,1216
        coarse = 100*y

        # refine_unet
        # coarse = y[:, 0:1, :, :]
        # mask = y[:, 1:2, :, :]
        # # print('----y',attention_mask[0,:,100:200,200]) 
        # # mask=self.attention(attention_mask)
        # attention_mask = self.softmax(mask)
        # rgb_mask=attention_mask*x['rgb']
        # # y= self.refinenet(coarse,rgb_mask)  

        # # # y= self.refinenet(y,x['rgb'])
        # # # if self.training:
        # # #     return   100*y , coarse
        # # # else:
        # # #     min_distance = 0.9
        # # #     return F.relu( 100*y - min_distance) + min_distance  , coarse  # the minimum range of Velodyne is around 3 feet ~= 0.9m
        

        y= self.refinenet(coarse,rgb)



        # # refine_resnet9
        # y= self.Res_refine(coarse,rgb_mask)
        # y= self.Res_refine(y,x['rgb'])
        # if curr_step%10==0:
        #     writer.add_image('feature_map', make_grid([coarse[0], attention_mask[0]*255, y[0]],nrow=1,padding=20, normalize=False, scale_each=True, pad_value=1), curr_step)
        # # tb_logger.add_image('channels', make_grid(feature_map[0].detach().cpu().unsqueeze(dim=1), nrow=1, padding=20, normalize=False, pad_value=1), curr_step)
        if self.training:
            return  100*y , coarse
        else:
            min_distance = 0.9
            return F.relu( 100*y - min_distance) + min_distance  , coarse  # the minimum range of Velodyne is around 3 feet ~= 0.9m
