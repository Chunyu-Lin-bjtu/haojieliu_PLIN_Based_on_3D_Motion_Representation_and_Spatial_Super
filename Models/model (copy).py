"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from .ERFNet import Net
from .subnet import refineNet 

# from submodels imporst *

class uncertainty_net(nn.Module):
    def __init__(self, in_channels, out_channels=1, thres=15):
        super(uncertainty_net, self).__init__()
        out_chan = 2

        combine = 'concat'
        self.combine = combine
        self.in_channels = in_channels

        out_channels = 2
        self.depthnet = Net(in_channels=in_channels, out_channels=out_channels)

        local_channels_in = 4 if self.combine == 'concat' else 1
        self.convbnrelu = nn.Sequential(convbn(local_channels_in, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))
        # self.convbnrelu_1 = nn.Sequential(convbn(1, 8, 3, 1, 1, 1),
        #                 # nn.BatchNorm2d(8),
        #                 nn.ReLU(inplace=True))
        # self.convbnrelu_2 = nn.Sequential(convbn(1, 8, 3, 1, 1, 1),
        #                 # nn.BatchNorm2d(8),
        #                 nn.ReLU(inplace=True))
        # self.convbnrelu_3 = nn.Sequential(convbn(1, 8, 3, 1, 1, 1),
        #                 # nn.BatchNorm2d(8),
        #                 nn.ReLU(inplace=True))
        # self.convbnrelu_4 = nn.Sequential(convbn(1, 8, 3, 1, 1, 1),
        #                 # nn.BatchNorm2d(8),
        #                 nn.ReLU(inplace=True))
        # self.convbnrelu = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
        #                 nn.ReLU(inplace=True))

        self.hourglass1 = hourglass_1(32)
        self.hourglass2 = hourglass_2(32)
        self.hourglass3 = hourglass_3(32)
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, out_chan, kernel_size=3, padding=1, stride=1, bias=True))
        self.activation = nn.ReLU(inplace=True)
        self.thres = thres
        self.softmax = torch.nn.Softmax(dim=1)

        self.fuse_two_brand = nn.Sequential(convbn(4, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   # convbn(32, 32, 3, 1, 1, 1),
                                   # nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=True))


        pretrainederf = True
        if pretrainederf == True:
            target_state = self.depthnet.state_dict()
            check = torch.load('erfnet_pretrained.pth')
            for name, val in check.items():
                mono_name = name[7:]
                if mono_name not in target_state:
                    # print('mono_name not in target_state',mono_name)
                    continue
                try:
                    target_state[mono_name].copy_(val)
                    # print('target_state[]',mono_name)
                except RuntimeError:
                    continue
            print('successfully loaded pretrained model')
        # self.local_net = refineNet()

    def forward(self, input, depth_flow,epoch=50):
        if self.in_channels > 1:
            rgb_in = input[:, 2:, :, :]
            lidar_in = input[:, 0:2, :, :]
        else:
            lidar_in = input
        # lidar_in = input
        # 1. GLOBAL NET
        embedding0, embedding1, embedding2 = self.depthnet(input)

        global_features = embedding0[:, 0:1, :, :]
        precise_depth = embedding0[:, 1:2, :, :]
        # conf = embedding0[:, 2:, :, :]

        # 2. Fuse 
        if self.combine == 'concat':
            # input = torch.cat((lidar_in, global_features,depth_flow,depth_t), 1)
            # '''
            # 12.11 featrue fuse
            # '''
            # # print('lidar_in:',lidar_in.shape)
            # # print('lidar_in:',input[:, 0:1, :, :].shape)
            # lidar_in_1 = self.convbnrelu_1(input[:, 0:1, :, :])
            # lidar_in_2 = self.convbnrelu_2(input[:, 1:2, :, :])
            # global_features = self.convbnrelu_3(global_features)
            # depth_t = self.convbnrelu_4(depth_t)
            # input = torch.cat((lidar_in_1,lidar_in_2, global_features,depth_t), 1)

            input_ = torch.cat((lidar_in, global_features,depth_flow), 1)
        elif self.combine == 'add':
            input = lidar_in + global_features
        elif self.combine == 'mul':
            input = lidar_in * global_features
        elif self.combine == 'sigmoid':
            input = lidar_in * nn.Sigmoid()(global_features)
        else:
            input = lidar_in


        # # 3. LOCAL NET
        # out = self.convbnrelu(input_)
        # out1, embedding3, embedding4 = self.hourglass1(out, embedding1, embedding2)
        # out1 = out1 + out
        # out2 = self.hourglass3(out1, embedding3, embedding4)
        # out2 = out2 + out

        # # #12.19
        # # out3 = self.hourglass2(out2, embedding3, embedding4)
        # # out2 = out2 + out

        # out = self.fuse(out2)
        # lidar_out = out

        # # out = self.local_net(input_, embedding1, embedding2)
        # # lidar_out = out

        # # 4. Late Fusion
        # lidar_to_depth, lidar_to_conf = torch.chunk(out, 2, dim=1)
        # lidar_to_conf, conf = torch.chunk(self.softmax(torch.cat((lidar_to_conf, conf), 1)), 2, dim=1)
        # out = conf * precise_depth + lidar_to_conf * lidar_to_depth
        # # return out, lidar_out, precise_depth, global_features
        # return out, lidar_to_depth, precise_depth, global_features



        # 3. LOCAL NET
        out = self.convbnrelu(input_)
        out1, embedding3, embedding4 = self.hourglass1(out, embedding1, embedding2)
        out1 = out1 + out
        out2, embedding5, embedding6 = self.hourglass2(out1, embedding3, embedding4)
        out2 = out2 + out

        # #12.19
        out3 = self.hourglass3(out2, embedding5, embedding6)
        out3 = out3 + out

        out = self.fuse(out3)
        lidar_out = out

        # out = self.local_net(input_, embedding1, embedding2)
        # lidar_out = out


        # 4. Late Fusion
        lidar_to_depth, lidar_to_conf = torch.chunk(out, 2, dim=1)
        # lidar_to_conf, conf = torch.chunk(self.softmax(torch.cat((lidar_to_conf, conf), 1)), 2, dim=1)
        # out = conf * precise_depth + lidar_to_conf * lidar_to_depth

        out = self.fuse_two_brand(torch.cat((lidar_out, embedding0[:, 0:, :, :]), 1))

        # return out, lidar_out, precise_depth, global_features
        return out, lidar_to_depth, precise_depth, global_features


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
                         # nn.BatchNorm2d(out_planes))


class hourglass_1(nn.Module):
    def __init__(self, channels_in):
        super(hourglass_1, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))
        # self.cam1 = contex_aggregation_layer(32)
        # self.cam1_2 = contex_aggregation_layer(64)
        # # self.cam1_3 = contex_aggregation_layer(64)

    def forward(self, x, em1, em2):

        # x = self.cam1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, em1), 1)
        # x = self.cam1_2(x)
        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = F.relu(x_prime, inplace=True)
        x_prime = torch.cat((x_prime, em2), 1)
        # x_prime = self.cam1_3(x_prime)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out, x, x_prime


class hourglass_2(nn.Module):
    def __init__(self, channels_in):
        super(hourglass_2, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*4, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))
        # self.cam2 = contex_aggregation_layer(32)

    def forward(self, x, em1, em2):
        # x = self.cam2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + em1
        x = F.relu(x, inplace=True)
        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = x_prime + em2
        x_prime = F.relu(x_prime, inplace=True)

        out = self.conv5(x_prime)
        out = self.conv6(out)
        return out ,x,x_prime

class hourglass_3(nn.Module):
    def __init__(self, channels_in):
        super(hourglass_3, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*4, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))
        # self.cam2 = contex_aggregation_layer(32)

    def forward(self, x, em1, em2):
        # x = self.cam2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + em1
        x = F.relu(x, inplace=True)
        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = x_prime + em2
        x_prime = F.relu(x_prime, inplace=True)

        out = self.conv5(x_prime)
        out = self.conv6(out)
        return out 


# def conv_bn_relu1(in_channels, out_channels, kernel_size, \
#         stride=1, padding=0, bn=False, relu=True):
#     bias = not bn
#     layers = []
#     layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
#         padding, bias=bias))
#     if bn:
#         layers.append(nn.BatchNorm2d(out_channels))
#     if relu:
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
#     layers = nn.Sequential(*layers)

#     # initialize the weights
#     for m in layers.modules():
#         init_weights(m)

#     return layers
# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         m.weight.data.normal_(0, 1e-3)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.ConvTranspose2d):
#         m.weight.data.normal_(0, 1e-3)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
       
# class contex_aggregation_layer(nn.Module):
#     def __init__(self, ninput):
#         super().__init__()

#         # self.conv = nn.Conv2d(ninput, noutput-ninput, (1, 1), stride=1, padding=1, bias=True)
#         self.pool = nn.MaxPool2d(7, stride=1,padding=3)
#         # self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
#         self.conv = conv_bn_relu1(ninput, ninput // 8 , kernel_size=1, stride=1, padding=0)
#         self.attention=conv_bn_relu1(in_channels=ninput // 8, out_channels=ninput, kernel_size=1, stride=1, padding=0, bn=False, relu=False)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, input):
#         maxpool = self.pool(input)
#         # print('maxpool',maxpool.shape)
#         conv1x1 = self.conv(maxpool)
#         # print('conv1x1',conv1x1.shape)
#         mask_feature = self.attention(conv1x1)
#         # print('mask_feature',mask_feature.shape)
#         mask = self.sigmoid(mask_feature)
#         output = input * mask
#         return output




if __name__ == '__main__':
    batch_size = 4
    in_channels = 4
    H, W = 256, 1216
    model = uncertainty_net(34, in_channels).cuda()
    print(model)
    print("Number of parameters in model is {:.3f}M".format(sum(tensor.numel() for tensor in model.parameters())/1e6))
    input = torch.rand((batch_size, in_channels, H, W)).cuda().float()
    out = model(input)
    print(out.shape)
