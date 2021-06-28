# from __future__ import print_function
# import torch.utils.data
# import torch.nn.functional as F
# import math
# import torch
# import torch.nn as nn

# def convbn(in_planes, out_planes, kernel_size, stride):

#     return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
#                          nn.BatchNorm2d(out_planes))

# def conv(in_planes, out_planes, kernel_size=3,stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
#         nn.BatchNorm2d(out_planes),
#         nn.ReLU(inplace=True)
#     )

# def deconv(in_planes, out_planes):
#     return nn.Sequential(
#         nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
#         nn.BatchNorm2d(out_planes),
#         nn.ReLU(inplace=True)
#     )

# def predict_normal(in_planes):
#     return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)

# def predict_normal2(in_planes):
#     return nn.Conv2d(in_planes, 3, kernel_size=3, stride=1, padding=1, bias=True)

# def predict_normalE2(in_planes):
#     return nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=True)

# def adaptative_cat3(out_conv, out_deconv, out_depth_up):
#     out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
#     out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
#     return torch.cat((out_conv, out_deconv, out_depth_up), 1)
# def adaptative_cat2(out_conv,out_sparse):
#     out_sparse = out_sparse[:, :, :out_conv.size(2), :out_conv.size(3)]
#     return torch.cat((out_conv, out_sparse), 1)
# def adaptative_cat4(out_conv, out_deconv, out_depth_up,out_sparse):
#     out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
#     out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
#     out_sparse = out_sparse[:, :, :out_conv.size(2), :out_conv.size(3)]
#     return torch.cat((out_conv, out_deconv, out_depth_up, out_sparse), 1)
# def adaptative_cat(out_conv, out_deconv, out_depth_up):
#     out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
#     out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
#     return torch.cat((out_conv, out_deconv, out_depth_up), 1)


# class UpProject(nn.Module):

#     def __init__(self, in_channels, out_channels, batch_size):
#         super(UpProject, self).__init__()
#         self.batch_size = batch_size

#         self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
#         self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
#         self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
#         self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

#         self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
#         self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
#         self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
#         self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

#         self.bn1_1 = nn.BatchNorm2d(out_channels)
#         self.bn1_2 = nn.BatchNorm2d(out_channels)

#         self.relu = nn.ReLU(inplace=True)

#         self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))

#         out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

#         out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

#         out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

#         out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))

#         out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

#         out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

#         out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

#         height = out1_1.size()[2]
#         width = out1_1.size()[3]

#         out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
#             self.batch_size, -1, height, width * 2)
#         out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
#             self.batch_size, -1, height, width * 2)

#         out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
#             self.batch_size, -1, height * 2, width * 2)

#         out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
#             self.batch_size, -1, height, width * 2)
#         out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
#             self.batch_size, -1, height, width * 2)

#         out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
#             self.batch_size, -1, height * 2, width * 2)

#         out1 = self.bn1_1(out1_1234)
#         out1 = self.relu(out1)
#         out1 = self.conv3(out1)
#         out1 = self.bn2(out1)

#         out2 = self.bn1_2(out2_1234)

#         out = out1 + out2
#         out = self.relu(out)

#         return out

# class ResBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride):
#         super(ResBlock, self).__init__()

#         self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride),
#                                    nn.ReLU(inplace=True))

#         self.conv2 = convbn(planes, planes, 3, 1)

#         self.ds = convbn(inplanes, planes, 3, stride)

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         x = self.ds(x)
#         out += x
#         out = self.relu(out)
#         return out


# class depthCompletionNew_block(nn.Module):
#     def __init__(self, bs):
#         super(depthCompletionNew_block, self).__init__()
#         self.bs = bs

#         self.convS = ResBlock(3, 32, 1)
#         self.convS0 = ResBlock(32, 97, 1)
#         self.convS1 = ResBlock(97, 193, 2)
#         self.convS2 = ResBlock(193, 385, 2)
#         self.convS3 = ResBlock(385, 513, 2)
#         self.convS4 = ResBlock(513, 512, 2)

#         self.conv1 = ResBlock(3, 32, 1)
#         self.conv2 = ResBlock(32, 64, 1)
#         self.conv3 = ResBlock(64, 128, 2)
#         self.conv3_1 = ResBlock(128, 128, 1)
#         self.conv4 = ResBlock(128, 256, 2)
#         self.conv4_1 = ResBlock(256, 256, 1)
#         self.conv5 = ResBlock(256, 256, 2)
#         self.conv5_1 = ResBlock(256, 256, 1)
#         self.conv6 = ResBlock(256, 512, 2)
#         self.conv6_1 = ResBlock(512, 512, 1)

#         self.deconv5 = self._make_upproj_layer(UpProject, 512, 256, self.bs)
#         self.deconv4 = self._make_upproj_layer(UpProject, 513, 128, self.bs)
#         self.deconv3 = self._make_upproj_layer(UpProject, 385, 64, self.bs)
#         self.deconv2 = self._make_upproj_layer(UpProject, 193, 32, self.bs)

#         self.predict_normal6 = predict_normal(512)
#         self.predict_normal5 = predict_normal(513)
#         self.predict_normal4 = predict_normal(385)
#         self.predict_normal3 = predict_normal(193)
#         self.predict_normal2 = predict_normalE2(97)

#         self.upsampled_normal6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsampled_normal5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsampled_normal4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsampled_normal3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

#         self.predict_mask = nn.Sequential(
#             nn.Conv2d(97, 1, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Sigmoid()
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_upproj_layer(self,block,in_channels,out_channels,bs):
#         return block(in_channels,out_channels,bs)

#     def forward(self, rgb_in,input):
#         # (input,em1, em2)
#         # inputM = mask
#         # inputS = torch.cat((sparse2, inputM), 1)
#         inputS = input
#         inputS_conv = self.convS(inputS)
#         input1 = inputS_conv
#         inputS_conv0 = self.convS0(input1)
#         inputS_conv1 = self.convS1(inputS_conv0)
#         inputS_conv2 = self.convS2(inputS_conv1)
#         inputS_conv3 = self.convS3(inputS_conv2)
#         inputS_conv4 = self.convS4(inputS_conv3)

#         input2 = rgb_in
#         out_conv2 = self.conv2(self.conv1(input2))
#         out_conv3 = self.conv3_1(self.conv3(out_conv2))
#         out_conv4 = self.conv4_1(self.conv4(out_conv3))
#         out_conv5 = self.conv5_1(self.conv5(out_conv4))
#         out_conv6 = self.conv6_1(self.conv6(out_conv5))+inputS_conv4

#         out6 = self.predict_normal6(out_conv6)
#         normal6_up = self.upsampled_normal6_to_5(out6)
#         out_deconv5 = self.deconv5(out_conv6)

#         concat5 = adaptative_cat(out_conv5, out_deconv5, normal6_up)+inputS_conv3
#         out5 = self.predict_normal5(concat5)
#         normal5_up = self.upsampled_normal5_to_4(out5)
#         out_deconv4 = self.deconv4(concat5)

#         concat4 = adaptative_cat(out_conv4, out_deconv4, normal5_up)+inputS_conv2
#         out4 = self.predict_normal4(concat4)
#         normal4_up = self.upsampled_normal4_to_3(out4)
#         out_deconv3 = self.deconv3(concat4)

#         concat3 = adaptative_cat(out_conv3, out_deconv3, normal4_up)+inputS_conv1
#         out3 = self.predict_normal3(concat3)

#         normal3_up = self.upsampled_normal3_to_2(out3)
#         out_deconv2 = self.deconv2(concat3)

#         concat2 = adaptative_cat(out_conv2, out_deconv2, normal3_up)+inputS_conv0
#         out2 = self.predict_normal2(concat2)
#         # normal2 = out2
#         # maskC2 = self.predict_mask(concat2)

#         # return normal2,maskC2
#         return out2

# class depthCompletionNewD(nn.Module):
#     def __init__(self):
#         super(depthCompletionNewD, self).__init__()
#         self.bs = 1
#         # self.normal = depthCompletionNewN(bs)
#         self.outC_block = depthCompletionNew_block(self.bs)
#         # self.outN_block = depthCompletionNew_blockN(bs)

#     def forward(self, rgb_in,input):
#         # normal_in = self.normal(left, sparse, mask)

#         out = self.outC_block(rgb_in,input)

#         # outN = self.outN_block(normal_in, sparse, maskC)

#         if self.training:
#             return out
#         else:
#             return out



import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# class UNet(nn.Module):
class refineNet(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv1_d = conv_bn_relu(1, 8 , kernel_size=3, stride=1, padding=1)
        # self.conv1_img = conv_bn_relu(3, 8 , kernel_size=3, stride=1, padding=1)
        self.input= conv_bn_relu(4, 64, kernel_size=3, stride=1, padding=1)

        self.dconv_down1 = double_conv(64, 64)
        self.dconv_down2 = double_conv(96, 128)
        self.dconv_down3 = double_conv(192, 256)
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
        
        self.conv_last = nn.Conv2d(64, 2, 1)
        
        
    def forward(self, input_,em1,em2):
       

        # conv1_d = self.conv1_d(x1)
        # conv1_img = self.conv1_img(x2)
        # in_put = torch.cat((conv1_d, conv1_img),1)
        # print(feature1.shape,feature2.shape)  #torch.Size([4, 32, 128, 608]) torch.Size([4, 64, 64, 304])
        in_put=self.input(input_)

        conv1 = self.dconv_down1(in_put)
        # print(conv1_.shape,em1.shape) #[2, 64, 256, 1216]) torch.Size([2, 32, 128, 608])
        x = self.maxpool(conv1)  

        x = torch.cat((x, em1), 1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        x = torch.cat((x, em2), 1)
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
