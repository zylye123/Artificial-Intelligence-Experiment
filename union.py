# %%
import torch
import cv2
import torch.nn as nn
from torchvision import models
import os
from torch.nn import functional as F
# from DetNet import DetNet
# from ASPP import ASPP_NET
# from SPP import SPP
# from Bottleneck import Bottleneck
import matplotlib.pyplot as plt# plt 用于显示图片
import matplotlib.image as mpimg# mpimg 用于读取图片
import numpy as np
import torchvision.transforms as transforms

device = ('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
"""
##一张图片运行
fea = mpimg.imread('D:/study/论文/人工智能中期/脊柱疾病智能诊断/脊柱疾病智能诊断/train/show/show_study10.jpg')
transform = transforms.Compose([
    transforms.ToTensor()
])
#fea = transform(fea).unsqueeze(0)
"""
#示例
# fea = torch.zeros((1,3,512,512))
# net = models.resnet18(pretrained= True)
# net = nn.Sequential(*list(net.children())[:-3])
# blocks = [DetNet, ASPP, SPP, Bottleneck]
# union_model = net.add_module("blocks", nn.Sequential(*blocks))
# %%
class BottleneckA(nn.Module):
    expansion: int = 4 #表示一个Bottleneck结构后通道数的扩张倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckA, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample #使残差部分和卷积前向出来之后通道一致
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual#参差部分相加
        out = self.relu(out)

        return out


class BottleneckB(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckB, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Det_Net(nn.Module):
    def __init__(self,inplanes = 256, planes =64, stride=1, downsample=None):
        super(Det_Net, self).__init__()
        self.layer1 = BottleneckB(inplanes, planes, stride=1, downsample=None)
        self.relu1 = nn.ReLU()
        self.layer2 = BottleneckA(inplanes, planes, stride=1, downsample=None)
        self.relu2 = nn.ReLU()
        self.layer3 = BottleneckA(inplanes, planes, stride=1, downsample=None)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        outa = self.layer1(x)
        outb = self.relu1(outa)
        outc = self.layer2(outb)
        outd = self.relu2(outc)
        oute = self.layer3(outd)
        outf = self.relu3(oute)
        return outf


# %%
class ASPP_NET(nn.Module):
    def __init__(self, inchannel = 256, outchannel = 256):
        super(ASPP_NET, self).__init__()
        self.dila_conv1 = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                        kernel_size= 3, stride = 1, padding = 1, dilation = 1),
                                         nn.BatchNorm2d(outchannel)
                                        )
        self.dila_conv2 = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                        kernel_size= 3, stride = 1, padding = 3, dilation = 3),
                                         nn.BatchNorm2d(outchannel)
                                        )
        self.conv1 = conv_1x1_bn(256,256)
        self.dila_conv3 =  nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                        kernel_size= 3, stride = 1, padding = 1, dilation = 1),
                                         nn.BatchNorm2d(outchannel)
                                        )
        self.dila_conv4 = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                        kernel_size= 3, stride = 1, padding = 3, dilation = 3),
                                         nn.BatchNorm2d(outchannel)
                                        )
        self.conv2 = conv_1x1_bn(256,256)
        self.dila_conv5 = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                        kernel_size= 3, stride = 1, padding = 1, dilation = 1),
                                         nn.BatchNorm2d(outchannel)
                                        )
        self.dila_conv6 = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                        kernel_size= 3, stride = 1, padding = 3, dilation = 3),
                                         nn.BatchNorm2d(outchannel)
                                        )
        self.dila_conv7 = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                        kernel_size= 3, stride = 1, padding = 5, dilation = 5),
                                         nn.BatchNorm2d(outchannel)
                                        )
        self.conv3 = conv_1x1_bn(256,256)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out1a = self.dila_conv1(x)
        out1b = self.relu(out1a)

        out2a = self.dila_conv2(x)
        out2b = self.relu(out2a)
        out2c = self.conv1(out2b)
        out2d = self.relu(out2c)

        out3a = self.dila_conv3(x)
        out3b = self.relu(out3a)
        out3c = self.dila_conv4(out3b)
        out3d = self.relu(out3c)
        out3e = self.conv2(out3d)
        out3f = self.relu(out3e)

        out4a = self.dila_conv5(x)
        out4b = self.relu(out4a)
        out4c = self.dila_conv6(out4b)
        out4d = self.relu(out4c)
        out4e = self.dila_conv7(out4d)
        out4f = self.relu(out4e)
        out4g = self.conv3(out4f)
        out4h = self.relu(out4g)

        net = self.relu(out1b + out2d + out3f + out4h + x)
        return net

def conv_1x1_bn(inp, oup =256):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup))


# %%
class SPP_NET(nn.Module):

    def __init__(self, inchannel = 256, outchannel=256):
        super(SPP_NET, self).__init__()
        self.mp1 =nn.MaxPool2d(kernel_size =2, stride=2)
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = 1, padding = 0, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.us1 =nn.Upsample(scale_factor=2)
        self.mp2 = nn.MaxPool2d(kernel_size =4, stride=4)
        self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = 1, padding = 0, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.us2 = nn.Upsample(scale_factor=4)
        self.mp3 = nn.MaxPool2d(kernel_size = 8, stride=8)
        self.conv3 = nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = 1, padding = 0, bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.us3 = nn.Upsample(scale_factor=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out1a = self.mp1(x)
        out1b = self.conv1(out1a)
        out1c = self.bn1(out1b)
        out1d = self.relu(out1c)
        out1e = self.us1(out1d)

        out2a = self.mp2(x)
        out2b = self.conv2(out2a)
        out2c = self.bn2(out2b)
        out2d = self.relu(out2c)
        out2e = self.us2(out2d)

        out3a = self.mp3(x)
        out3b = self.conv3(out3a)
        out3c = self.bn3(out3b)
        out3d = self.relu(out3c)
        out3e = self.us3(out3d)

        net = torch.cat((out1e,out2e,out3e,x),1)
        return net

# %%
class Bottleneck(nn.Module):
    def __init__(self, inchannel = 1024):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inchannel, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 1024, 1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.extra = nn.Conv2d(1024,1024,1)
        self.bn_extra = nn.BatchNorm2d(1024) #！！！！通道数较大，后续可能需要修改

    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)
        out4 = self.conv2(out3)
        out5 = self.bn2(out4)
        out6 = self.relu(out5)
        out7 = self.conv3(out6)
        out8 = self.bn3(out7)

        out0 = self.extra(x)
        net = out8 + out0
        output = self.relu(net)

        return output
# %%
class union_model(nn.Module):
    def __init__(self, in_channels):
        super(union_model, self).__init__()
        model = models.resnet18(pretrained = True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        res = model.to(device)
        #res =models.resnet18(pretrained = True)
        self.backbone = nn.Sequential(*list(res.children())[:-3])
        self.detnet = Det_Net()
        self.aspp = ASPP_NET()
        self.spp = SPP_NET()
        self.bottleneck = Bottleneck()
        self.conv1 = nn.Conv2d(1024, 11, 1)
        self.conv2 = nn.Conv2d(1024, 11, 1)
        self.conv3 = nn.Conv2d(1024, 11, 1)
        self.conv4 = nn.Conv2d(1024, 35, 1)
        self.sigmoid = torch.sigmoid
        self.dropout = nn.Dropout(0.3)
        self.tanh = torch.tanh
    
    def forward(self, img):
        #output1 = self.dropout(self.backbone(img))
        # output2 = self.dropout(self.detnet(output1))
        # output3 = self.dropout(self.aspp(output2))
        # output4 = self.dropout(self.spp(output3))
        # output5 = self.dropout(self.bottleneck(output4))

        out1 = self.sigmoid(self.conv1(self.dropout(self.bottleneck(self.spp(self.aspp(self.detnet(self.backbone(img))))))))
        out2 = self.tanh(self.conv2(self.dropout(self.bottleneck(self.spp(self.aspp(self.detnet(self.backbone(img))))))))
        out3 = self.tanh(self.conv3(self.dropout(self.bottleneck(self.spp(self.aspp(self.detnet(self.backbone(img))))))))
        out4 = self.sigmoid(self.conv4(self.dropout(self.bottleneck(self.spp(self.aspp(self.detnet(self.backbone(img))))))))


        


        return out1, out2, out3, out4

# %%
# model = Union_model(3)
# model = model.to(device)
# matrix = torch.zeros((1, 3, 512, 512))
# matrix = matrix.to(device)
# (out1, out2, out3, out4) = model(matrix)
# %%
if __name__ == '__main__':
    print('model')
# # %%

# #学习率和Adam用了老师上次作业的先试试
# def define_optimizer():
#     learning_rate = 1e-4
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#     return optimizer
# #训练函数
# def train():
#     for t in range(100):
#         img=cv2.imread(img_files[0])#大致读一个
#         model = Union_model(3)
#         model = model.to(device)
#         matrix = img
#         matrix = matrix.to(device)
#         (out1, out2, out3, out4) = model(matrix)
#         #把load_txt的变量读进来，然后丢进loss函数
#         #loss= Loss_sum(Rmap,Rmap_true,Offymap_true,Offymap,Offxmap_true,Offxmap,Classmap_true,Classmap,is_zhuiti)
#         optimizer = define_optimizer()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     net_path = r'C:\Users\一脚一个小朋友\Desktop\人工智能综合实验'
#     torch.save(net, net_path)
#     return net_path

# %%
