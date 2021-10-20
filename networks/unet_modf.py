import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class ResNet50(nn.Module):
    def __init__(self,pretrained=False):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

    def forward(self,x):
        x = self.resnet(x)
        return x

class ClassLinear(nn.Module):
    def __init__(self, num_classes=2, in_dim=1000):
        super(ClassLinear, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64,18),
                                nn.BatchNorm1d(18),
                                nn.ReLU(),
                                nn.Linear(18,num_classes),
                                )
    def forward(self,x):
        x = self.fc(x)
        # x = torch.sigmoid(x)
        return x

class CenterLinear(nn.Module):
    def __init__(self):
        super(CenterLinear, self).__init__()
        self.fc = nn.Sequential(nn.Linear(1000,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64,18),
                                nn.BatchNorm1d(18),
                                nn.ReLU(),
                                nn.Linear(18,5),
                                )
    def forward(self,x):
        x = self.fc(x)
        return x



class Unet_Conv(nn.Module):
    def __init__(self,nc=1):
        super(Unet_Conv, self).__init__()
        self.nc = nc
        # Divde Convolution Path as 4 distinct layers, details is in the paper shao2020
        self.layer_1_1 = nn.Sequential(nn.Conv2d(self.nc,16,3,1,1), nn.LeakyReLU(True))      # 16, 256, 256,
        self.layer_1_2 = nn.Sequential(nn.Conv2d(self.nc,16,5,2,2), nn.LeakyReLU(True))       # 16, 128, 128
        self.layer_1_3 = nn.Sequential(nn.Conv2d(self.nc,16,7,4,3), nn.LeakyReLU(True))     # 16, 64, 64
        self.layer_1_4 = nn.Sequential(nn.Conv2d(self.nc,16,9,8,4), nn.LeakyReLU(True))      # 16, 32, 32

        self.layer_2_1 = nn.Sequential(nn.Conv2d(16,16,3,2,1), nn.LeakyReLU(True))      # 16, 128, 128

        self.layer_3_1 = nn.Sequential(nn.Conv2d(16,16,3,2,1), nn.LeakyReLU(True))      # 16, 64, 64
        self.layer_3_2 = nn.Sequential(nn.Conv2d(32,32,3,2,1), nn.LeakyReLU(True))      # 32, 64, 64

        self.layer_4_1 = nn.Sequential(nn.Conv2d(16,16,3,2,1), nn.LeakyReLU(True))      # 16, 32, 32
        self.layer_4_2 = nn.Sequential(nn.Conv2d(32,32,3,2,1), nn.LeakyReLU(True))      # 32, 32, 32
        self.layer_4_3 = nn.Sequential(nn.Conv2d(64,64,3,2,1), nn.LeakyReLU(True))      # 64, 32, 32


        self.layer_1 = [self.layer_1_1,self.layer_1_2 ,self.layer_1_3 ,self.layer_1_4]
        self.layer_2 = [self.layer_2_1]
        self.layer_3 = [self.layer_3_1,self.layer_3_2]
        self.layer_4 = [self.layer_4_1,self.layer_4_2,self.layer_4_3]

    # each path represent a layer calculation and return of stacks that needed for ruther path
    # output for each path is a list of tensor and a stacked layer
    def pass_1(self, input):
        output = []
        for i in range(len(self.layer_1)):
            out = self.layer_1[i](input)
            output.append(out)
        stack_1 = output[0]
        return output, stack_1

    def pass_2(self, input):
        output = input
        output[0] = self.layer_2[0](input[0])
        return output

    def pass_3(self, input):
        output = input
        stack_2 = torch.cat([input[0],input[1]],1)
        output[0] = self.layer_3[0](input[0])
        output[1] = self.layer_3[1](stack_2)
        return output, stack_2

    def pass_4(self, input):
        output = input
        stack_3 = torch.cat([input[0],input[1],input[2]],1)
        output[0] = self.layer_4[0](input[0])
        output[1] = self.layer_4[1](input[1])
        output[2] = self.layer_4[2](stack_3)
        return output, stack_3

    def forward(self,x):

        # stack_1 sould be in shape of [nc, 16, 256, 256]
        out_1, stack_1 = self.pass_1(x)
        out_2 = self.pass_2(out_1)
        # stack_2 sould be in shape of [nc, 32, 128, 128]
        out_3, stack_2 = self.pass_3(out_2)
        # stack_3 sould be in shape of [nc, 64, 64, 64]
        out_4, stack_3 = self.pass_4(out_3)
        # stack_4 sould be in shape of [nc, 128, 32, 32]
        stack_4 = torch.cat(out_4,1)
        # return list of stacks for De-Conv
        return [stack_1,stack_2,stack_3,stack_4]


class Unet_DeConv(nn.Module):
    def __init__(self, nc=128):
        super(Unet_DeConv, self).__init__()
        self.nc = nc
        # Divde Convolution Path as 4 distinct layers, details is in the paper shao2020
        self.layer_1 = nn.Sequential(
                                    nn.ConvTranspose2d(self.nc,128,2,2), # 128, 64, 64,
                                    nn.LeakyReLU(True)
                                    )
        self.layer_2 = nn.Sequential(
                                    nn.Conv2d(192,128,3,1,1), # 128, 64, 64
                                    nn.LeakyReLU(True),
                                    nn.ConvTranspose2d(128,64,2,2), # 64, 128, 128
                                    nn.LeakyReLU(True)
                                    )
        self.layer_3 = nn.Sequential(
                                    nn.Conv2d(96,64,3,1,1), # 64, 128, 128
                                    nn.LeakyReLU(True),
                                    nn.ConvTranspose2d(64,32,2,2), # 32, 256, 256
                                    nn.LeakyReLU(True)
                                    )
        self.layer_4 = nn.Sequential(
                                    nn.Conv2d(48,32,3,1,1), # 32, 258, 258
                                    nn.LeakyReLU(True),

                                    # Added for smaller faltten layer
                                    nn.Conv2d(32,8,3,1,1), # 8, 258, 258
                                    nn.LeakyReLU(True),

                                    # nn.Conv2d(8,4,3,1,1), # 4, 258, 258
                                    # nn.LeakyReLU(True),

                                    ## orginal last lyer is 32*256*256
                                    nn.AdaptiveAvgPool2d((6,6)),
                                    Flatten(),
                                    # nn.Linear(8*256*256,1000), # 1, 1000, 1000
                                    nn.Linear(8*6*6,256), # 1, 1000, 1000
                                    nn.LeakyReLU(True),
                                    # nn.Dropout(0.5)
                                    )

    def forward(self,x):
        # unpact results from encoder
        stack_1,stack_2,stack_3,stack_4 = x
        # layer 1 calculation
        input_1 = stack_4
        out_1 = self.layer_1(input_1)
        # layer 2 calculation
        input_2 = torch.cat([stack_3,out_1],1)
        out_2 = self.layer_2(input_2)
        # layer 3 calculation
        input_3 = torch.cat([stack_2,out_2],1)
        out_3 = self.layer_3(input_3)
        # layer 4 calculation
        input_4 = torch.cat([stack_1,out_3],1)
        out_4 = self.layer_4(input_4)
        return out_4


class Unet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(Unet, self).__init__()
        self.Conv = Unet_Conv(in_channels)
        self.DeConv = Unet_DeConv(128)
        self.classifier = ClassLinear(num_classes=num_classes, in_dim=256)


    def forward(self,x, *args):
        conv_out = self.Conv(x)
        deconv_out = self.DeConv(conv_out)
        classified = self.classifier(deconv_out)
        return classified


# test_input = torch.rand(50,1,256,256)
#
# unet = Unet()
#
# output = unet(test_input)
# print(output.size())
if __name__ == "__main__":
    import numpy as np
    from torchinfo import summary
    # for net_name in __all__:
    #     if net_name.startswith('resnet20'):
    #         print(net_name)
    #         test(globals()[net_name]())
    #         print()
    net = Unet(num_classes=2, in_channels=1)
    summary(net, input_size=[(2, 1, 256, 256), (2, 12)])