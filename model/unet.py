import torch
import torch.nn as nn
import torch.nn.functional as F

# Unet without transpose convolution instead upsampled convolution
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
        

class TestUNet(nn.Module):

    def __init__(self,in_channels,n_class):
        super().__init__()
                
        self.dconv_down1 = Residual(in_channels, 64,use_1x1conv=True)
        self.dconv_down2 = Residual(64, 128,use_1x1conv=True)
        self.dconv_down3 = Residual(128, 256,use_1x1conv=True)
        self.dconv_down4 = Residual(256, 512,use_1x1conv=True)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = Residual(512, 256,use_1x1conv=True)
        self.dconv_up2 = Residual(256 + 256, 128,use_1x1conv=True)
        self.dconv_up1 = Residual(256, 64,use_1x1conv=True)
        # self.dconv_up3 = double_conv(512, 256)
        # self.dconv_up2 = double_conv(256 + 256, 128)
        # self.dconv_up1 = double_conv(256, 64)
        
        self.conv_last = nn.Conv2d(128, n_class, 1)
        
        
    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        x = self.upsample(x)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x,conv3],dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x,conv2],dim=1)
        
        x = self.dconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x,conv1],dim=1)
        
        out = self.conv_last(x)
        return out


class MaskUNet(nn.Module):

    def __init__(self,in_channels,n_class):
        super().__init__()
                
        # self.dconv_down1 = Residual(in_channels, 64,use_1x1conv=True)
        # self.dconv_down2 = Residual(64, 128,use_1x1conv=True)
        # self.dconv_down3 = Residual(128, 256,use_1x1conv=True)
        # self.dconv_down4 = Residual(256, 512,use_1x1conv=True)
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.dconv_up3 = Residual(512, 256,use_1x1conv=True)
        # self.dconv_up2 = Residual(256 + 256, 128,use_1x1conv=True)
        # self.dconv_up1 = Residual(256, 64,use_1x1conv=True)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256+256 , 128)
        self.dconv_up1 = double_conv(128+128, 64)
        
        self.conv_last = nn.Conv2d(64+64, n_class, 1)
        
        
    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        x = self.upsample(x)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x,conv3],dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x,conv2],dim=1)
        
        x = self.dconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x,conv1],dim=1)
        
        out = self.conv_last(x)
        return out


