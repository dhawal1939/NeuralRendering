import torch
import torch.nn as nn
import torch.nn.functional as F


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            # nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, output_pad=0, concat=True, final=False, tanh=False):
        super(up, self).__init__()
        self.concat = concat
        self.final = final
        if self.final:
            if tanh:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                    # nn.InstanceNorm2d(out_ch),
                    nn.Tanh()
                )
            else:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                    # nn.InstanceNorm2d(out_ch),
                    nn.Sigmoid()
                )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                # nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x1, x2):
        if self.concat:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv(x1)
        return x1

class UNetSH(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.down1 = down(input_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.up1 = up(512, 256, output_pad=1, concat=False)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, output_channels, final=True, tanh=False)

    # Adjusting for the input of real data, 176x176
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, None)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


class UNet(nn.Module):
    # def __init__(self, input_channels, output_channels):
    #     super(UNet, self).__init__()
    #     self.down1 = down(input_channels, 64)
    #     self.down2 = down(64, 128)
    #     self.down3 = down(128, 256)
    #     self.down4 = down(256, 512)
    #     self.down5 = down(512, 512)
    #     self.up1 = up(512, 512, output_pad=1, concat=False)
    #     self.up2 = up(1024, 512)
    #     self.up3 = up(768, 256)
    #     self.up4 = up(384, 128)
    #     self.up5 = up(192, output_channels, final=True)

    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.down1 = down(input_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.up1 = up(512, 256, output_pad=1, concat=False)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, output_channels, final=True)

    # Adjusting for the input of real data, 176x176
    def forward(self, x):
        print(x.shape)
        x1 = self.down1(x)
        print(x1.shape)
        x2 = self.down2(x1)
        print(x2.shape)
        x3 = self.down3(x2)
        print(x3.shape)
        x4 = self.down4(x3)
        print(x4.shape)

        x = self.up1(x4, None)
        print(x.shape)
        x = self.up2(x, x3)
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        print(x.shape)
        return x

    # def forward(self, x):
    #     x1 = self.down1(x)
    #     x2 = self.down2(x1)
    #     x3 = self.down3(x2)
    #     x4 = self.down4(x3)
    #     x5 = self.down5(x4)
    #     # print(x.shape)
    #     x = self.up1(x5, None)
    #     x = self.up2(x, x4)
    #     x = self.up3(x, x3)
    #     x = self.up4(x, x2)
    #     x = self.up5(x, x1)
    #     return x
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
        # self.bn1 = nn.BatchNorm2d(num_channels)
        # self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn1 = nn.InstanceNorm2d(num_channels)
        self.bn2 = nn.InstanceNorm2d(num_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        # Y = F.relu(self.bn1(self.conv1(X)))
        # print(Y.shape)
        # Y = self.bn2(self.conv2(Y))
        Y = self.conv2(Y)
        # print(Y.shape)
        if self.conv3:
            X = self.conv3(X)
            # print(X.shape)
        Y += X
        return F.relu(Y)
        
class TestUNetHybrid(nn.Module):

    def __init__(self,in_channels,n_class):
        super().__init__()
                
        self.dconv_down1 = Residual(in_channels, 64,use_1x1conv=True,strides=2)
        self.dconv_down2 = Residual(64, 128,use_1x1conv=True,strides=2)
        self.dconv_down3 = Residual(128, 256,use_1x1conv=True,strides=2)
        self.dconv_down4 = Residual(256, 512,use_1x1conv=True,strides=2)        

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
        # x = self.maxpool(conv1)
        # print(x.shape)
        conv2 = self.dconv_down2(conv1)
        # x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(conv2)
        # x = self.maxpool(conv3)
        
        x = self.dconv_down4(conv3)
        # x = self.maxpool(x)
        # x = self.upsample(x)

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
        out = self.upsample(out)
        return out
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

class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2):  # x1--up , x2 ---down
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (
            diffY // 2, diffY - diffY // 2,
            diffX // 2, diffX - diffX // 2,))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up3, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        # print(x1.shape)
        x1 = self.up(x1)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up4, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3, x4):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = self.up(x1)
        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_in(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 5, padding=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


cc = 16  # you can change it to 8, then the model can be more faster ,reaching 35 fps on cpu when testing


class Unet_2D(nn.Module):
    def __init__(self, n_channels, n_classes, mode='train'):
        super(Unet_2D, self).__init__()
        self.inconv = inconv(n_channels, cc)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.up1 = up(12 * cc, 4 * cc)
        self.up20 = up(6 * cc, 2 * cc)
        self.up2 = up3(8 * cc, 2 * cc)
        self.up30 = up(3 * cc, cc)
        self.up31 = up3(4 * cc, cc)
        self.up3 = up4(5 * cc, cc)
        self.outconv = outconv(cc, n_classes)
        self.mode = mode

    def forward(self, x):
        if self.mode == 'train':  # use the whole model when training
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x21 = self.up20(x3, x2)
            x = self.up2(x, x21, x2)
            x11 = self.up30(x2, x1)
            x12 = self.up31(x21, x11, x1)
            x = self.up3(x, x12, x11, x1)
            #output 0 1 2
            y2 = self.outconv(x)
            y0 = self.outconv(x11)
            y1 = self.outconv(x12)
            # return y0, y1, y2
            return y0
        else:  # prune the model when testing
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x11 = self.up30(x2, x1)
            # output 0
            y0 = self.outconv(x11)
            return y0
