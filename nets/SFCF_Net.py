#----------------------------------------------------#
# Spatial-Frequency Cross-Fusion Network (SFCF-Net)
# @author:  Xiang Tian
#----------------------------------------------------#
import torch
import torch.nn as nn

import math
import torch.nn.functional as F
import numpy as np

# ============================================================================================
#                                    Model Components
# ============================================================================================

# ------------------------------------ Basic Modules -----------------------------------------

def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_Asym_Inception(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size, 1], padding=tuple([padding, 0]), dilation=(dilation, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0, padding]), dilation=(1, dilation)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model


# TODO: Change order of block: BN + Activation + Conv
def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(3)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2, dtype=int)
    croppBorders[0] = int(diff[0] / 2)
    croppBorders[1] = int(diff[1] / 2)

    return tensorToCrop[:,
           :,
           croppBorders[0]:org_shape[2] - croppBorders[0],
           croppBorders[1]:org_shape[3] - croppBorders[1]]


# ---------------------------------- Inception Modules ---------------------------------
# The Inception-V3 module employs convolution kernels of different sizes, working concurrently to capture multiscale features

class Conv_residual_conv_Inception_Dilation(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,dilation=1)
        self.conv_2_2 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2_3 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv_2_4 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=2, dilation=2)

        self.conv_2_output = conv_block(self.out_dim * 4, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)

        out1 = torch.cat([conv_2_1,conv_2_2, conv_2_3, conv_2_4], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3


class Conv_residual_conv_Inception_Dilation_asymmetric(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation_asymmetric, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,dilation=1)
        self.conv_2_2 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=1, dilation=1)
        self.conv_2_3 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1,
                                                  padding=2, dilation=1)
        self.conv_2_4 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=2, dilation=2)

        self.conv_2_output = conv_block(self.out_dim * 4, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)

        out1 = torch.cat([conv_2_1,conv_2_2, conv_2_3, conv_2_4], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3


# ---------------------------------- Attention Modules ---------------------------------
# The channel attention and spatial attention mechanism below is proposed in the paper:

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x



# ============================================================================================
#                                     Complete Model
# ============================================================================================
#    Spatial-Frequency Cross-Fusion Network (SFCF-Net)
#    Model instantiation in the published paper:

class SFCF_Net(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super(SFCF_Net, self).__init__()
        print('~' * 50)
        print(' ----- Creating Spatial-Frequency Cross-Fusion Network...')
        print('~' * 50)

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        act_fn = nn.ReLU()

        act_fn_2 = nn.ReLU()

        # ~~~ Spatial and frequency information extraction~~~~~~ #
        # (Frame 1)
        self.down_1_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_0 = maxpool()
        self.down_2_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_0 = maxpool()
        self.down_3_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_0 = maxpool()
        self.down_4_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_0 = maxpool()

        # (Frame 2)
        self.down_1_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_1 = maxpool()
        self.down_2_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_1 = maxpool()
        self.down_3_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_1 = maxpool()
        self.down_4_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_1 = maxpool()

        # (Frame 3)
        self.down_1_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_2 = maxpool()
        self.down_2_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_2 = maxpool()
        self.down_3_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_2 = maxpool()
        self.down_4_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_2 = maxpool()

        # (Frame 4)
        self.down_1_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_3 = maxpool()
        self.down_2_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_3 = maxpool()
        self.down_3_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_3 = maxpool()
        self.down_4_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_3 = maxpool()

        # Bridge between two modules
        self.bridge = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 60, self.out_dim * 16, act_fn)

        # ~~~ Information fusion  ~~~~~~ #
        self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv_Inception_Dilation(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.cbam1 = cbam_block(self.out_dim * 8)

        self.deconv_2 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv_Inception_Dilation(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.cbam2 = cbam_block(self.out_dim * 4)

        self.deconv_3 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv_Inception_Dilation(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.cbam3 = cbam_block(self.out_dim * 2)

        self.deconv_4 = conv_decod_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv_Inception_Dilation(self.out_dim, self.out_dim, act_fn_2)
        self.cbam4 = cbam_block(self.out_dim)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)



    def forward(self, i0,i1,i2,i3):

        # ############################# #

        # ~~~ Spatial and frequency information extraction~~~~~~ #
        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)
        down_1_1 = self.down_1_1(i1)
        down_1_2 = self.down_1_2(i2)
        down_1_3 = self.down_1_3(i3)

        # -----  Second Level --------
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3)), dim=1)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0)), dim=1)

        input_2nd_2 = torch.cat((self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1)), dim=1)

        input_2nd_3 = torch.cat((self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2)), dim=1)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        down_2_2 = self.down_2_2(input_2nd_2)
        down_2_3 = self.down_2_3(input_2nd_3)

        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_0(down_2_1)
        down_2_2m = self.pool_2_0(down_2_2)
        down_2_3m = self.pool_2_0(down_2_3)


        input_3rd_0 = torch.cat((down_2_0m, down_2_1m, down_2_2m, down_2_3m), dim=1)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)

        input_3rd_1 = torch.cat((down_2_1m, down_2_2m, down_2_3m, down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        input_3rd_2 = torch.cat((down_2_2m, down_2_3m, down_2_0m, down_2_1m), dim=1)
        input_3rd_2 = torch.cat((input_3rd_2, croppCenter(input_2nd_2, input_3rd_2.shape)), dim=1)

        input_3rd_3 = torch.cat((down_2_3m, down_2_0m, down_2_1m, down_2_2m), dim=1)
        input_3rd_3 = torch.cat((input_3rd_3, croppCenter(input_2nd_3, input_3rd_3.shape)), dim=1)

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)
        down_3_3 = self.down_3_3(input_3rd_3)


        # -----  Fourth Level --------
        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_0(down_3_1)
        down_3_2m = self.pool_3_0(down_3_2)
        down_3_3m = self.pool_3_0(down_3_3)

        input_4th_0 = torch.cat((down_3_0m, down_3_1m, down_3_2m, down_3_3m), dim=1)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1m, down_3_2m, down_3_3m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        input_4th_2 = torch.cat((down_3_2m, down_3_3m, down_3_0m, down_3_1m), dim=1)
        input_4th_2 = torch.cat((input_4th_2, croppCenter(input_3rd_2, input_4th_2.shape)), dim=1)

        input_4th_3 = torch.cat((down_3_3m, down_3_0m, down_3_1m, down_3_2m), dim=1)
        input_4th_3 = torch.cat((input_4th_3, croppCenter(input_3rd_3, input_4th_3.shape)), dim=1)

        down_4_0 = self.down_4_0(input_4th_0)
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)
        down_4_3 = self.down_4_3(input_4th_3)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)
        down_4_3m = self.pool_4_0(down_4_3)

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m, down_4_3m), dim=1)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)
        bridge = self.bridge(inputBridge)


        # ############################# #
        # ~~~~~~ Information fusion ~~~~~~~  #
        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4_0 + down_4_1 + down_4_2 + down_4_3) / 5  # Residual connection
        skip_1 = self.cbam1(skip_1)
        up_1 = self.up_1(skip_1)

        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3_0 + down_3_1 + down_3_2 + down_3_3) / 5  # Residual connection
        skip_2=self.cbam2(skip_2)
        up_2 = self.up_2(skip_2)

        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2_0 + down_2_1 + down_2_2 + down_2_3) / 5  # Residual connection
        skip_3 = self.cbam3(skip_3)
        up_3 = self.up_3(skip_3)

        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1_0 + down_1_1 + down_1_2 + down_1_3) / 5  # Residual connection
        skip_4 = self.cbam4(skip_4)
        up_4 = self.up_4(skip_4)

        return self.out(up_4)