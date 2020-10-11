import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim



import torch
import torch.nn as nn
from torch.nn import Parameter
from functools import wraps
import torch.nn.utils.weight_norm as WeightNorm

class GlobalMaxPool(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, input):
        output = torch.max(input, dim=1)[0]

        return torch.unsqueeze(output, 1)


class UNet(nn.Module):

    # an implementation of Unet with global maxpooling presented in
    # http://people.csail.mit.edu/miika/eccv18_deblur/aittala_eccv18_deblur_preprint.pdf
    # dimensions are specified at http://people.csail.mit.edu/miika/eccv18_deblur/aittala_eccv18_deblur_appendix.pdf
    # Implemented by Frederik Warburg
    # For implementation details contact: frewar1905@gmail.com

    def conv_elu(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(out_channels)
        )

        return block

    def contracting_block(self, in_channels, out_channels, kernel_size=4):
        block = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, padding=1), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(out_channels),
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, stride = 2, padding=0), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=mid_channel, padding=1), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(mid_channel),
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(mid_channel),
            WeightNorm(torch.nn.ConvTranspose2d(kernel_size=4, in_channels=mid_channel, out_channels=out_channels, stride=2, padding=1), name = "weight")
        )
        return block

    def final_block(self, in_channels, out_channels, mid_channel, kernel_size=3):
        block = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding = 1), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(mid_channel),
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding = 1), name = "weight")
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        # Encode
        self.conv_encode1 = self.conv_elu(in_channels=in_channel,   out_channels= 64,  kernel_size= 3) #id 1
        self.conv_encode2 = self.contracting_block(in_channels= 2*64,  out_channels= 128, kernel_size= 4) #id 4, 5
        self.conv_encode3 = self.contracting_block(in_channels= 2*128, out_channels= 256, kernel_size= 4) #id 8, 9
        self.conv_encode4 = self.contracting_block(in_channels= 2*256, out_channels= 384, kernel_size= 4) #id 12, 13

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=1, in_channels=2*384, out_channels=384), name = "weight"), #id 16
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(384),
            WeightNorm(torch.nn.Conv2d(kernel_size=4, in_channels=384, out_channels=384, stride=2), name = "weight"), #id 17
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(384),
            WeightNorm(torch.nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, padding=1), name = "weight"), #id 18
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(384),
            WeightNorm(torch.nn.ConvTranspose2d(kernel_size=4, in_channels=384, out_channels=384, stride=2), name = "weight") #id 19
        )
        # Decode
        self.conv_decode3 = self.expansive_block(in_channels = 3 * 384, out_channels= 256, mid_channel= 384) #id 22, 23, 24
        self.conv_decode2 = self.expansive_block(in_channels = 3 * 256, out_channels= 192, mid_channel= 256) #id 27, 28, 29
        self.conv_decode1 = self.expansive_block(in_channels = 2 * 192 + 128, out_channels= 96 , mid_channel= 192) #id 32, 33, 34
        self.conv_decode0 = self.conv_elu(in_channels= 2*96 + 64, out_channels= 96,  kernel_size= 3)             #id 37, 38
        self.final_layer  = self.final_block(in_channels=96, out_channels = out_channel, mid_channel= 64)      #id 40, 41

        self.pooling = GlobalMaxPool()

    def concat(self, x, max_):

        b, im, c, h, w = x.size()

        x = x.view(b*im, c, h, w)
        max_ = max_.repeat(1, im, 1, 1, 1).view(b*im, c, h, w)
        output = torch.cat([x, max_], dim=1)

        return output

    def concat2(self, x, max_, y):
        b, im, c1, h, w = x.size()
        _, _, c2, _, _ = y.size()

        x = x.view(b * im, c1, h, w)
        y = y.view(b * im, c2, h, w)
        max_ = max_.repeat(1, im, 1, 1, 1).view(b * im, c1, h, w)
        output = torch.cat([x, max_, y], dim=1)

        return output

    def forward(self, x):

        b, im, c, h, w = x.size()

        # Encode
        encode_block1 = self.conv_encode1(x.view((b*im, c, h, w))) # id = 1
        _, _, h, w = encode_block1.size()
        features1 = encode_block1.view((b, im, -1, h, w))
        max_global_features1 = self.pooling(features1)  # id = 2

        encode_pool1 = self.concat(features1, max_global_features1) # id = 3
        encode_block2 = self.conv_encode2(encode_pool1) # id = 4, 5
        _, _, h, w = encode_block2.size()
        features2 = encode_block2.view((b, im, -1, h, w))
        max_global_features2 = self.pooling(features2) # id = 6

        encode_pool2 = self.concat(features2, max_global_features2)
        encode_block3 = self.conv_encode3(encode_pool2) # id = 8, 9
        _, _, h, w = encode_block3.size()
        features3 = encode_block3.view((b, im, -1, h, w))
        max_global_features3 = self.pooling(features3) # id = 10

        encode_pool3 = self.concat(features3, max_global_features3)
        encode_block4 = self.conv_encode4(encode_pool3) # id = 12, 13
        _, _, h, w = encode_block4.size()
        features4 = encode_block4.view((b, im, -1, h, w))
        max_global_features4 = self.pooling(features4) # id = 14

        # Bottleneck
        encode_pool4 = self.concat(features4, max_global_features4)
        bottleneck = self.bottleneck(encode_pool4) # id = 16, 17, 18, 19
        _, _, h, w = bottleneck.size()
        features5 = bottleneck.view((b, im, -1, h, w))
        max_global_features5 = self.pooling(features5) # id = 20

        # Decode
        decode_block4 = self.concat2(features5, max_global_features5, features4)
        cat_layer3 = self.conv_decode3(decode_block4) # id = 22, 23, 24
        _, _, h, w = cat_layer3.size()
        features6 = cat_layer3.view((b, im, -1, h, w))
        max_global_features6 = self.pooling(features6) # id = 25

        decode_block3 = self.concat2(features6, max_global_features6, features3)
        cat_layer2 = self.conv_decode2(decode_block3) # id = 27, 28, 29
        _, _, h, w = cat_layer2.size()
        features7 = cat_layer2.view((b, im, -1, h, w))
        max_global_features7 = self.pooling(features7) # id = 30

        decode_block2 = self.concat2(features7, max_global_features7, features2)
        cat_layer1 = self.conv_decode1(decode_block2) # id = 32, 33, 34
        _, _, h, w = cat_layer1.size()
        features8 = cat_layer1.view((b, im, -1, h, w))
        max_global_features8 = self.pooling(features8) # id = 35

        decode_block1 = self.concat2(features8, max_global_features8, features1)
        cat_layer0 = self.conv_decode0(decode_block1) # id = 37, 38
        _, _, h, w = cat_layer0.size()
        features9 = cat_layer0.view((b, im, -1, h, w))
        max_global_features9 = self.pooling(features9) # id = 39

        final_layer = self.final_layer(torch.squeeze(max_global_features9, dim = 1))
        return final_layer

