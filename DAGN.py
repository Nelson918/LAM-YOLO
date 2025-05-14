import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // (2*self.groups), channels // (2*self.groups), kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(channels // (2*self.groups), channels // (2*self.groups), kernel_size=5, stride=1, padding=2)
        # self.conv7x7 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=7, stride=1, padding=3)

        self.G=factor
        self.channel=channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn1 = nn.GroupNorm(channels // (2 * factor), channels // (2 * factor))
        self.cweight = Parameter(torch.zeros(1, channels // (2 * factor), 1, 1))
        self.cbias = Parameter(torch.ones(1, channels // (2 * factor), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channels // (2 * factor), 1, 1))
        self.sbias = Parameter(torch.ones(1, channels // (2 * factor), 1, 1))
        self.sigmoid=nn.Sigmoid()

    @staticmethod
    def channel_shuffle(y, g):
        b, c, h, w = y.shape
        y = y.reshape(b, g, -1, h, w)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(b, -1, h, w)

        return y

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w

        # x_hw = self.conv7x7(group_x)
        x_h = self.pool_h(x_hw)
        x_w = self.pool_w(x_hw).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # group into subfeatures
        # x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w
        # channel_split
        x_0, x_1 = group_x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w
        # x_0 = self.conv3x3(x_0)
        # x_1 = self.conv5x5(x_1)
        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)
        # spatial attention
        x_spatial = self.gn1(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w
        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        # out = out.contiguous().view(b, -1, h, w)
        # channel shuffle
        x2 = self.channel_shuffle(out, 2)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w) # b*g, 1, h, w

        # y = group_x * weights.sigmoid()
        # y = self.channel_shuffle(y, 2)
        # output = y.reshape(b, c, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)                  # (group_x * weights.sigmoid()).reshape(b, c, h, w)
