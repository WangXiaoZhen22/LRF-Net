import torch
import torch.nn as nn

import torch.nn as nn
import torch
from einops import rearrange
from torchsummary import summary

import torch.nn.functional as F
from functools import partial


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src


class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out, ks, dil=1):
        super(depthwise_separable_conv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=ks, padding='same', groups=ch_in,
                                    dilation=dil)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, kernel_size=5):
        super(REBNCONV, self).__init__()
        self.out_ch = out_ch
        self.change = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv_s1 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding='same', dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)
        self.conv_s2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding='same', dilation=1 * dirate)
        self.bn_s2 = nn.BatchNorm2d(out_ch)
        self.relu_s2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.change(x)
        hx = self.conv_s1(x)
        # hx = self.bn_s1(hx)
        # hx = self.relu_s1(hx)
        hx = self.conv_s2(hx)
        hx = self.bn_s2(hx)
        hx = self.relu_s2(hx)
        hx = hx + x
        return hx


class ResDWConv(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, kernel_size=3):
        super(ResDWConv, self).__init__()
        self.out_ch = out_ch
        self.change = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv_s1 = depthwise_separable_conv(out_ch, out_ch, ks=kernel_size, dil=dirate)
        self.relu_s1 = nn.ReLU(inplace=True)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.conv_s2 = depthwise_separable_conv(out_ch, out_ch, ks=kernel_size, dil=dirate)
        self.relu_s2 = nn.ReLU(inplace=True)
        self.bn_s2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.change(x)
        hx = self.conv_s1(x)
        hx = self.bn_s1(hx)
        hx = self.relu_s1(hx)
        hx = self.conv_s2(hx)
        hx = self.bn_s2(hx)
        hx = self.relu_s2(hx)
        hx = hx + x

        return hx


class Inception(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inception, self).__init__()

        mid_channel = int(out_channel / 4)
        self.change1 = REBNCONV(mid_channel, out_channel, kernel_size=1)
        self.change = REBNCONV(in_channel, mid_channel, kernel_size=1)

        # self.branch1x1 = REBNCONV(mid_channel, mid_channel, kernel_size=1)  # 输入外界通道数，经过1*1的卷积核，输出通道16
        # self.branch1x1_1 = REBNCONV(mid_channel, mid_channel, kernel_size=1)  # 输入外界通道数，经过1*1的卷积核，输出通道16

        self.branch5x5_1 = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=1)
        self.branch5x5_2 = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=1)

        self.branch5x5_1d = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=2)

        self.branch3x3_1 = REBNCONV(mid_channel, mid_channel, kernel_size=1)
        self.branch3x3_2 = REBNCONV(mid_channel, mid_channel, kernel_size=3)

        self.batch_pool = REBNCONV(mid_channel, mid_channel, kernel_size=1)

        self.conv_block1 = REBNCONV(4 * mid_channel, out_channel, kernel_size=3)
        self.conv_block2 = REBNCONV(out_channel, out_channel, kernel_size=1)
        self.trans = CBAMLayer(out_channel)

    def forward(self, x):
        x = self.change(x)
        mx = self.change1(x)

        # 处理路径2
        branch5 = self.branch5x5_1d(x)

        # 处理路径3
        branch3 = self.branch3x3_1(x)
        branch3 = self.branch3x3_2(branch3)
        # 处理路径4
        batch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        batch_pool = self.batch_pool(batch_pool)  # 平均池化，操作后图像尺寸不变
        # 处理路径5
        branch7 = self.branch5x5_1(x)
        branch7 = self.branch5x5_2(branch7)

        # 4条路径进行融合
        cats = [branch7, branch5, branch3, batch_pool]  # cats包含来自4条路径的4个特征图，他们具有相同的高度和宽度，不同的通道数
        output = torch.cat(cats, dim=1)  # torch.cat()用于在指定维度上连接多个张量，此处将四个特征图在通道数这个维度上进行连接起来
        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.trans(output)
        output = output + mx

        return output


class ResBlock(nn.Module):

    def __init__(self, in_ch=3, out_ch=12, max_ks=7, layer_num=7):
        super(ResBlock, self).__init__()

        self.convin = REBNCONV(in_ch, out_ch, kernel_size=1, dirate=1)
        ks = max_ks
        layers = []
        for i in range(layer_num):
            ks = ks - i * 2
            if ks < 3:
                ks = 3
                layers.append(REBNCONV(out_ch, out_ch, kernel_size=ks, dirate=1))
            elif ks <= 5:
                layers.append(REBNCONV(out_ch, out_ch, kernel_size=ks, dirate=1))
            else:
                layers.append(ResDWConv(out_ch, out_ch, kernel_size=ks, dirate=1))
        self.convblock = nn.Sequential(*layers)

        self.trans = CBAMLayer(out_ch)

    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx = self.convblock(hxin)
        hx = self.trans(hx)
        return hx + hxin


class UNET_CBAM(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, num_blocks=None, max_ks=7):
        super(UNET_CBAM, self).__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2, 2]
        self.stage1 = ResBlock(in_ch, 16, max_ks=max_ks, layer_num=num_blocks[0])
        # self.stage1 = RSU7(in_ch, 8, 16)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # ceil_mode=True：表示当池化核在输入特征图上滑动时，如果无法整除步幅时，将对输入进行向上取整的填充。

        self.stage2 = ResBlock(16, 32, max_ks=max_ks, layer_num=num_blocks[1])
        # self.stage2 = RSU6(16, 8, 32)

        self.stage3 = ResBlock(32, 64, max_ks=max_ks, layer_num=num_blocks[2])
        # self.stage3 = RSU5(32, 16, 64)

        self.stage4 = ResBlock(64, 128, max_ks=max_ks, layer_num=num_blocks[3])
        # self.stage4 = RSU4(64, 32, 128)

        self.stage5 = Inception(128, 256)

        self.stage5d = Inception(256, 128)
        self.stage4d = ResBlock(256, 64, max_ks=max_ks, layer_num=num_blocks[3])
        self.stage3d = ResBlock(128, 32, max_ks=max_ks, layer_num=num_blocks[2])
        self.stage2d = ResBlock(64, 16, max_ks=max_ks, layer_num=num_blocks[1])
        self.stage1d = ResBlock(32, 16, max_ks=max_ks, layer_num=num_blocks[0])

        self.side1 = nn.Conv2d(16, out_ch, 3, padding=1)

    def forward(self, x):
        # stage 1
        hx1 = self.stage1(x)

        # stage 2
        hx2 = self.stage2(self.pool(hx1))

        # stage 3
        hx3 = self.stage3(self.pool(hx2))

        # stage 4
        hx4 = self.stage4(self.pool(hx3))

        # stage 5
        hxd = self.stage5(self.pool(hx4))

        # decoder
        hxd = self.stage5d(hxd)
        hxd = _upsample_like(hxd, hx4)

        hxd = self.stage4d(torch.cat((hxd, hx4), 1))
        hxd = _upsample_like(hxd, hx3)

        hxd = self.stage3d(torch.cat((hxd, hx3), 1))
        hxd = _upsample_like(hxd, hx2)

        hxd = self.stage2d(torch.cat((hxd, hx2), 1))
        hxd = _upsample_like(hxd, hx1)

        hxd = self.stage1d(torch.cat((hxd, hx1), 1))

        # side output
        hxd = self.side1(hxd)

        return F.sigmoid(hxd)


if __name__ == '__main__':
    # num_blocks = [6, 6, 6, 6, 6]
    #
    # model = UNET(num_blocks=num_blocks).cuda()
    model = U2NET(1).cuda()
    # # model = Transfomer(in_channel=3, out_channel=16).cuda()

    # nb_filter = [16, 32, 64, 128, 256]
    # num_blocks = [5, 5, 5, 5]
    # model = res_UNet(num_classes=1, input_channels=3, num_blocks=num_blocks,
    #                  nb_filter=nb_filter).cuda()
    summary(model, input_size=(1, 256, 256), device='cuda')
