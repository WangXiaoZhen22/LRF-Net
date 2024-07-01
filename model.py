import torch.nn as nn
import torch
from einops import rearrange
from torchsummary import summary

import torch.nn.functional as F
from functools import partial


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src


# RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, kernel_size=7, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, kernel_size=7, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, kernel_size=5, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, kernel_size=5, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, kernel_size=5, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, kernel_size=5, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, kernel_size=7, dirate=1)
        self.cbam_module = Transfomer(in_channel=out_ch, out_channel=out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx = self.pool4(hx4)

        hxd = self.rebnconv5(hx)
        hxd = self.rebnconv6(hxd)
        hxd = self.rebnconv7(hxd)
        hxd = self.rebnconv6d(hxd)
        hxd = self.rebnconv5d(hxd)
        hxd = _upsample_like(hxd, hx4)
        # hx5dup = self.attention1(hx5dup, hx4)

        hxd = self.rebnconv4d(torch.cat((hxd, hx4), 1))
        hxd = _upsample_like(hxd, hx3)
        # hx4dup = self.attention2(hx4dup,hx3)

        hxd = self.rebnconv3d(torch.cat((hxd, hx3), 1))
        hxd = _upsample_like(hxd, hx2)

        hxd = self.rebnconv2d(torch.cat((hxd, hx2), 1))
        hxd = _upsample_like(hxd, hxin)

        hxd = self.rebnconv1d(torch.cat((hxd, hx1), 1))

        hxd = self.cbam_module(hxd)
        hxd = hxd + hxin  #
        # print(hxout.shape)

        return hxd


# RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, kernel_size=7, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, kernel_size=5, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5p = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, kernel_size=5, dirate=1)
        self.cbam_module = Transfomer(in_channel=out_ch, out_channel=out_ch)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hxd = self.rebnconv4(hx)

        hxd = self.rebnconv5(hxd)

        hxd = self.rebnconv6(hxd)

        hxd = self.rebnconv5d(hxd)

        hxd = self.rebnconv4d(hxd)
        hxd = _upsample_like(hxd, hx3)

        hxd = self.rebnconv3d(torch.cat((hxd, hx3), 1))
        hxd = _upsample_like(hxd, hx2)

        hxd = self.rebnconv2d(torch.cat((hxd, hx2), 1))
        hxd = _upsample_like(hxd, hxin)

        hxd = self.rebnconv1d(torch.cat((hxd, hx1), 1))
        # hxout = self.cbam_module(hx1d + hxin)
        hxd = self.cbam_module(hxd)
        hxd = hxd + hxin
        # print(hxout.shape)

        return hxd


# RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, kernel_size=7, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, kernel_size=5, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        # self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, kernel_size=5, dirate=1)
        self.cbam_module = Transfomer(in_channel=out_ch, out_channel=out_ch)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hxd = self.rebnconv3(hx)

        # hx = self.pool3(hx3)

        hxd = self.rebnconv4(hxd)

        hxd = self.rebnconv5(hxd)

        hxd = self.rebnconv4d(hxd)

        hxd = self.rebnconv3d(hxd)
        hxd = _upsample_like(hxd, hx2)

        hxd = self.rebnconv2d(torch.cat((hxd, hx2), 1))
        hxd = _upsample_like(hxd, hxin)

        hxd = self.rebnconv1d(torch.cat((hxd, hx1), 1))
        # hxout = self.cbam_module(hx1d)

        hxd = self.cbam_module(hxd)
        hxd = hxd + hxin  #
        # print(hxout.shape)

        return hxd


# RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        # self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.cbam_module = Transfomer(in_channel=out_ch, out_channel=out_ch)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hxd = self.rebnconv2(hx)
        # hx = self.pool2(hx2)

        hxd = self.rebnconv3(hxd)

        hxd = self.rebnconv4(hxd)

        hxd = self.rebnconv3d(hxd)

        hxd = self.rebnconv2d(hxd)
        hxd = _upsample_like(hxd, hx1)

        hxd = self.rebnconv1d(torch.cat((hxd, hx1), 1))
        hxd = self.cbam_module(hxd)
        hxd = hxd + hxin  #

        return hxd


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


# 定义SE注意力机制的类
class se_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(se_block, self).__init__()

        # 属性分配
        ks = int(int(in_channel / 20) * 2 + 1)
        if ks < 3:
            ks = 3
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.conv1 = nn.Conv1d(2, 1, kernel_size=ks, padding='same')

        # relu激活
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(1, 1, kernel_size=ks, padding='same')
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        y_avg = self.avg_pool(inputs)
        y_max = self.max_pool(inputs)

        y = torch.concat([y_avg, y_max], dim=2)  # [b,c,2,1]
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = y.squeeze(-1).transpose(-1, -2)  # [b,2,c]

        y = self.conv1(y)  # [b,1,c]

        # 对通道权重归一化处理
        y = y.view([b, 1, c])
        y = self.conv2(y)
        y = self.sigmoid(y)

        # 调整维度 [b,c]==>[b,c,1,1]
        y = y.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = y * inputs
        return outputs


class CBAMLayer(nn.Module):
    def __init__(self, channel, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.channel = channel
        self.eca = se_block(self.channel)
        # spatial attention
        self.conv = nn.Conv2d(4, 1, kernel_size=spatial_kernel,
                              padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(channel, 2, kernel_size=1, stride=1)
        # self.conv2 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding='same')
        # self.conv3 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding='same')
        # self.conv4 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        torch.cuda.empty_cache()
        # eca 通道注意力
        x = self.eca(x)

        # 空间注意力
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        conv_out = self.conv1(x)
        spatial_out = torch.cat([max_out, avg_out, conv_out], dim=1)

        spatial_out = self.conv(spatial_out)
        spatial_out = self.sigmoid(spatial_out)

        x = spatial_out * x
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
        self.trans = Transfomer(out_channel, out_channel)

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


class Attention(nn.Module):
    def __init__(self,
                 in_ch,  # 输入token的dim
                 num_heads=8,
                 qk_scale=None,
                 patch_size=16,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = in_ch // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(in_ch, in_ch * 3, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, C, H, W = x.shape
        C1 = int(C * H * W / self.patch_size ** 2)
        # x = x.reshape(B,C1,self.patch_size,self.patch_size)

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, self.patch_size ** 2, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_ch, mlp_ratio=2):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch * mlp_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=in_ch * mlp_ratio, out_channels=in_ch, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class Transfomer(nn.Module):
    def __init__(self, in_channel, out_channel=64, mlp_ratio=2):
        super(Transfomer, self).__init__()
        self.out_ch = out_channel

        self.change = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.attention = CBAMLayer(channel=out_channel, spatial_kernel=11)
        self.mlp = MLP(out_channel, mlp_ratio=mlp_ratio)
        # self.conv = REBNCONV(out_channel, out_channel, kernel_size=5)

    def forward(self, x):
        BS, _, h, w = x.size()
        hx = self.change(x)
        # hx1 = self.conv(hx)
        hx1 = self.attention(hx)
        hx1 = self.norm1(hx1)
        hx = hx1 + hx
        hx = self.mlp(hx)
        hx = self.norm2(hx)
        hx = hx + hx1
        return hx


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

        self.trans = Transfomer(out_ch, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx = self.convblock(hxin)
        hx = self.trans(hx)
        return hx + hxin


# class ResBlock(nn.Module):
#
#     def __init__(self, in_ch=3, out_ch=12, max_ks=7, layer_num=7):
#         super(ResBlock, self).__init__()
#
#         self.convin = REBNCONV(in_ch, out_ch, kernel_size=1, dirate=1)
#         ks = max_ks
#         layers = []
#         if ks <= 5:
#             layers.append(REBNCONV(out_ch, out_ch, kernel_size=ks, dirate=1))
#         else:
#             layers.append(ResDWConv(out_ch,out_ch,kernel_size=ks,dirate=1))
#
#         layers.append(REBNCONV(out_ch, out_ch, kernel_size=5, dirate=1))
#         for i in range(layer_num - 2):
#             layers.append(REBNCONV(out_ch, out_ch, kernel_size=3, dirate=1))
#
#         self.convblock = nn.Sequential(*layers)
#
#         self.trans = Transfomer(out_ch, out_ch)
#
#     def forward(self, x):
#         hx = x
#         hxin = self.convin(hx)
#         hx = self.convblock(hxin)
#         hx = self.trans(hx)
#         return hx + hxin


class UNET(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, num_blocks=None, max_ks=7):
        super(UNET, self).__init__()

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


# U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 8, 16)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(16, 8, 32)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(32, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 32, 128)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = Inception(128, 256)

        # decoder
        self.stage5d = Inception(256, 128)
        self.stage4d = RSU4(256, 32, 64)
        self.stage3d = RSU5(128, 16, 32)
        self.stage2d = RSU6(64, 8, 16)
        self.stage1d = RSU7(32, 8, 16)
        # merge
        self.merge1 = MLP(16, 2)
        self.merge2 = MLP(32, 2)
        self.merge3 = MLP(64, 2)
        self.merge4 = MLP(128, 2)
        self.merge5 = MLP(256, 2)

        self.side1 = nn.Conv2d(16, out_ch, 3, padding=1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx1 = self.merge1(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx2 = self.merge2(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx3 = self.merge3(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx4 = self.merge4(hx4)

        # stage 5
        hxd = self.stage5(hx)
        hxd = self.merge5(hxd)
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
