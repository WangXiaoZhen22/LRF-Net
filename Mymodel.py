import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


# ECA模块
class EcaLayer(nn.Module):
    """Constructs an ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(EcaLayer, self).__init__()
        k_size = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.conv = nn.Conv1d(2, 1, kernel_size=k_size,
                              padding=1, bias=False)  # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]

        # feature descriptor on the global spatial information
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        y = torch.concat([y_avg, y_max], dim=2)

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        return x * y.expand_as(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.channel = channel
        self.eca = EcaLayer(self.channel)
        # spatial attention
        self.conv = nn.Conv2d(4, 1, kernel_size=spatial_kernel,
                              padding=int((spatial_kernel - 1) / 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(channel, 2, kernel_size=1, stride=1)

    def forward(self, x):
        torch.cuda.empty_cache()
        # eca 通道注意力
        x = self.eca(x)

        # 空间注意力
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        conv_out = self.conv2(x)

        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out, conv_out], dim=1)))
        x = spatial_out * x
        return x


class UpSample(nn.Module):
    def __init__(self, x):
        super(UpSample, self).__init__()
        batch_size, channels, height, width = x.shape
        self.height = height
        self.width = width
        self.in_ch = channels
        self.out_ch = int(channels / 2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_ch,
                      self.out_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = F.interpolate(x, size=(self.height * 2, self.width * 2), mode="bilinear")
        out1 = self.conv_block(out1)
        return out1


class CrossAttention(nn.Module):
    def __init__(self, input_ch, output_ch, head_n=4):
        super(CrossAttention, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.softmax = nn.Softmax(dim=-1)
        self.headn = head_n
        self.conv1 = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(self.output_ch, self.output_ch, kernel_size=1, stride=1)

    def forward(self, input_tensor1, input_tensor2):
        batch_size, channels, height, width = input_tensor2.shape
        self_k = self.conv1(input_tensor1).view(batch_size * self.headn, height * width, -1)
        # print(self_k.shape)
        # (bs*n,h*w,c/n)
        self_q = self.conv2(input_tensor1).view(batch_size * self.headn, height * width, -1).permute(0, 2, 1)
        # print(self_q.shape)
        # permute()函数对维度进行重排，将第二个和第三个维度进行置换。
        # (bs*n,c/n,h*w)
        self_v = self.conv3(input_tensor2).view(batch_size * self.headn, height * width, -1)
        # (bs*n,h*w,c/n)
        # print(self_v.shape)

        kq = torch.matmul(self_k, self_q)
        # print("kq", kq.shape)
        # (bs*n,h*w,h*w)
        kq = kq.view(batch_size * self.headn, -1)

        kqs = self.softmax(kq)
        kqs = kqs.view(batch_size * self.headn, -1, height * width)
        kqv = torch.matmul(kqs, self_v)
        # (bs*n,c/n,h*w)
        kqv = kqv.view(batch_size, channels, height, width)
        # (bs,c,h,w)
        out = self.conv4(kqv)
        return out


class Merge(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Merge, self).__init__()
        self.input_ch = in_channel
        self.output_ch = out_channel
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.input_ch,
                      self.output_ch,
                      kernel_size=3,
                      stride=1,
                      padding='same',
                      bias=False),
            nn.BatchNorm2d(self.output_ch),
            nn.ReLU()
        )
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.output_ch,
                      self.output_ch,
                      kernel_size=3,
                      stride=1,
                      padding='same',
                      bias=False),
            nn.BatchNorm2d(self.output_ch),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            REBNCONV(self.output_ch, self.output_ch * 2, dirate=1, kernel_size=1),
            REBNCONV(self.output_ch * 2, self.output_ch * 2, dirate=1, kernel_size=1),
            REBNCONV(self.output_ch * 2, self.output_ch * 2, dirate=1, kernel_size=1),
            REBNCONV(self.output_ch * 2, self.output_ch * 2, dirate=1, kernel_size=1),
            REBNCONV(self.output_ch * 2, self.output_ch * 2, dirate=1, kernel_size=1),
            REBNCONV(self.output_ch * 2, self.output_ch, dirate=1, kernel_size=1),
        )

        # self.attention = SelfAttention(self.output_ch, self.output_ch)

    def forward(self, input_tensor):
        out1 = self.conv_block(input_tensor)
        # out = self.attention(out)
        out1 = self.conv_block1(out1)
        out1 = self.conv_block2(out1)
        out1 = out1 + input_tensor

        return out1


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, kernel_size=3):
        super(REBNCONV, self).__init__()
        self.change = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding='same', dilation=1 * dirate)
        self.conv_s2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding='same', dilation=1 * dirate)
        self.bn_s2 = nn.BatchNorm2d(out_ch)
        self.relu_s2 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.change(x)

        hx = self.conv_s1(hx)
        hx = self.conv_s2(hx)
        xout = self.relu_s2(self.bn_s2(hx))
        xout = xout+hxin

        return xout


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src


# RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch,kernel_size=7, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch,kernel_size=7, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, kernel_size=5,dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
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
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch,kernel_size=5, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch,kernel_size=7, dirate=1)
        self.cbam_module = CBAMLayer(out_ch)

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

        self.rebnconvin = REBNCONV(in_ch, out_ch,kernel_size=7, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch,kernel_size=5, dirate=1)
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
        self.cbam_module = CBAMLayer(out_ch)

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

        self.rebnconvin = REBNCONV(in_ch, out_ch, kernel_size=7,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch,kernel_size=5, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        # self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch , mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch,kernel_size=5, dirate=1)
        self.cbam_module = CBAMLayer(out_ch)

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
        self.cbam_module = CBAMLayer(out_ch)

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


class Inception(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inception, self).__init__()

        mid_channel = int(out_channel / 8)
        self.change1 = REBNCONV(mid_channel, out_channel, kernel_size=1)
        self.change = REBNCONV(in_channel, mid_channel, kernel_size=1)

        # self.branch1x1 = REBNCONV(mid_channel, mid_channel, kernel_size=1)  # 输入外界通道数，经过1*1的卷积核，输出通道16
        # self.branch1x1_1 = REBNCONV(mid_channel, mid_channel, kernel_size=1)  # 输入外界通道数，经过1*1的卷积核，输出通道16

        self.branch5x5_1 = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=1)
        self.branch5x5_2 = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=1)
        self.branch5x5_3 = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=1)
        self.branch5x5_4 = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=1)

        self.branch5x5_1d = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=2)  # 输入外界通道数，经过1*1的卷积核，输出通道16
        self.branch5x5_2d = REBNCONV(mid_channel, mid_channel, kernel_size=3, dirate=2)
        # 连接branch5x5_1，输入通道16，输出通道24，经过5*5的卷积，padding=2表示在输入图像的每边填充2个像素，使得输出特征图与输入特征图的尺寸相同

        self.branch3x3_1 = REBNCONV(mid_channel, mid_channel, kernel_size=1)  # 输入外界通道数，经过1*1的卷积核，输出通道16
        self.branch3x3_2 = REBNCONV(mid_channel, mid_channel, kernel_size=3)
        # 连接branch3x3_1，输入通道16，输出通道24，经过3*3的卷积，padding=1表示在输入图像的每边填充1个像素，使得输出特征图与输入特征图的尺寸相同
        self.branch3x3_3 = REBNCONV(mid_channel, mid_channel, kernel_size=3)
        # 连接branch3x3_2，输入通道24，经过3*3的卷积，padding=1表示在输入图像的每边填充1个像素，使得输出特征图与输入特征图的尺寸相同

        self.batch_pool = REBNCONV(mid_channel, mid_channel, kernel_size=1)  # 输入外界通道数，经过1*1的卷积核，输出通道24

        self.conv_block1 = REBNCONV(4 * mid_channel, out_channel, kernel_size=3)
        self.conv_block2 = REBNCONV(out_channel, out_channel, kernel_size=1)
        self.cbam = CBAMLayer(out_channel)

    def forward(self, x):
        x = self.change(x)
        mx = self.change1(x)

        # # 处理路径1
        # branch1 = self.branch1x1(x)
        # branch1 = self.branch1x1_1(branch1)
        # 处理路径2
        branch5 = self.branch5x5_1d(x)
        branch5 = self.branch5x5_2d(branch5)
        # 处理路径3
        branch3 = self.branch3x3_1(x)
        branch3 = self.branch3x3_2(branch3)
        branch3 = self.branch3x3_3(branch3)
        # 处理路径4
        batch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        batch_pool = self.batch_pool(batch_pool)  # 平均池化，操作后图像尺寸不变
        # 处理路径5
        branch7 = self.branch5x5_1(x)
        branch7 = self.branch5x5_2(branch7)
        branch7 = self.branch5x5_3(branch7)
        branch7 = self.branch5x5_4(branch7)

        # 4条路径进行融合
        cats = [branch7, branch5, branch3, batch_pool]  # cats包含来自4条路径的4个特征图，他们具有相同的高度和宽度，不同的通道数
        output = torch.cat(cats, dim=1)  # torch.cat()用于在指定维度上连接多个张量，此处将四个特征图在通道数这个维度上进行连接起来
        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.cbam(output)
        output = output + mx

        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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

        self.stage5 = Inception(128, 128)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = Inception(128, 128)

        # decoder
        self.stage5d = Inception(256,128)
        self.stage4d = RSU4(256, 32, 64)
        self.stage3d = RSU5(128, 16, 32)
        self.stage2d = RSU6(64, 8, 16)
        self.stage1d = RSU7(32, 8, 16)
        # merge
        self.merge1 = Merge(16,16)
        self.merge2 = Merge(32, 32)
        self.merge3 = Merge(64,64)
        self.merge4 = Merge(128, 128)
        self.merge5 = Merge(128, 128)

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
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx5 = self.merge5(hx5)

        # stage 6
        hxd = self.stage6(hx)
        hxd = _upsample_like(hxd, hx5)

        # -------------------- decoder --------------------
        hxd= self.stage5d(torch.cat((hxd, hx5), 1))
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


# U^2-Net small ###
class MFEUNET(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(MFEUNET, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = Inception(64, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = Inception(64, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = Inception(64, 64)

        # merge
        self.merge1 = Merge(64, 64)
        self.merge2 = Merge(64, 64)
        self.merge3 = Merge(64, 64)
        self.merge4 = Merge(64, 64)
        self.merge5 = Merge(64, 64)

        # decoder
        self.stage5d = Inception(128, 64)
        self.stage4d = Inception(128, 64)
        self.stage3d = RSU5(128, 32, 64)
        self.stage2d = RSU6(128, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        # self.attention = CrossAttention(64, 64)
        # self.attention1 = CrossAttention(64, 64)

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
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx5 = self.merge5(hx5)

        # stage 6
        hxd = self.stage6(hx)
        hxd = _upsample_like(hxd, hx5)

        # decoder
        # hx6up = self.attention(hx6up,hx5)
        hxd = self.stage5d(torch.cat((hxd, hx5), 1))
        hxd = _upsample_like(hxd, hx4)
        # hx5dup = self.attention1(hx5dup,hx4)

        hxd = self.stage4d(torch.cat((hxd, hx4), 1))
        hxd = _upsample_like(hxd, hx3)

        hxd = self.stage3d(torch.cat((hxd, hx3), 1))
        hxd = _upsample_like(hxd, hx2)

        hxd = self.stage2d(torch.cat((hxd, hx2), 1))
        hxd = _upsample_like(hxd, hx1)

        hxd = self.stage1d(torch.cat((hxd, hx1), 1))

        hxd = self.side1(hxd)

        return F.sigmoid(hxd)


if __name__ == '__main__':
    infrared_model = U2NET().cuda()
    summary(infrared_model, input_size=(1, 512,512), device='cuda')


