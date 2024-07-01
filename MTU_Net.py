import torch
import torch.nn as nn
from torchsummary import summary

from vit import ViT

from einops import rearrange


class _FCNHead(nn.Module):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(inter_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=inter_channels, out_channels=channels, kernel_size=1)
        )

    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.block(x)


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
        hx = self.bn_s1(hx)
        hx = self.relu_s1(hx)
        hx = self.conv_s2(hx)
        hx = self.bn_s2(hx)
        hx = self.relu_s2(hx)
        hx = hx + x
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
                layers.append(REBNCONV(out_ch, out_ch, kernel_size=3, dirate=1))
        self.convblock = nn.Sequential(*layers)

        self.trans = Transfomer(out_ch, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx = self.convblock(hxin)
        hx = self.trans(hx)
        return hx + hxin


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
    def __init__(self, in_channel, out_channel=64, mlp_ratio=2, img_size=1024):
        super(Transfomer, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.img_size = img_size

        self.change = nn.Conv2d(in_channels=self.in_ch, out_channels=out_channel,
                                kernel_size=1)
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


class res_UNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter):
        super(res_UNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0_0 = self._make_layer(block, input_channels, input_channels)
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])

        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])

        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.vit0 = Transfomer(in_channel=nb_filter[0], out_channel=nb_filter[0], mlp_ratio=2)
        self.vit1 = Transfomer(in_channel=nb_filter[1], out_channel=nb_filter[1], mlp_ratio=2)
        self.vit2 = Transfomer(in_channel=nb_filter[2], out_channel=nb_filter[2], mlp_ratio=2)
        self.vit3 = Transfomer(in_channel=nb_filter[3], out_channel=nb_filter[3], mlp_ratio=2)
        self.vit4 = Transfomer(in_channel=nb_filter[4], out_channel=nb_filter[4], mlp_ratio=2)

        # self.conv4_1 = self._make_layer(block, nb_filter[4] + nb_filter[4], nb_filter[4])

        self.conv3_1_1 = self._make_layer(block, nb_filter[4],
                                          nb_filter[4])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], 2)

        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], 2)

        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], 2)

        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0], 2)

        self.head = _FCNHead(nb_filter[0], channels=num_classes, momentum=0.9)

        # self.final2 = nn.BatchNorm2d(num_classes)
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        # x0_0_0 = self.conv0_0_0(input)
        x0_0 = self.conv0_0(input)
        # (4,16,256,256)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # (4,32,128,128)
        x2_0 = self.conv2_0(self.pool(x1_0))
        # (4,64,64,64)
        x3_0 = self.conv3_0(self.pool(x2_0))  # (4,128,32,32)

        out = self.conv4_0(self.pool(x3_0))
        # (4,256,16,16)

        out = self.vit4(out)

        out = self.conv3_1_1(out)

        out = self.conv3_1(torch.cat([self.vit3(x3_0), self.up(out)], 1))

        out = self.conv2_2(torch.cat([self.vit2(x2_0), self.up(out)], 1))

        out = self.conv1_3(torch.cat([self.vit1(x1_0), self.up(out)], 1))

        out = self.conv0_4(torch.cat([self.vit0(x0_0), self.up(out)], 1))

        out = self.final(out)

        return out


if __name__ == '__main__':
    nb_filter = [16, 32, 64, 128, 256]
    num_blocks = [2, 2, 2, 2]
    model = res_UNet(num_classes=1, input_channels=1, block=Res_block, num_blocks=num_blocks,
                     nb_filter=nb_filter).cuda()
    summary(model, input_size=(1, 1024, 1024), device='cuda')
