import cv2
import cv2 as cv
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet
import numpy as np
from model import UNET, REBNCONV, ResDWConv, Transfomer

import matplotlib.pyplot as plt


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
                # layers.append(ResDWConv(out_ch, out_ch, kernel_size=ks, dirate=1))
                layers.append(REBNCONV(out_ch, out_ch, kernel_size=ks, dirate=1))

        self.convblock = nn.Sequential(*layers)
        # self.trans = Transfomer(out_ch, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx = self.convblock(hxin)
        # hx = self.trans(hx)

        return hx + hxin


class K_3(nn.Module):

    def __init__(self, in_ch=3, out_ch=12, max_ks=7, layer_num=30):
        super(K_3, self).__init__()

        self.convin = REBNCONV(in_ch, out_ch, kernel_size=1, dirate=1)
        layers = []
        for i in range(layer_num):
            layers.append(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), padding='same'))
        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx = self.convblock(hxin)
        return hx


class K_5(nn.Module):

    def __init__(self, in_ch=3, out_ch=12, max_ks=7, layer_num=15):
        super(K_5, self).__init__()

        self.convin = REBNCONV(in_ch, out_ch, kernel_size=1, dirate=1)
        layers = []
        for i in range(layer_num):
            layers.append(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(5, 5), padding='same'))
        layers.append(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), padding='same'))
        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx = self.convblock(hxin)
        return hx


class K_7(nn.Module):

    def __init__(self, in_ch=3, out_ch=12, max_ks=7, layer_num=10):
        super(K_7, self).__init__()

        self.convin = REBNCONV(in_ch, out_ch, kernel_size=1, dirate=1)
        layers = []
        for i in range(layer_num):
            layers.append(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(7, 7), padding='same'))

        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx = self.convblock(hxin)
        return hx


class TinyNet(nn.Module):
    def __init__(self, max_ks=11, layer_num=6):
        super(TinyNet, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(16)
        )

        self.tinypart1 = nn.Sequential(
            ResBlock(16, 16, max_ks=max_ks, layer_num=layer_num)
        )

    def forward(self, x):
        c1 = self.c1(x)

        c8 = self.tinypart1(c1)
        return c8


def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


if __name__ == "__main__":
    model = K_3()
    for module in model.modules():
        try:
            nn.init.constant_(module.weight, 0.05)
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
        except Exception as e:
            pass
        if type(module) is nn.BatchNorm2d:
            module.eval()
    x = torch.ones(1, 3, 64, 64, requires_grad=True)
    contribution_scores = get_input_grad(model, x)
    # pred = model(x)
    #
    # grad = torch.zeros_like(pred, requires_grad=True)
    # with torch.no_grad():
    #     grad[0, 0, 3, 3] = 1
    #
    # pred.backward(gradient=grad)
    # grad_input = x.grad[0, 0, ...].data.numpy()
    # grad_input = grad_input / np.max(grad_input)
    # # 有效感受野 0.75 - 0.85
    # # grad_input = np.where(grad_input>0.85,1,0)
    # grad_input = np.where(grad_input>0.75,1,0)
    # # 注释掉即为感受野
    # grad_input = (grad_input * 255).astype(np.uint8)
    # kernel = np.ones((5, 5), np.uint8)
    # grad_input = cv.dilate(grad_input, kernel=kernel) # 膨胀

    # contours, _ = cv.findContours(grad_input, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE) # 轮廓检测
    # rect = cv.boundingRect(contours[0]) # 包围轮廓的最小矩形
    # print(rect[-2:])
    # save_image_path =
    # cv2.imwrite(os.path.join(save_image_path, 'vertira.jpg'), image)

    plt.figure("Image")  # 图像窗口名称
    plt.imshow(contribution_scores)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.savefig('./k_3.svg')

    plt.show()

    # 必须有这个，要不然无法显示

    # cv2.imwrite('./erf.png', grad_input)
    # cv.imshow("a", grad_input)
    # cv.waitKey(0)
