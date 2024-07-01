from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy import ndimage
import cv2
import torch
from PIL import Image
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
from time import sleep
from model import UNET, U2NET
from ablation.ab_resnext import UNET_resnext
from ablation.ab_cbam import UNET_CBAM

from dataloader import SIRSTDataset, IRSTD_1KDataset, G1G2Dataset, NUDT_SIRSTDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from metric import *


class draw_roc_curve:
    def __init__(self):
        self.pred = torch.ones(1)
        self.gt = torch.ones(1)

    def update(self, pre, label):
        pre = pre.view(-1)
        label = label.view(-1)

        self.pred = torch.concat([self.pred, pre])
        self.gt = torch.concat([self.gt, label])

    def roc_plot(self):
        self.gt = self.gt.detach().numpy()
        self.pred = self.pred.detach().numpy()
        # 2.定义一个画布
        plt.figure(1)
        # 3.计算fpr、tpr及roc曲线的面积
        fpr, tpr, thresholds = roc_curve(self.gt.astype(int), self.pred)
        # np.save('./fpr_sirst.npy', fpr)
        # np.save('./tpr_sirst.npy', tpr)
        # np.save('./fpr.npy', fpr)
        # np.save('./tpr.npy', tpr)
        # np.save('./fpr_1k.npy', fpr)
        # np.save('./tpr_1k.npy', tpr)
        roc_auc = auc(fpr, tpr)
        # 4.绘制roc曲线
        plt.plot(fpr, tpr, label='UNet (area = {:.4f})'.format(roc_auc), color='blue')
        # 5.格式个性化
        font1 = {
            'weight': 'normal',
            'size': 14, }
        plt.xlabel("FPR (False Positive Rate)", font1)
        plt.ylabel("TPR (True Positive Rate)", font1)
        plt.legend(loc="lower right", fontsize=12)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.axis([0, 1, 0, 1])
        plt.title('ROC Curve', font1)

        plt.show()
        self.pred = torch.ones((1))
        self.gt = torch.ones((1))
        print('Done!')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备为：", device)
    num_blocks = [6, 6, 6, 6, 6]
    # model = UNET_CBAM(num_blocks=num_blocks, max_ks=11).cuda()
    model = UNET(num_blocks=num_blocks, max_ks=11).cuda()
    # model = U2NET()
    checkpoint = torch.load('./checkpoint/ckpt_best_mdfa_11.pth', map_location=device)
    model.load_state_dict(checkpoint['parameter'])
    model.to(device)
    model.eval()

    PD_FA = PD_FA(1, 5)
    mIoU = mIoU(1)
    roc_ploter = draw_roc_curve()

    mIoU.reset()
    PD_FA.reset()

    dataset1 = SIRSTDataset('test')
    dataset3 = G1G2Dataset('test')
    dataset2 = NUDT_SIRSTDataset('test')
    dataset5 = IRSTD_1KDataset('test')

    test_dataset = dataset3
    # 创建DataLoader来批量加载数据
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    total_iou = 0
    img_name = []
    # with open('/home/WangXiaoZhen/dataset/IRSTD-1k/test.txt', 'r', encoding='utf-8') as f:
    #     for ann in f.readlines():
    #         ann = ann.strip('\n')   # 去除文本中的换行符
    #         img_name.append(ann)
    # with open('/home/WangXiaoZhen/dataset/sirst-master/test.txt', 'r', encoding='utf-8') as f:
    #     for ann in f.readlines():
    #         ann = ann.strip('\n') # 去除文本中的换行符
    #         img_name.append(ann)
    stime = time.time()
    with torch.no_grad():
        for i, (img, labels) in enumerate(test_dataloader, 0):
            # img = img.to(torch.float32)
            data = img.to(device)
            outputs = model(data).to('cpu')
            labels = labels.to('cpu')
            if outputs.shape != labels.shape:
                continue
            roc_ploter.update(outputs, labels)
            outputs = torch.where(outputs > 0.5, 1, 0)
            # image = torch.squeeze(outputs, dim=[0, 1]).to('cpu')
            # image = image.detach().numpy()
            # image = Image.fromarray((image * 255).astype(np.uint8))
            # result_dir = os.path.join('./1k_result')
            # result_path = os.path.join(result_dir, img_name[i])
            # # result_path = os.path.join(result_dir, "%05d.png" % i)
            # image.save(result_path)

            # print(i,outputs.shape,labels.shape)
            mIoU.update(outputs, labels)
            _, mean_IOU = mIoU.get()

            PD_FA.update(outputs, labels)

            total_iou = total_iou + mean_IOU
            print(i)

        _, mean_IOU = mIoU.get()
        etime = time.time()
        print(etime - stime)
        FA, PD, nIoU = PD_FA.get(100)
        roc_ploter.roc_plot()

        print("平均交并比", mean_IOU, total_iou / (i + 1))
        print("检测率", PD[0])
        print("虚警率", FA[0])
        print("归一化交并比", nIoU[0])
        PD_FA.reset()
        mIoU.reset()
