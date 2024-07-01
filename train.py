import torch

from sklearn.metrics import auc, roc_curve
from torch import optim
from torch.utils.data import random_split
import torch.nn as nn

from metric import mIoU, PD_FA
from model import UNET, U2NET
# from ablation.ab_resnext import UNET_resnext
from ablation.ab_cbam import UNET_CBAM
from dataloader import SIRSTDataset, IRSTD_1KDataset, G1G2Dataset, NUDT_SIRSTDataset, sea_256
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from loss import DiceLoss, FocalLoss


class roc_curve_auc:
    def __init__(self):
        self.pred = torch.ones(1)
        self.gt = torch.ones(1)

    def update(self, pre, label):
        pre = pre.view(-1)
        label = label.view(-1)

        self.pred = torch.concat([self.pred, pre])
        self.gt = torch.concat([self.gt, label])

    def auc_get(self):
        gt = (self.gt.detach().numpy())
        pred = (self.pred.detach().numpy())

        fpr, tpr, thresholds = roc_curve(gt.astype(int), pred)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def reset(self):
        self.pred = torch.ones(1)
        self.gt = torch.ones(1)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    #
    dataset1 = SIRSTDataset('train')
    dataset2 = IRSTD_1KDataset('train')
    dataset3 = G1G2Dataset('train')
    dataset4 = NUDT_SIRSTDataset('train')
    dataset5 = sea_256('train')
    # testset = NUDT_SIRSTDataset('valid')
    # testset = SIRSTDataset('test')
    testset = IRSTD_1KDataset('test')
    print(len(testset))

    train_set = dataset2

    # 创建DataLoader来批量加载数据
    train_dataloader = DataLoader(train_set, batch_size=12, shuffle=True)

    # model = U2NET()

    # num_blocks = [6, 6, 6, 6, 6]
    num_blocks = [8, 8, 8, 8, 8]
    # model = UNET_CBAM(num_blocks=num_blocks, max_ks=11).cuda()
    model = UNET(num_blocks=num_blocks, max_ks=11).cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-04, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # lr(float, 可选) – 学习率（默认：1e-3）
    # betas(Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
    # eps(float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    # weight_decay(float, 可选) – 权重衰减（L2惩罚）（默认: 0）

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    mse_loss = nn.MSELoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6, last_epoch=-1)
    total_times = 5000
    epoch = 0
    best_sum = 0
    total = 0
    accuracy_rate = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备为：", device)
    #
    # checkpoint = torch.load('./checkpoint/ckpt_best_1k_11.pth', map_location=device)
    # model.load_state_dict(checkpoint['parameter'])
    # best_sum = checkpoint['best_sum']

    model.to(device)
    auc_f = roc_curve_auc()
    PD_FA = PD_FA(1, 3)
    mIoU = mIoU(1)

    while epoch <= total_times:

        running_loss = 0.0
        for i, (img, labels) in enumerate(train_dataloader, 0):
            model.train()
            model.to(device)
            data = img.to(device)
            outputs = model(data).to(device)
            labels = labels.to(torch.float32)
            labels = labels.to(device)
            loss = 0.1 * mse_loss(outputs, labels) + dice_loss(outputs, labels) + focal_loss(outputs, labels)

            optimizer.zero_grad()
            # 将梯度置零
            loss.backward()
            # 计算梯度
            optimizer.step()  # 梯度优化
            running_loss += loss.item()
            if i >= 0:
                print(f"正在进行第{epoch + 1}次训练, 第{i + 1}批图片,running_loss={running_loss}")
                running_loss = 0.0
            if i >= 0:
                total_iou = 0
                _, test_dataset = random_split(testset, [len(testset) - 20, 20])
                test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

                with torch.no_grad():
                    model.eval()
                    model.to(device)
                    for j, (imgt, labels_t) in enumerate(test_dataloader, 0):
                        # img = img.to(torch.float32)
                        data = imgt.to(device)
                        outputs = model(data).to('cpu')
                        labels_t = labels_t.to('cpu')
                        if outputs.shape != labels_t.shape:
                            continue
                        auc_f.update(outputs, labels_t)
                        outputs = torch.where(outputs > 0.5, 1, 0)
                        mIoU.update(preds=outputs, labels=labels_t)
                        _, mean_IOU = mIoU.get()
                        total_iou = total_iou + mean_IOU
                        PD_FA.update(outputs, labels=labels_t)
                    _, mean_IOU = mIoU.get()
                    FA, PD, nIoU = PD_FA.get(5)
                    auc_n = auc_f.auc_get()
                    metric_sum = total_iou / (j + 1) + 1.5 * PD[0] - 2 * FA[0] + 0.5 * auc_n
                    if metric_sum > best_sum:
                        best_sum = metric_sum
                        checkpoint = {'parameter': model.state_dict(),
                                      'best_sum': best_sum
                                      }
                        torch.save(checkpoint, './checkpoint/ckpt_best_1k_11_8.pth')
                    auc_f.reset()
                    PD_FA.reset()
                    mIoU.reset()
        scheduler.step()  # 更新学习率调度器

        checkpoint = {'parameter': model.state_dict()}
        torch.save(checkpoint, './checkpoint/ckpt_{}.pth'.format(epoch))
        epoch = epoch + 1
