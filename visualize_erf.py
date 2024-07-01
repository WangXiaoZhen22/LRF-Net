# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import argparse

import numpy as np
import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import AverageMeter
from torch import optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import SIRSTDataset, IRSTD_1KDataset, G1G2Dataset, NUDT_SIRSTDataset
from model import UNET,ResBlock


def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='resnet101', type=str, help='model name')
    parser.add_argument('--weights', default=None, type=str, help='path to weights file. For resnet101/152, ignore '
                                                                  'this arg to download from torchvision')
    parser.add_argument('--data_path', default='path_to_imagenet', type=str, help='dataset path')
    parser.add_argument('--save_path', default='temp.npy', type=str, help='path to save the ERF matrix (.npy file)')
    parser.add_argument('--num_images', default=50, type=int, help='num of images to use')
    args = parser.parse_args()
    return args


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


def main(args):
    #   ================================= transform: resize to 1024x1024
    t = [
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    dataset1 = SIRSTDataset('train')
    dataset3 = G1G2Dataset('train')
    dataset2 = NUDT_SIRSTDataset('train')
    dataset5 = IRSTD_1KDataset('train')

    test_dataset = dataset5
    # 创建DataLoader来批量加载数据
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_blocks = [6, 6, 6, 6, 6]
    # model = UNET_CBAM(num_blocks=num_blocks, max_ks=11).cuda()
    model = ResBlock(1, 16, max_ks=11, layer_num=6)
    # checkpoint = torch.load('./checkpoint/ckpt_best_mdfa_11.pth', map_location=device)
    # model.load_state_dict(checkpoint['parameter'])
    model.cuda()
    model.eval()  # fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for _, (samples, _) in enumerate(test_dataloader):

        if meter.count == args.num_images:
            np.save(args.save_path, meter.avg)
            exit()

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)


if __name__ == '__main__':
    args = parse_args()
    main(args)
