import glob
from abc import ABC

import numpy as np
from PIL import Image as pilimage
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import torch.nn.functional as F
import torch
import cv2
import os
import xml.etree.ElementTree as ET
import albumentations as A

aug = A.Compose([
    A.HorizontalFlip(always_apply=False, p=0.15),  # 围绕y轴水平翻转输入。
    A.VerticalFlip(always_apply=False, p=0.15),  # 围绕X轴垂直翻转输入。
    A.RandomRotate90(always_apply=False, p=0.15),  # 将输入随机旋转90度，零次或多次。
    A.Rotate(always_apply=False, p=0.15),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.15)
])


def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    image_path = dir_path + r'\Images'

    label_path = dir_path + r'\Annotation'
    image_path = Path(image_path)
    label_path = Path(label_path)
    # 将一个路径转换为Path对象
    classes = []  # 类别名列表
    # 创建了一个空列表 classes
    for category in image_path.iterdir():
        # iterdir() 方法来迭代遍历 dir_path 目录中的每个项目（文件或子目录）
        if category.is_dir():
            # 检测 category 是否是一个目录
            classes.append(category.name)
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for index, name in enumerate(classes):
        class_path = label_path / name
        if not class_path.is_dir():
            continue
        for xml_path in class_path.glob('*.xml'):
            in_file = open(xml_path, 'r')
            tree = ET.parse(in_file)
            # 使用 ElementTree 的 parse() 方法解析 in_file 文件对象，得到一个 XML 树对象，并将其赋给 tree 变量。
            root = tree.getroot()
            img_path = root.find('path').text
            images_list.append(str(img_path))
            labels_list.append(str(xml_path))
    return images_list, labels_list


class TrackingDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path  # 数据集根目录
        self.transform = transform
        self.images, self.labels = get_images_and_labels(self.dir_path)

    def get_label(self, index):
        xml_path = self.labels[index]
        in_file = open(xml_path, 'r')
        tree = ET.parse(in_file)
        # 使用 ElementTree 的 parse() 方法解析 in_file 文件对象，得到一个 XML 树对象，并将其赋给 tree 变量。
        root = tree.getroot()
        xml_box = []
        target_vector = [0] * 480 * 640
        if root.find('object') is None:
            xml_box.append((0, 0, 0, 0))
        for obj in tree.iter("object"):
            xmlbox = obj.find('bndbox')
            xmin = int(float(xmlbox.find('xmin').text))
            ymin = int(float(xmlbox.find('ymin').text))
            xmax = int(float(xmlbox.find('xmax').text))
            ymax = int(float(xmlbox.find('ymax').text))
            b = (xmin, ymin, xmax, ymax)
            xml_box.append(b)
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    position = j * 480 + i
                    target_vector[position] = 1
        target_vector = torch.tensor(target_vector)
        target_vector = target_vector.view(1, 480, 640)
        target_vector = target_vector.to(torch.float32)
        lenth = len(xml_box)
        # xml_box = torch.tensor(xml_box)
        # xml_box = xml_box.view(lenth, 4)
        # xml_box = F.pad(xml_box, pad=(0, 0, 0, 10 - lenth))

        return target_vector

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.get_label(index)
        # 根据当前目录和图片路径，得到完整的图片路径
        full_path = os.path.join('track_dataset', img_path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        # 创建张量
        tensor = torch.from_numpy(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)
        return img, label


def get_png_and_labels(image_path, label_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param label_path:
    :param image_path:
    :return: images_list, labels_list
    '''

    image_path = Path(image_path)
    label_path = Path(label_path)
    # 将一个路径转换为Path对象
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表
    # 构建一个搜索模式，用于查找所有的PNG图像文件
    search_image_pattern = os.path.join(image_path, "*.png")
    # 使用glob模块查找匹配的文件
    image_png_files = glob.glob(search_image_pattern)

    # 打印所有找到的PNG图像文件路径
    for image_file_path in image_png_files:
        images_list.append(str(image_file_path))
    # 构建一个搜索模式，用于查找所有的PNG图像文件
    search_label_pattern = os.path.join(label_path, "*.png")
    # 使用glob模块查找匹配的文件
    label_png_files = glob.glob(search_label_pattern)

    # 打印所有找到的PNG图像文件路径
    for label_file_path in label_png_files:
        labels_list.append(str(label_file_path))

    return images_list, labels_list


class IRSTD_1KDataset(Dataset):
    def __init__(self, mode):
        image_path = os.path.join('/home/WangXiaoZhen/dataset/IRSTD-1k/IRSTD1k_Img')
        label_path = os.path.join('/home/WangXiaoZhen/dataset/IRSTD-1k/IRSTD1k_Label')
        self.images = []
        self.labels = []
        self.mode = mode
        if mode == 'test':
            with open('/home/WangXiaoZhen/dataset/IRSTD-1k/test.txt', 'r', encoding='utf-8') as f:
                for ann in f.readlines():
                    ann = ann.strip('\n')  # 去除文本中的换行符
                    self.images.append(os.path.join(image_path, ann))
                    self.labels.append(os.path.join(label_path, ann))
        elif mode == 'train':
            self.images, self.labels = get_png_and_labels(image_path, label_path)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'train':
            image = cv2.resize(image, (512,512))
            label = cv2.resize(label, (512,512))
            data = aug(image=image, mask=label)
            image = data['image']
            label = data['mask']

        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        image = torch.unsqueeze(image, dim=0)
        label = torch.unsqueeze(label, dim=0)

        return image, label


class sirst_mlcl(Dataset):
    def __init__(self, mode):
        if mode == 'test':
            image_path = os.path.join('/home/WangXiaoZhen/dataset/sirst_mlcl/test/image')
            label_path = os.path.join('/home/WangXiaoZhen/dataset/sirst_mlcl/test/mask')
        elif mode == 'train':
            image_path = os.path.join('/home/WangXiaoZhen/dataset/sirst_mlcl/train/image')
            label_path = os.path.join('/home/WangXiaoZhen/dataset/sirst_mlcl/train/mask')
        else:
            raise NotImplementedError
        self.images, self.labels = get_png_and_labels(image_path, label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (320, 320))
        # label = cv2.resize(label, (320, 320))

        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        image = torch.unsqueeze(image, dim=0)
        label = torch.unsqueeze(label, dim=0)

        return image, label


class SIRSTDataset(Dataset):
    def __init__(self, mode):
        image_path = os.path.join('/home/WangXiaoZhen/dataset/sirst-master/images')
        label_path = os.path.join('/home/WangXiaoZhen/dataset/sirst-master/masks')
        self.mode = mode
        self.images = []
        self.labels = []
        if mode == 'test':
            with open('/home/WangXiaoZhen/dataset/sirst-master/test.txt', 'r', encoding='utf-8') as f:
                for ann in f.readlines():
                    ann = ann.strip('\n')  # 去除文本中的换行符
                    self.images.append(os.path.join(image_path, ann))
                    self.labels.append(os.path.join(label_path, ann))
        elif mode == 'train':
            with open('/home/WangXiaoZhen/dataset/sirst-master/train.txt', 'r', encoding='utf-8') as f:
                for ann in f.readlines():
                    ann = ann.strip('\n')  # 去除文本中的换行符
                    self.images.append(os.path.join(image_path, ann))
                    self.labels.append(os.path.join(label_path, ann))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'train':
            image = cv2.resize(image, (320, 320))
            label = cv2.resize(label, (320, 320))
            data = aug(image=image, mask=label)
            image = data['image']
            label = data['mask']

        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        image = torch.unsqueeze(image, dim=0)
        label = torch.unsqueeze(label, dim=0)
        return image, label


class NUDT_SIRSTDataset(Dataset):
    def __init__(self, mode):

        self.mode = mode
        self.images = []
        self.labels = []
        if mode == 'test':
            image_path = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/images')
            label_path = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/masks')
            with open('/home/WangXiaoZhen/dataset/sea_sirst/test.txt', 'r',
                      encoding='utf-8') as f:
                for ann in f.readlines():
                    ann = int(ann.strip('\n'))  # 去除文本中的换行符
                    ann = '%05d' % ann
                    ann = ann + '.png'
                    self.images.append(os.path.join(image_path, ann))
                    self.labels.append(os.path.join(label_path, ann))
        elif mode == 'train':
            img_dir = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/images')
            mask_dir = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/masks')
            self.images, self.labels = get_png_and_labels(img_dir, mask_dir)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lab_path = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'train':
            data = aug(image=image, mask=label)
            image = data['image']
            label = data['mask']

        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        image = torch.unsqueeze(image, dim=0)
        label = torch.unsqueeze(label, dim=0)
        return image, label


class sea_256(Dataset):
    def __init__(self, mode):

        self.mode = mode
        self.images = []
        self.labels = []
        if mode == 'test':
            img_dir = os.path.join('/home/WangXiaoZhen/dataset/sirst_sea_256/test/images')
            mask_dir = os.path.join('/home/WangXiaoZhen/dataset/sirst_sea_256/test/masks')
            self.images, self.labels = get_png_and_labels(img_dir, mask_dir)

        elif mode == 'train':
            img_dir = os.path.join('/home/WangXiaoZhen/dataset/sirst_sea_256/train/images')
            mask_dir = os.path.join('/home/WangXiaoZhen/dataset/sirst_sea_256/train/masks')
            self.images, self.labels = get_png_and_labels(img_dir, mask_dir)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lab_path = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'train':
            data = aug(image=image, mask=label)
            image = data['image']
            label = data['mask']

        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        image = torch.unsqueeze(image, dim=0)
        label = torch.unsqueeze(label, dim=0)
        return image, label


# 定义dataset
class G1G2Dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.imageset_dir = os.path.join('/home/WangXiaoZhen/dataset/data/training')
            self.imageset_gt_dir = os.path.join('/home/WangXiaoZhen/dataset/data/training')
        elif self.mode == 'test':
            self.imageset_dir = os.path.join('/home/WangXiaoZhen/dataset/data/test_org')
            self.imageset_gt_dir = os.path.join('/home/WangXiaoZhen/dataset/data/test_gt')
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode == 'train':
            return 10000
        elif self.mode == 'test':
            return 100
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_dir = os.path.join(self.imageset_dir, "%06d_1.png" % idx)
            gt_dir = os.path.join(self.imageset_gt_dir, "%06d_2.png" % idx)
            # 一个占位符，用于表示一个十进制整数，最小宽度为6位。如果数字少于6位数，将会在前面添加零。

            image = cv2.imread(img_dir, cv2.IMREAD_ANYDEPTH)
            label = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)

            if image is None or label is None:
                img_dir = os.path.join(self.imageset_dir, "%06d_1.png" % (idx + 1))
                gt_dir = os.path.join(self.imageset_gt_dir, "%06d_2.png" % (idx + 1))
                image = cv2.imread(img_dir, cv2.IMREAD_ANYDEPTH)
                label = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)

            image = torch.from_numpy(image).float() / 255.0
            image = image.view(1, 128, 128)

            # _, label = cv2.threshold(label, 2, 1, cv2.THRESH_BINARY)

            label = torch.from_numpy(label).float() / 255.0
            label = label.view(1, 128, 128)
            return image, label

        elif self.mode == 'test':
            img_dir = os.path.join(self.imageset_dir, "%05d.png" % idx)
            gt_dir = os.path.join(self.imageset_gt_dir, "%05d.png" % idx)

            image = cv2.imread(img_dir, cv2.IMREAD_ANYDEPTH)
            label = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)

            if image is None or label is None:
                img_dir = os.path.join(self.imageset_dir, "%05d.png" % (idx + 1))
                gt_dir = os.path.join(self.imageset_gt_dir, "%05d.png" % (idx + 1))
                image = cv2.imread(img_dir, cv2.IMREAD_ANYDEPTH)
                label = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)

            image = torch.from_numpy(image).float() / 255.0
            image = torch.unsqueeze(image, dim=0)

            _, label = cv2.threshold(label, 2, 1, cv2.THRESH_BINARY)
            label = torch.from_numpy(label).float()
            label = torch.unsqueeze(label, dim=0)

            return image, label
        else:
            raise NotImplementedError


if __name__ == '__main__':
    x = torch.ones(16, 1, 512, 512)

    print(x.shape)

    # image_path = os.path.join('/home/WangXiaoZhen/dataset/NUDT-SIRST/images')
    # label_path = os.path.join('/home/WangXiaoZhen/dataset/NUDT-SIRST/masks')
    # img_path = os.path.join('/home/WangXiaoZhen/dataset/NUDT-SIRST/train/images')
    # mask_path = os.path.join('/home/WangXiaoZhen/dataset/NUDT-SIRST/train/masks')
    #
    # with open('/home/WangXiaoZhen/dataset/NUDT-SIRST/idx_4961+847/train.txt', 'r', encoding='utf-8') as f:
    #     i = 0
    #     for ann in f.readlines():
    #         ann = ann.strip('\n') + '.png'  # 去除文本中的换行符
    #         image_pa = (os.path.join(image_path, ann))
    #         label_pa = (os.path.join(label_path, ann))
    #         image = pilimage.open(image_pa)
    #         label = pilimage.open(label_pa)
    #         region1 = image.resize((512,512))
    #         region2 = label.resize((512,512))
    #         region1.save(os.path.join(img_path, "%05d.png" % i))
    #         region2.save(os.path.join(mask_path, "%05d.png" % i))
    #         i = i + 1
    #         # w, h = image.size
    #         # if w == h == 1024:
    #         #     region1 = image.crop((0, 0, 512, 512))  # 左、上、右、下
    #         #     region2 = image.crop((512, 0, 1024, 512))
    #         #     region3 = image.crop((0, 512, 512, 1024))
    #         #     region4 = image.crop((512, 512, 1024, 1024))
    #         #     region5 = label.crop((0, 0, 512, 512))  # 左、上、右、下
    #         #     region6 = label.crop((512, 0, 1024, 512))
    #         #     region7 = label.crop((0, 512, 512, 1024))
    #         #     region8 = label.crop((512, 512, 1024, 1024))
    #         #     region1 = region1.resize((256,256))
    #         #     region2 = region2.resize((256,256))
    #         #     region3 = region3.resize((256,256))
    #         #     region4 = region4.resize((256,256))
    #         #     region5 = region5.resize((256,256))
    #         #     region6 = region6.resize((256,256))
    #         #     region7 = region7.resize((256,256))
    #         #     region8 = region8.resize((256,256))
    #         #     region1.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region5.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #     region2.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region6.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #     region3.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region7.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #     region4.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region8.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         # if w == 740 & h == 1024:
    #         #     region1 = image.resize((256,256))
    #         #     region2 = label.resize((256,256))
    #         #     region1.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region2.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #     region1 = image.crop((0, 0, 740, 512))  # 左、上、右、下
    #         #     region2 = image.crop((0, 512, 740, 1024))
    #         #     region3 = label.crop((0, 0, 740, 512))  # 左、上、右、下
    #         #     region4 = label.crop((0, 512, 740, 1024))
    #         #     region1 = region1.resize((256,256))
    #         #     region2 = region2.resize((256,256))
    #         #     region3 = region3.resize((256,256))
    #         #     region4 = region4.resize((256,256))
    #         #     region1.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region3.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #     region2.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region4.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #
    #         # if w == 1024 & h == 740:
    #         #     region1 = image.resize((256,256))
    #         #     region2 = label.resize((256,256))
    #         #     region1.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region2.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #     region1 = image.crop((0, 0, 512, 740))  # 左、上、右、下
    #         #     region2 = image.crop((512, 0, 1024, 740))
    #         #     region3 = label.crop((0, 0, 512, 740))  # 左、上、右、下
    #         #     region4 = label.crop((512, 0, 1024, 740))
    #         #     region1 = region1.resize((256,256))
    #         #     region2 = region2.resize((256,256))
    #         #     region3 = region3.resize((256,256))
    #         #     region4 = region4.resize((256,256))
    #         #     region1.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region3.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         #     region2.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region4.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         # if w == h == 740:
    #         #     region1 = image.resize((256,256))
    #         #     region2 = label.resize((256,256))
    #         #     region1.save(os.path.join(img_path, "%05d.png" % i))
    #         #     region2.save(os.path.join(mask_path, "%05d.png" % i))
    #         #     i = i + 1
    #         print(i)
    f1 = open("/home/WangXiaoZhen/dataset/IRSTD-1k/test.txt", "w")

    list1 = list(range(0, 1001))
    l1 = random.sample(range(0, 1001), 100)
    l2 = list(set(list1) - set(l1))
    for i in range(0, len(l1)):
        k = str(l1[i])
        k = 'XDU' + k.zfill(5)
        print(k)
        f1.write(k + '\n')
    f1.close()
    f2 = open("/home/WangXiaoZhen/dataset/IRSTD-1k/train.txt", "w")
    for i in range(0, len(l2)):
        k = str(l2[i])
        k = 'XDU' + k.zfill(5)
        print(k)
        f2.write(k + '\n')
    f2.close()
