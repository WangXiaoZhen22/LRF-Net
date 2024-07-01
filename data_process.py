import glob
from pathlib import Path
import numpy as np
import cv2
import os

from skimage import measure


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


if __name__ == '__main__':
    j = 0

    # img_dir = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/images')
    # mask_dir = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/masks')
    # images, labels = get_png_and_labels(img_dir, mask_dir)
    images = []
    labels = []
    image_path = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/images')
    label_path = os.path.join('/home/WangXiaoZhen/dataset/sea_sirst/masks')
    with open('/home/WangXiaoZhen/dataset/sea_sirst/train.txt', 'r',
              encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n').rjust(5, '0')  # 去除文本中的换行符
            ann = ann + '.png'
            images.append(os.path.join(image_path, ann))
            labels.append(os.path.join(label_path, ann))
    img_save_dir = os.path.join('/home/WangXiaoZhen/dataset/sirst_sea_512/train/images')
    mask_save_dir = os.path.join('/home/WangXiaoZhen/dataset/sirst_sea_512/train/masks')
    for i in range(len(labels)):
        image = cv2.imread(images[i])
        label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        label_sum = np.sum(label)
        labelss = label
        labelss = measure.label(labelss, connectivity=2)
        # 标记输入的图像或数组中相互连接的像素块
        coord_label = measure.regionprops(labelss)
        # 函数会计算每个标记区域的属性
        if len(coord_label) == 0 or image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            continue
        else:
            # image = cv2.resize(image, (1024, 1024))
            # label = cv2.resize(label, (1024, 1024))
            if label.shape == (1024, 1024):
                for x in range(8):
                    for y in range(8):
                        mask_crop = label[y * 128:y * 128 + 512, x * 128:x * 128 + 512]

                        if np.sum(mask_crop) == 0 or mask_crop.shape != (512, 512):
                            continue
                        else:
                            image_crop = image[y * 128:y * 128 + 512, x * 128:x * 128 + 512]
                            print(image_crop.shape)

                            image_path = os.path.join(img_save_dir, "%06d.png" % j)
                            cv2.imwrite(image_path, image_crop)
                            mask_path = os.path.join(mask_save_dir, "%06d.png" % j)
                            cv2.imwrite(mask_path, mask_crop)
                            j = j + 1
                            print(j)
            if label.shape == (740, 1024):
                image = cv2.resize(image, (256 * 3, 1024))
                label = cv2.resize(label, (256 * 3, 1024))
                for x in range(8):
                    for y in range(6):
                        mask_crop = label[y * 128:y * 128 + 512, x * 128:x * 128 + 512]
                        # 第一个范围表示高度 第二个范围表示宽度
                        if np.sum(mask_crop) == 0 or mask_crop.shape != (512, 512):
                            continue
                        else:
                            image_crop = image[y * 128:y * 128 + 512, x * 128:x * 128 + 512]
                            image_path = os.path.join(img_save_dir, "%06d.png" % j)
                            cv2.imwrite(image_path, image_crop)
                            mask_path = os.path.join(mask_save_dir, "%06d.png" % j)
                            cv2.imwrite(mask_path, mask_crop)
                            j = j + 1
                            print(j)
            if label.shape == (1024, 740):
                image = cv2.resize(image, (1024, 256 * 3))
                label = cv2.resize(label, (1024, 256 * 3))
                for x in range(6):
                    for y in range(8):
                        mask_crop = label[y * 128:y * 128 + 512, x * 128:x * 128 + 512]
                        # 第一个范围表示高度 第二个范围表示宽度
                        if np.sum(mask_crop) == 0 or mask_crop.shape != (512, 512):
                            continue
                        else:
                            image_crop = image[y * 128:y * 128 + 512, x * 128:x * 128 + 512]
                            image_path = os.path.join(img_save_dir, "%06d.png" % j)
                            cv2.imwrite(image_path, image_crop)
                            mask_path = os.path.join(mask_save_dir, "%06d.png" % j)
                            cv2.imwrite(mask_path, mask_crop)
                            j = j + 1
                            print(j)
            if label.shape == (740, 740):
                image = cv2.resize(image, (256 * 3, 256 * 3))
                label = cv2.resize(label, (256 * 3, 256 * 3))
                for x in range(6):
                    for y in range(6):
                        mask_crop = label[y * 128:y * 128 + 512, x * 128:x * 128 + 512]
                        # 第一个范围表示高度 第二个范围表示宽度
                        if np.sum(mask_crop) == 0 or mask_crop.shape != (512, 512):
                            continue
                        else:
                            image_crop = image[y * 128:y * 128 + 512, x * 128:x * 128 + 512]
                            image_path = os.path.join(img_save_dir, "%06d.png" % j)
                            cv2.imwrite(image_path, image_crop)
                            mask_path = os.path.join(mask_save_dir, "%06d.png" % j)
                            cv2.imwrite(mask_path, mask_crop)
                            j = j + 1
                            print(j)
