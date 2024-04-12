#色彩增强
import cv2
import numpy as np
import math
import os
def stretchImage(data, s=0.005, bins=2000):  # 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)

g_para = {}

def getPara(radius=5):  # 根据半径计算权重参数矩阵
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


def zmIce(I, ratio=4, radius=300):  # 常规的ACE实现
    para = getPara(radius)
    height, width = I.shape
    zh, zw = [0] * radius + [x for x in range(height)] + [height - 1] * radius, [0] * radius + [x for x in range(width)] + [width - 1] * radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


def zmIceFast(I, ratio, radius):  # 单通道ACE快速增强实现
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, ((width + 1) // 2, (height + 1) // 2))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


def zmIceColor(I, ratio=4, radius=3):  # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res


if __name__ == '__main__':
    # 获取指定路径下的所有图片文件
    image_folder_path = r'C:\Users\123\Desktop\yolo\data_collect\val\images'
    image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

    # 遍历每张图片进行色彩增强并保存
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        #归一化操作 / 255.0 将像素值范围从0到255缩放到0到1之间，这样做有助于后续处理，
        # 例如色彩增强。因此，image / 255.0 将图像的像素值归一化到了0到1之间。
        image = cv2.imread(image_path) / 255.0  # 读取图片并归一化
        #在进行色彩增强后，为了将像素值恢复到原始的0到255范围，需要进行反归一化操作。
        # 因此，乘以255的目的是将增强后的图像像素值从0到1的范围映射回0到255的范围
        enhanced_image = zmIceColor(image) * 255  # 进行色彩增强
        cv2.imwrite(image_path, enhanced_image)  # 覆盖原图

    print("Color enhancement completed.")
