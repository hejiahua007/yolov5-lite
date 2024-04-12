# from PIL import Image, ImageChops
# #1. 像素反转
# img = Image.open("../pic/2.jpg")
# inv_img = ImageChops.invert(img)    #像素值反转
# inv_img.show()

# # 2. 色彩抖动
# import numpy as np
# from PIL import Image
# from PIL import ImageEnhance
# import cv2
# import random
# import matplotlib.pyplot as plt
#
# def randomColor(image, saturation=0, brightness=0, contrast=0, sharpness=0):
#     if random.random() < saturation:
#         random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
#         image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
#     if random.random() < brightness:
#         random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
#         image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
#     if random.random() < contrast:
#         random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
#         image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
#     if random.random() < sharpness:
#         random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
#         image = ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
#     return image
#
# img = cv2.imread("../pic/3.jpg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB
# cj_img = Image.fromarray(img_rgb)
# sa_img = np.asarray(randomColor(cj_img, saturation=1))
# br_img = np.asarray(randomColor(cj_img, brightness=1))
# co_img = np.asarray(randomColor(cj_img, contrast=1))
# sh_img = np.asarray(randomColor(cj_img, sharpness=1))
# rc_img = np.asarray(randomColor(cj_img, saturation=1, brightness=1, contrast=1, sharpness=1))
#
# #plt.title设置中文
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.figure(figsize=(15, 10))
# plt.subplot(2, 3, 1), plt.imshow(img_rgb)
# plt.axis('off')
# plt.title('原图')
# plt.subplot(2, 3, 2), plt.imshow(sa_img)
# plt.axis('off')
# plt.title('调整饱和度')
# plt.subplot(2, 3, 3), plt.imshow(br_img)
# plt.axis('off')
# plt.title('调整亮度')
# plt.subplot(2, 3, 4), plt.imshow(co_img)
# plt.axis('off')
# plt.title('调整对比度')
# plt.subplot(2, 3, 5), plt.imshow(sh_img)
# plt.axis('off')
# plt.title('调整锐度')
# plt.subplot(2, 3, 6), plt.imshow(rc_img)
# plt.axis('off')
# plt.title('调整所有项')
# plt.tight_layout()
# plt.show()

# # 3. 色彩增强
# # 概述：ACE考虑了图像中颜色和亮度的空间位置关系，进行局部特性的自适应滤波，实现具有局部和非线性特征的图像亮度与色彩调整和对比度调整，同时满足灰色世界理论假设和白色斑点假设。
# import cv2
# import numpy as np
# import math
# def stretchImage(data, s=0.005, bins=2000):  # 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
#     ht = np.histogram(data, bins);
#     d = np.cumsum(ht[0]) / float(data.size)
#     lmin = 0;
#     lmax = bins - 1
#     while lmin < bins:
#         if d[lmin] >= s:
#             break
#         lmin += 1
#     while lmax >= 0:
#         if d[lmax] <= 1 - s:
#             break
#         lmax -= 1
#     return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)
# g_para = {}
# def getPara(radius=5):  # 根据半径计算权重参数矩阵
#     global g_para
#     m = g_para.get(radius, None)
#     if m is not None:
#         return m
#     size = radius * 2 + 1
#     m = np.zeros((size, size))
#     for h in range(-radius, radius + 1):
#         for w in range(-radius, radius + 1):
#             if h == 0 and w == 0:
#                 continue
#             m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
#     m /= m.sum()
#     g_para[radius] = m
#     return m
# def zmIce(I, ratio=4, radius=300):  # 常规的ACE实现
#     para = getPara(radius)
#     height, width = I.shape
#     zh, zw = [0] * radius + [x for x in range(height)] + [height - 1] * radius, [0] * radius + [x for x in range(width)] + [width - 1] * radius
#     Z = I[np.ix_(zh, zw)]
#     res = np.zeros(I.shape)
#     for h in range(radius * 2 + 1):
#         for w in range(radius * 2 + 1):
#             if para[h][w] == 0:
#                 continue
#             res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
#     return res
# def zmIceFast(I, ratio, radius):  # 单通道ACE快速增强实现
#     height, width = I.shape[:2]
#     if min(height, width) <= 2:
#         return np.zeros(I.shape) + 0.5
#     Rs = cv2.resize(I, ((width + 1) // 2, (height + 1) // 2))
#     Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
#     Rf = cv2.resize(Rf, (width, height))
#     Rs = cv2.resize(Rs, (width, height))
#
#     return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)
# def zmIceColor(I, ratio=4, radius=3):  # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
#     res = np.zeros(I.shape)
#     for k in range(3):
#         res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
#     return res
# if __name__ == '__main__':
#     m = zmIceColor(cv2.imread("../pic/3.jpg") / 255.0) * 255
#     cv2.imwrite('../pic/zmIce.jpg', m)

# # 4. 减色处理——色彩量化
# # 概述：将图像用 32、96、160、224 这4个像素值表示。即将图像由256³压缩至4³，RGB的值只取{32,96,160,224}，这被称作色彩量化。
# import cv2
# import numpy as np
# # 减色处理
# def dicrease_color(img):
#     out = img.copy()
#     out = out // 64 * 64 + 32
#     return out
# # 读入图像
# img = cv2.imread("../pic/3.jpg")
# img = cv2.resize(img,(800,600))
# # 减色处理，也叫色彩量化
# out = dicrease_color(img)
# cv2.imwrite("../pic/2.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #色彩空间转换
# import matplotlib.pyplot as plt
# import cv2 as cv
# import numpy as np
#
# # 显示汉字用
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# # 定义显示一张图片函数
# def imshow(image):
#     if image.ndim == 2:
#         plt.imshow(image, cmap='gray')                     # 指定为灰度图像
#     else:
#         plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
#
#
# # 定义坐标数字字体及大小
# def label_def():
#     plt.xticks(fontproperties='Times New Roman', size=8)
#     plt.yticks(fontproperties='Times New Roman', size=8)
#     plt.axis('off')                                     # 关坐标，可选
#
#
# # 读取图片
# img_orig = cv.imread('../pic/3.jpg', 1)    # 读取彩色图片
#
#
# # RGB到HSI的变换
# def rgb2hsi(image):
#     b, g, r = cv.split(image)                    # 读取通道
#     r = r / 255.0                                # 归一化
#     g = g / 255.0
#     b = b / 255.0
#     eps = 1e-6                                   # 防止除零
#
#     img_i = (r + g + b) / 3                      # I分量
#
#     img_h = np.zeros(r.shape, dtype=np.float32)
#     img_s = np.zeros(r.shape, dtype=np.float32)
#     min_rgb = np.zeros(r.shape, dtype=np.float32)
#     # 获取RGB中最小值
#     min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
#     min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
#     min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
#     img_s = 1 - 3*min_rgb/(r+g+b+eps)                                            # S分量
#
#     num = ((r-g) + (r-b))/2
#     den = np.sqrt((r-g)**2 + (r-b)*(g-b))
#     theta = np.arccos(num/(den+eps))
#     img_h = np.where((b-g) > 0, 2*np.pi - theta, theta)                           # H分量
#     img_h = np.where(img_s == 0, 0, img_h)
#
#     img_h = img_h/(2*np.pi)                                                       # 归一化
#     temp_s = img_s - np.min(img_s)
#     temp_i = img_i - np.min(img_i)
#     img_s = temp_s/np.max(temp_s)
#     img_i = temp_i/np.max(temp_i)
#
#     image_hsi = cv.merge([img_h, img_s, img_i])
#     return img_h, img_s, img_i, image_hsi
#
#
# # HSI到RGB的变换
# def hsi2rgb(image_hsi):
#     eps = 1e-6
#     img_h, img_s, img_i = cv.split(image_hsi)
#
#     image_out = np.zeros((img_h.shape[0], img_h.shape[1], 3))
#     img_h = img_h*2*np.pi
#     print(img_h)
#
#     img_r = np.zeros(img_h.shape, dtype=np.float32)
#     img_g = np.zeros(img_h.shape, dtype=np.float32)
#     img_b = np.zeros(img_h.shape, dtype=np.float32)
#
#     # 扇区1
#     img_b = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), img_i * (1 - img_s), img_b)
#     img_r = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3),
#                      img_i * (1 + img_s * np.cos(img_h) / (np.cos(np.pi/3 - img_h))), img_r)
#     img_g = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), 3 * img_i - (img_r + img_b), img_g)
#
#     # 扇区2                                                                                        # H=H-120°
#     img_r = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), img_i * (1 - img_s), img_r)
#     img_g = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3),
#                      img_i * (1 + img_s * np.cos(img_h-2*np.pi/3) / (np.cos(np.pi - img_h))), img_g)
#     img_b = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), 3 * img_i - (img_r + img_g), img_b)
#
#     # 扇区3                                                                                        # H=H-240°
#     img_g = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), img_i * (1 - img_s), img_g)
#     img_b = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi),
#                      img_i * (1 + img_s * np.cos(img_h-4*np.pi/3) / (np.cos(5*np.pi/3 - img_h))), img_b)
#     img_r = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), 3 * img_i - (img_b + img_g), img_r)
#
#     temp_r = img_r - np.min(img_r)
#     img_r = temp_r/np.max(temp_r)
#
#     temp_g = img_g - np.min(img_g)
#     img_g = temp_g/np.max(temp_g)
#
#     temp_b = img_b - np.min(img_b)
#     img_b = temp_b/np.max(temp_b)
#
#     image_out = cv.merge((img_r, img_g, img_b))                   # 按RGB合并，后面不用转换通道
#     # print(image_out.shape)
#     return image_out
#
#
# if __name__ == '__main__':                                           # 运行当前函数
#
#     h, s, i, hsi = rgb2hsi(img_orig)                                 # RGB到HSI的变换
#     img_revise = np.float32(hsi2rgb(hsi))                            # HSI复原到RGB
#
#     # h, s, i = cv.split(cv.cvtColor(img_orig, cv.COLOR_BGR2HSV))     # 自带库函数HSV模型
#     im_b, im_g, im_r = cv.split(img_orig)                            # 获取RGB通道数据
#
#     plt.subplot(241), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.title('原始图'), label_def()
#     plt.subplot(242), plt.imshow(im_r, 'gray'), plt.title('R'), label_def()
#     plt.subplot(243), plt.imshow(im_g, 'gray'), plt.title('G'), label_def()
#     plt.subplot(244), plt.imshow(im_b, 'gray'), plt.title('B'), label_def()
#
#     plt.subplot(245), plt.imshow(hsi), plt.title('HSI图'), label_def()
#     plt.subplot(246), plt.imshow(h, 'gray'), plt.title('H(色调)'), label_def()
#     plt.subplot(247), plt.imshow(s, 'gray'), plt.title('S(饱和度)'), label_def()
#     plt.subplot(248), plt.imshow(i, 'gray'), plt.title('I(亮度)'), label_def()
#     plt.show()
#
#     plt.subplot(121), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.title('原RGB'), label_def()
#     plt.subplot(122), plt.imshow(img_revise), plt.title('HSI重建RGB'), label_def()
#     plt.show()

# #6. 彩色图像直方图均衡化
# import numpy as np
# import cv2
# def hisEqulColor(img):
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#     channels = cv2.split(ycrcb)
#     print(len(channels))
#     cv2.equalizeHist(channels[0], channels[0])
#     cv2.merge(channels, ycrcb)
#     cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
#     return img
# im = cv2.imread('../pic/2.jpg')
# print(np.shape(im))
# im=cv2.resize(im,(800,600))
# cv2.imshow('im1', im)
# cv2.waitKey(0)
# eq = hisEqulColor(im)
# cv2.imshow('image2',eq )
# cv2.waitKey(0)


# 已有：翻转，色域变换，噪声，大小改变，模糊，色彩抖动，均衡化
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os
import random


# 限制对比度自适应直方图均衡
def clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe


# 伽马变换
def gamma(image):
    fgamma = 2
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma


# 直方图均衡
def hist(image):
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo


# 椒盐噪声
def sp_noise(image):
    output = np.zeros(image.shape, np.uint8)
    prob = rand(0.0005, 0.001)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


# 色彩抖动
def randomColor(image):
    saturation = random.randint(0, 1)
    brightness = random.randint(0, 1)
    contrast = random.randint(0, 1)
    sharpness = random.randint(0, 1)
    if random.random() < saturation:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_data(image, input_shape=[200, 200], random=True, jitter=.5, hue=.1, sat=1.5, val=1.5, proc_img=True):
    iw, ih = image.size
    h, w = input_shape
    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.15, 2.5)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        # 翻转图像
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # 噪声或者虚化，二选一
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    a1 = np.random.randint(0, 3)
    if a1 == 0:
        image = sp_noise(image)
    elif a1 == 1:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    else:
        image = image
    # 均衡化
    index_noise = np.random.randint(0, 10)
    print(index_noise)
    if index_noise == 0:
        image = hist(image)
        print('hist,done')
    elif index_noise == 1:
        image = clahe(image)
        print('clahe,done')
    elif index_noise == 2:
        image = gamma(image)
        print('gamma,done')
    else:
        image = image

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 色彩抖动
    image = randomColor(image)
    print(image.size)
    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)

    image_data = np.array(image)
    return image_data


if __name__ == "__main__":

    # 图像批量处理
    dirs = '../pic/'  # 原始图像所在的文件夹
    dets = '../result/'  # 图像增强后存放的文件夹
    mylist = os.listdir(dirs)
    l = len(mylist)  # 文件夹图片的数量
    for j in range(0, l):
        image = cv2.imread(dirs + mylist[j])
        img = Image.fromarray(np.uint8(image))
        for i in range(0, 2):  # 自定义增强的张数
            img_ret = get_data(img)
            # imwrite(存入图片路径+图片名称+‘.jpg’,img)
            # 注意：名称应该是变化的，不然会覆盖原来的图片
            cv2.imwrite(dets + '1' + str(j) + '0' + str(i) + '.jpg', img_ret)
            print('done')











