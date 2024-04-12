import os
import cv2
import numpy as np

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    ycrcb = cv2.merge(channels)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def decrease_brightness(img, factor):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = list(cv2.split(ycrcb))
    channels[0] = np.clip(channels[0] * factor, 0, 255).astype(np.uint8)
    ycrcb = cv2.merge(channels)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def process_image(image_path, brightness_factor):
    im = cv2.imread(image_path)
    eq_with_decreased_brightness = decrease_brightness(hisEqulColor(im), brightness_factor)
    cv2.imwrite(image_path, eq_with_decreased_brightness)

# 获取指定路径下的所有图片文件
image_folder_path = r'C:\Users\123\Desktop\yolo\1'
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# 亮度降低因子
brightness_factor = 0.8  # 调整这个因子以控制亮度的降低程度

# 遍历每张图片进行处理
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    process_image(image_path, brightness_factor)

print("Image processing completed.")
