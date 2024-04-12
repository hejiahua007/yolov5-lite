import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import random

def randomColor(image, saturation=1, brightness=1, contrast=1, sharpness=1):
    if random.random() < saturation:
        random_factor = np.random.uniform(0.8, 1.2)
        image = ImageEnhance.Color(image).enhance(random_factor)
    if random.random() < brightness:
        random_factor = np.random.uniform(0.5, 0.8)
        image = ImageEnhance.Brightness(image).enhance(random_factor)
    if random.random() < contrast:
        random_factor = np.random.uniform(0.8, 1.2)
        image = ImageEnhance.Contrast(image).enhance(random_factor)
    if random.random() < sharpness:
        random_factor = np.random.uniform(0.8, 1.2)
        image = ImageEnhance.Sharpness(image).enhance(random_factor)
    return image

def process_and_save_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB
    pil_image = Image.fromarray(img_rgb)
    processed_image = randomColor(pil_image)
    processed_image_rgb = np.array(processed_image)
    cv2_img = cv2.cvtColor(processed_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, cv2_img)

image_folder_path = r'C:\Users\123\Desktop\yolo\data_collect\valid\images'
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    process_and_save_image(image_path)

print("All images have been processed and saved.")
