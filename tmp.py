import cv2
import os

def draw_bounding_boxes_for_each_image(image_dir, result_dir):
    # 确保结果目录存在
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 遍历image_dir中的所有文件
    for image_filename in os.listdir(image_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_filename)
            annotation_path = image_path.rsplit('.', 1)[0] + '.txt'

            # 检查标注文件是否存在
            if os.path.exists(annotation_path):
                # 读取图片
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to read image '{image_filename}'.")
                    continue
                height, width, _ = image.shape

                # 读取并处理每个标注
                with open(annotation_path, 'r') as f:
                    annotations = f.readlines()
                    for annotation in annotations:
                        parts = annotation.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, box_width, box_height = map(float, parts)
                            # 将坐标转换为绝对像素值
                            x_center, box_width = x_center * width, box_width * width
                            y_center, box_height = y_center * height, box_height * height
                            x1, y1 = int(x_center - box_width / 2), int(y_center - box_height / 2)
                            x2, y2 = int(x_center + box_width / 2), int(y_center + box_height / 2)

                            # 在图片上绘制边界框
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        else:
                            print(f"Invalid annotation format in file: {annotation_path}")

                # 保存处理后的图片到结果目录
                result_path = os.path.join(result_dir, image_filename)
                cv2.imwrite(result_path, image)
            else:
                print(f"No annotation file found for {image_filename}")

# 定义路径
image_dir = "D:\\git_cangku\\data\\book\\book_data_collect2\\train\\images"
result_dir = "D:\\git_cangku\\data\\book\\book_data_collect2\\train\\result"

# 调用函数
draw_bounding_boxes_for_each_image(image_dir, result_dir)
