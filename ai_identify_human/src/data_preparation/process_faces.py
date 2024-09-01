import os
import cv2
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

from ai_identify_human.src.config.config import Config

# 加载配置
config = Config()

# 设置路径
raw_dir = config.raw_data_dir
processed_dir = config.processed_data_dir

# 确保保存目录存在
os.makedirs(processed_dir, exist_ok=True)

# 加载预训练的 ViT 模型和特征提取器
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 设置模型为评估模式
model.eval()

# 使用 OpenCV 的预训练 Haar Cascade 模型进行人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_and_save_faces(image_path, save_dir):
    """
    检测图片中的人脸并将其保存到指定目录
    :param image_path: 原始图片路径
    :param save_dir: 保存处理后人脸图像的目录
    """
    # 读取图像并转换为灰度图像以便进行人脸检测
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 对每一张人脸进行处理
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y + h, x:x + w]

        # 将人脸区域转换为 RGB 格式的 PIL 图像
        face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # 使用 ViT 进行分类
        inputs = feature_extractor(images=face_image, return_tensors="pt")

        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # 保存人脸
        face_filename = os.path.join(save_dir,
                                     f'{os.path.splitext(os.path.basename(image_path))[0]}_face_{i + 1}_class_{predicted_class_idx}.jpg')
        face_image.save(face_filename)
        print(f'Saved detected face {i + 1} to {face_filename}')


def process_images_in_directory(raw_dir, processed_dir):
    """
    处理目录中的所有图像文件
    :param raw_dir: 原始图片所在目录
    :param processed_dir: 处理后图片保存的目录
    """
    # 创建保存处理后图片的目录（如果不存在）
    os.makedirs(processed_dir, exist_ok=True)

    # 遍历目录中的所有图像文件
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                detect_and_save_faces(image_path, processed_dir)


if __name__ == "__main__":
    # 处理 raw 目录下的所有图片，并将结果保存到 processed 目录
    process_images_in_directory(raw_dir, processed_dir)