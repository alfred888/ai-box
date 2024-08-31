import os
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from config_path import Config

def predict_image(image_path, model, processor, device):
    """
    对单张图像进行预测
    :param image_path: 图像路径
    :param model: 加载的ViT模型
    :param processor: 图像处理器
    :param device: 设备（CPU或GPU）
    :return: 预测的类别标签
    """
    image = Image.open(image_path)

    # 确保图像是RGB格式
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs['pixel_values'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs).logits
        predicted_class = torch.argmax(outputs, dim=-1).item()

    return predicted_class

def predict_directory_recursive(directory_path, model, processor, device):
    """
    对目录及其所有子目录中的图像进行预测
    :param directory_path: 目录路径
    :param model: 加载的ViT模型
    :param processor: 图像处理器
    :param device: 设备（CPU或GPU）
    """
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(dirpath, filename)
                predicted_class = predict_image(file_path, model, processor, device)
                class_names = ['others', 'son']  # 根据训练时的类别顺序
                print(f"Image: {file_path} | Predicted class: {class_names[predicted_class]}")

if __name__ == "__main__":
    # 加载配置
    config = Config()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 ViT 模型，并忽略大小不匹配
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    model.to(device)

    # 加载图像处理器
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # 使用配置中的 test_image_path
    test_directory_path = config.test_image_path  # 使用Config中的test_image_path

    # 对目录及其所有子目录中的图像进行预测
    predict_directory_recursive(test_directory_path, model, processor, device)