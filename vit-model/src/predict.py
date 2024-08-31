import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from config_path import Config

def predict_image(image_path, model, feature_extractor, device):
    """
    对图像进行预测
    :param image_path: 图像路径
    :param model: 加载的ViT模型
    :param feature_extractor: 图像特征提取器
    :param device: 设备（CPU或GPU）
    :return: 预测的类别标签
    """
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs['pixel_values'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs).logits
        predicted_class = torch.argmax(outputs, dim=-1).item()

    return predicted_class

if __name__ == "__main__":
    # 加载配置
    config = Config()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 ViT 模型
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2)
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    model.to(device)

    # 加载特征提取器
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # 指定要预测的图像
    test_image_path = config.test_image_path

    # 进行预测
    predicted_class = predict_image(test_image_path, model, feature_extractor, device)

    # 输出预测结果
    class_names = ['son', 'others']  # 根据训练时的类别顺序
    print(f"The predicted class for the image is: {class_names[predicted_class]}")