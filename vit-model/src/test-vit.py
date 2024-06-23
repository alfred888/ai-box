import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os

# 定义模型名称和预训练权重
model_name = 'google/vit-base-patch16-224'

# 加载预训练的 ViT 模型和特征提取器
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# 定义图像目录
image_dir = '/Users/yafei/test-data/original'

# 遍历目录中的所有图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

# 存储识别结果
results = []

for image_file in image_files:
    # 加载图像
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert('RGB')

    # 预处理图像
    image = transform(image)

    # 添加 batch 维度
    image = image.unsqueeze(0)

    # 将图像传入模型
    with torch.no_grad():
        outputs = model(image)

    # 获取预测结果
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]

    # 打印或存储结果
    print(f"Image: {image_file}, Predicted class: {predicted_class_label}")
    results.append((image_file, predicted_class_label))

# 输出所有识别结果
for result in results:
    print(f"Image: {result[0]}, Predicted class: {result[1]}")