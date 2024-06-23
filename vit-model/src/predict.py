from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from config import Config

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
config = Config()
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2)
model.load_state_dict(torch.load(config.model_save_path))
model = model.to(device)
model.eval()

# 定义图像预处理
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def predict(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(inputs).logits
        _, preds = torch.max(outputs, 1)

    return 'son' if preds.item() == 0 else 'others'

# 测试推理
if __name__ == "__main__":
    print(f'The image is predicted as: {predict(config.test_image_path)}')