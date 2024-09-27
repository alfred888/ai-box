import os
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from enum import Enum
from ai_identify_human.src.config.config import Config


class Predictor:
    def __init__(self, config):
        # 初始化
        self.config = config
        self.models = {}
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载每个人的模型
        for person in self.config.persons:
            model_path = os.path.join(self.config.model_path, f"{person}_model.pth")
            print(f"Loading model for {person} from {model_path}")
            if os.path.exists(model_path):
                model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=2,  # 目标人物和Others两个类别
                    ignore_mismatched_sizes=True
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models[person] = model
            else:
                print(f"Model for {person} not found at {model_path}. Skipping...")

    def predict(self, image_path):
        # 读取并预处理图片
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        # 提取特征
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)

        # 遍历所有模型进行预测
        predictions = {}
        for person, model in self.models.items():
            with torch.no_grad():
                outputs = model(**inputs).logits
                probs = torch.softmax(outputs, dim=-1)
                predictions[person] = float(probs[0][1].item())  # 获取该类别的概率值

        # 找到最高概率的分类
        if predictions:
            predicted_person = max(predictions, key=predictions.get)
            return predicted_person
        else:
            return "No valid predictions available."

if __name__ == '__main__':
    # 加载配置
    config = Config()

    # 初始化预测器
    predictor = Predictor(config)

    # 示例图片路径（替换为要预测的图片路径）
    test_image_path = "/Users/yafei/test1/3rd_data/other_faces/test/Angelica_Romero/Angelica_Romero_0001.jpg"

    # 进行预测
    result = predictor.predict(test_image_path)
    print(f"Predicted person: {result}")