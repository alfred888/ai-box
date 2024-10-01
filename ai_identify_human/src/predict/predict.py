import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from src.config.config import Config

class Predictor:
    def __init__(self, config):
        # 初始化
        self.config = config
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载训练好的模型
        model_path = os.path.join(self.config.model_path, 'final_model.pth')
        print(f"Loading model from {model_path}")
        if os.path.exists(model_path):
            # 使用通用的分类模型加载
            self.model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=len(self.config.persons) + 1,  # 人数 + Others 类别
                ignore_mismatched_sizes=True
            )
            # 加载模型参数
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'], strict=False)
            self.model.to(self.device)
            self.model.eval()
        else:
            print(f"Model not found at {model_path}. Exiting...")
            exit(1)

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

        # 进行预测
        with torch.no_grad():
            outputs = self.model(**inputs).logits
            probs = torch.softmax(outputs, dim=-1)

        # 找到预测概率最高的类别
        predicted_index = torch.argmax(probs, dim=-1).item()
        categories = self.config.persons + ['Others']  # 全部类别列表
        if predicted_index < len(categories):
            predicted_person = categories[predicted_index]
            return predicted_person
        else:
            return "Unknown"

    def predict_folder(self, folder_path):
        # 遍历文件夹中的所有图片
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # 检查是否是图片文件（你可以根据文件扩展名进行过滤）
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                result = self.predict(file_path)
                print(f"Image: {file_name}, Predicted person: {result}")

if __name__ == '__main__':
    # 加载配置
    config = Config()
    config.persons = config.persons

    # 初始化预测器
    predictor = Predictor(config)

    # 图片文件夹路径（替换为要预测的文件夹路径）
    test_folder_path = os.path.join(config.user_home_path, config.project_name)

    # 预测文件夹中所有图片
    predictor.predict_folder(test_folder_path)