from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# 加载模型和特征提取器
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 加载图像
url = "https://example.com/path/to/your/image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 文本输入
texts = ["a photo of a cat", "a photo of a dog"]

# 预处理图像和文本
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取图像和文本之间的相似性得分
logits_per_image = outputs.logits_per_image
logits_per_text = outputs.logits_per_text

# 打印相似性得分
print("Logits per image:", logits_per_image)
print("Logits per text:", logits_per_text)