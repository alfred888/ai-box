import torch
import clip
from PIL import Image
import os
from tqdm import tqdm

# 加载 CLIP 模型和预处理方法
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义图像目录
image_dir = '/Users/yafei/test-data/original'

# 遍历目录中的所有图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

# 定义描述模板
descriptions = [
    "a photo of a person",
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a landscape",
    "a photo of a cityscape",
    "a photo of a group of people",
    "a photo of an animal",
    "a photo of food",
    "a photo of a car",
    "a photo of a house",
    "a photo of a tree",
    "a photo of a flower",
    "a photo of a bird",
    "a photo of a beach",
    "a photo of a mountain",
    "a photo of a river",
    "a photo of a forest"
]

# 将描述模板转换为特征向量
text_inputs = torch.cat([clip.tokenize(description) for description in descriptions]).to(device)

# 存储识别结果
results = []

# 显示进度条
for image_file in tqdm(image_files, desc="Processing images"):
    # 加载图像
    image_path = os.path.join(image_dir, image_file)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 计算图像和文本的特征向量
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # 计算相似度
    logits_per_image, logits_per_text = model(image, text_inputs)
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()

    # 获取最可能的描述
    best_description_idx = probs.argmax()
    best_description = descriptions[best_description_idx]

    # 打印或存储结果
    print(f"Image: {image_file}, Description: {best_description}")
    results.append((image_file, best_description))

# 输出所有识别结果
for result in results:
    print(f"Image: {result[0]}, Description: {result[1]}")