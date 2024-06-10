import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
from datasets import load_metric
import numpy as np

# 定义数据预处理
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 加载预训练的 ViT 模型
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=1)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

# 定义计算准确率的函数
metric = load_metric("accuracy")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)


# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()


