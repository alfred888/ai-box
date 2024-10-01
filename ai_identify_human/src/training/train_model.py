import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTForImageClassification
from src.config.config import Config
from PIL import UnidentifiedImageError
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import logging

logging.getLogger("urllib3").setLevel(logging.ERROR)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理，增加更多的数据增强策略
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),  # 增加旋转角度，提高数据多样性
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义数据集类以跳过损坏的图像并打印出有问题的文件路径
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        if os.path.basename(path).startswith("._"):
            print(f"Skipping hidden file: {path}")
            return None, None
        try:
            return super().__getitem__(index)
        except UnidentifiedImageError:
            print(f"Error loading image: {path}")  # 打印出有问题的文件路径
            return None, None


def filter_valid_data(batch):
    """过滤掉 None 的数据项"""
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def train_and_validate(train_dir, val_dir, model_save_path):
    # 打印出加载的训练集和验证集目录路径
    print(f"Loading training dataset from: {train_dir}")
    print(f"Loading validation dataset from: {val_dir}")

    # 加载训练集
    train_dataset = SafeImageFolder(root=train_dir, transform=train_transform)

    # 打印训练集的类别名称和顺序
    print(f"Training dataset classes (order is important!): {train_dataset.classes}")
    for class_name in train_dataset.classes:
        class_idx = train_dataset.class_to_idx[class_name]
        print(f"Class '{class_name}' has index {class_idx}.")

    # 加载验证集
    val_dataset = SafeImageFolder(root=val_dir, transform=val_transform)

    # 打印验证集的类别名称和顺序
    print(f"Validation dataset classes (order is important!): {val_dataset.classes}")
    for class_name in val_dataset.classes:
        class_idx = val_dataset.class_to_idx[class_name]
        print(f"Class '{class_name}' has index {class_idx}.")

    # 打印训练集中的类别数量
    print(f"Number of classes in training dataset: {len(train_dataset.classes)}")

    # 训练和验证数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=filter_valid_data)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=filter_valid_data)

    # 模型初始化
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=len(train_dataset.classes),  # 根据类别数量动态设置输出维度
        ignore_mismatched_sizes=True
    )

    # 冻结 ViT 的前几个层，只训练分类器部分
    for param in model.vit.parameters():
        param.requires_grad = False

    model = model.to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    scaler = GradScaler()

    num_epochs = 10
    best_val_loss = float('inf')
    early_stopping_patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        train_loader_tqdm = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')
        for batch in train_loader_tqdm:
            if batch is None:
                continue
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs).logits
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        val_loader_tqdm = tqdm(val_loader, desc=f'Validating Epoch {epoch + 1}')
        with torch.no_grad():
            for batch in val_loader_tqdm:
                if batch is None:
                    continue
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(inputs).logits
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_loader_tqdm.set_postfix(loss=loss.item())

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        epoch_duration = time.time() - start_time
        print(f'Epoch {epoch + 1} completed in {epoch_duration // 60:.0f}m {epoch_duration % 60:.0f}s')

        # 调整学习率
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"Validation loss improved. Saving model to {model_save_path}.")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, model_save_path)
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter} epochs.")

        # 提前停止
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break


if __name__ == '__main__':
    config = Config()

    # 设置训练和验证集路径
    train_dir = os.path.join(config.data_path, 'train')
    val_dir = os.path.join(config.data_path, 'val')

    # 保存模型路径
    model_save_path = os.path.join(config.model_path, 'final_model.pth')

    # 调用训练和验证函数
    train_and_validate(
        train_dir=train_dir,
        val_dir=val_dir,
        model_save_path=model_save_path
    )