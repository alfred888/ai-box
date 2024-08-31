import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from config_path import Config
from PIL import UnidentifiedImageError

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义数据集类以跳过损坏的图像
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]

        # 跳过以 `._` 开头的文件
        if os.path.basename(path).startswith("._"):
            print(f"Warning: Skipping macOS metadata file {path}.")
            return None, None

        try:
            return super().__getitem__(index)
        except UnidentifiedImageError:
            print(f"Warning: Unable to load image {path}. Skipping.")
            return None, None


def filter_valid_data(batch):
    """过滤掉 None 的数据项"""
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    # 加载训练和验证数据集
    config = Config()
    train_dataset = SafeImageFolder(root=config.son_train_dir, transform=train_transform)
    val_dataset = SafeImageFolder(root=config.son_val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=filter_valid_data)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=filter_valid_data)

    # 加载预训练的 ViT 模型
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2,
                                                      ignore_mismatched_sizes=True)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # 训练模型
    num_epochs = 10

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}')
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
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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

    # 保存模型
    torch.save(model.state_dict(), config.model_save_path)