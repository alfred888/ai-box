import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config_path import Config


def get_all_images(directory):
    """获取目录及其子目录中的所有图像文件"""
    all_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_images.append(os.path.join(root, file))
    return all_images


def prepare_data(source_dir, train_dir, val_dir, test_size=0.2, random_state=42):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有图像文件
    all_images = get_all_images(source_dir)

    if len(all_images) < 2:
        raise ValueError(f"Not enough images in {source_dir} to split into training and validation sets.")

    # 划分训练集和验证集
    train_images, val_images = train_test_split(all_images, test_size=test_size, random_state=random_state)

    # 复制图像文件到目标目录
    print(f"Copying training images from {source_dir} to {train_dir}...")
    for image in tqdm(train_images):
        shutil.copy(image, train_dir)

    print(f"Copying validation images from {source_dir} to {val_dir}...")
    for image in tqdm(val_images):
        shutil.copy(image, val_dir)


if __name__ == "__main__":
    config = Config()
    prepare_data(config.son_source_dir, config.son_train_dir, config.son_val_dir)
    prepare_data(config.others_source_dir, config.others_train_dir, config.others_val_dir)