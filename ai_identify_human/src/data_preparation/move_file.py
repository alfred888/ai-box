import os
import shutil
from src.config.config import Config


def move_files_to_parent_directory(parent_directory):
    # 遍历 parent_directory 下的所有子文件夹
    for root, dirs, files in os.walk(parent_directory):
        # 如果 root 是父目录本身，跳过
        if root == parent_directory:
            continue

        # 移动每个子文件夹中的文件到父目录
        for file in files:
            file_path = os.path.join(root, file)
            new_path = os.path.join(parent_directory, file)

            # 如果文件名冲突，可以选择重命名或覆盖，这里默认覆盖
            if os.path.exists(new_path):
                print(f"Overwriting existing file: {new_path}")

            shutil.move(file_path, new_path)
            print(f"Moved: {file_path} -> {new_path}")

        # 删除空的子文件夹
        if not os.listdir(root):
            os.rmdir(root)
            print(f"Removed empty directory: {root}")


if __name__ == '__main__':
    # 从配置文件中读取路径
    config = Config()
    parent_directory = config.data_path+"/train/Others"  # 假设该路径指向 Others 文件夹
    print(f"Moving files in: {parent_directory}")
    # 调用函数移动文件
    move_files_to_parent_directory(parent_directory)
    parent_directory = config.data_path + "/val/Others"  # 假设该路径指向 Others 文件夹
    print(f"Moving files in: {parent_directory}")
    # 调用函数移动文件
    move_files_to_parent_directory(parent_directory)