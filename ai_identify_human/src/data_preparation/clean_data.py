# ai_identify_human/src/data_preparation/clean_data.py
import os

from src.config.config import Config

def delete_macos_metadata_files(root_dir):
    """
    删除指定目录及其子目录中所有以 '._' 开头的 macOS 元数据文件。

    :param root_dir: 要清理的根目录
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("._"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    config = Config()
    delete_macos_metadata_files(config.data_path)