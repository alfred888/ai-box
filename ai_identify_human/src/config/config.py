import os
import json


class Config:
    def __init__(self, config_file_path=None):
        if config_file_path is None:
            # 使用当前文件所在目录的绝对路径作为基准
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(current_dir, 'config.json')
        # 加载配置文件
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)

        # 从配置文件中读取 data_home_path 和 datasets_home_path，并检查是否存在
        self.data_home_path = config_data.get('data_home_path')
        self.datasets_home_path = config_data.get('datasets_home_path')

        if not self.data_home_path:
            raise ValueError("Configuration error: 'data_home_path' is missing in the configuration file.")

        if not self.datasets_home_path:
            raise ValueError("Configuration error: 'datasets_home_path' is missing in the configuration file.")

        # 初始化 data 目录及其子目录路径
        self.raw_data_dir = os.path.join(self.data_home_path,'data','raw')
        self.processed_data_dir = os.path.join(self.data_home_path,'data', 'processed')
        self.interim_data_dir = os.path.join(self.data_home_path,'data', 'interim')
        self.external_data_dir = os.path.join(self.data_home_path,'data', 'external')

        # 初始化 datasets 目录及其子目录路径
        self.dataset1_dir = os.path.join(self.datasets_home_path, 'dataset1')
        self.dataset2_dir = os.path.join(self.datasets_home_path, 'dataset2')

    def create_directories(self):
        """
        创建所有必要的目录
        """
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.interim_data_dir, exist_ok=True)
        os.makedirs(self.external_data_dir, exist_ok=True)
        os.makedirs(self.dataset1_dir, exist_ok=True)
        os.makedirs(self.dataset2_dir, exist_ok=True)


# 使用示例
if __name__ == "__main__":
    try:
        # 创建 Config 实例
        config = Config()

        # 创建所有目录
        config.create_directories()

        # 打印目录路径
        print("Raw Data Directory:", config.raw_data_dir)
        print("Processed Data Directory:", config.processed_data_dir)
        print("Interim Data Directory:", config.interim_data_dir)
        print("External Data Directory:", config.external_data_dir)
        print("Dataset 1 Directory:", config.dataset1_dir)
        print("Dataset 2 Directory:", config.dataset2_dir)

    except ValueError as e:
        print(e)
        exit(1)