import os
import json
from enum import Enum


class Persons(Enum):
    YAFEI = "Yafei"
    XIAOMI = "Xiaomi"
    XIAOHU = "Xiaohu"


class Config:
    def __init__(self, config_file_path=None):
        if config_file_path is None:
            # 使用当前文件所在目录的绝对路径作为基准
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(current_dir, 'config.json')

        # 加载配置文件
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)

        # 从配置文件中读取 user_home_path, data_path, model_path 和 persons
        self.user_home_path = config_data.get('user_home_path')
        self.project_name = config_data.get('project_name')
        self.data_path = os.path.join(self.user_home_path, self.project_name, config_data.get('data_path'))
        self.model_path = os.path.join(self.user_home_path, self.project_name, config_data.get('model_path'))

        # 获取第三方数据路径
        self.third_party_data_path = os.path.join(self.user_home_path, config_data.get('3rd_data_path'))
        self.third_party_other_faces_path = os.path.join(self.third_party_data_path,
                                                         config_data.get('3rd_data_other_faces_path'))

        # 检查必要的路径
        if not self.user_home_path:
            raise ValueError("Configuration error: 'user_home_path' is missing in the configuration file.")
        if not self.data_path:
            raise ValueError("Configuration error: 'data_path' is missing in the configuration file.")
        if not self.model_path:
            raise ValueError("Configuration error: 'model_path' is missing in the configuration file.")
        if not self.third_party_data_path:
            raise ValueError("Configuration error: 'third_party_data_path' is missing in the configuration file.")
        if not self.third_party_other_faces_path:
            raise ValueError(
                "Configuration error: 'third_party_other_faces_path' is missing in the configuration file.")

        self.persons = [person.value for person in Persons]

    def create_directories(self):
        """
        创建所有必要的目录
        """
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.third_party_other_faces_path, exist_ok=True)

        for person in self.persons:
            person_dir = os.path.join(self.data_path, person)
            os.makedirs(person_dir, exist_ok=True)
            for folder in ['train', 'validate', 'test']:
                os.makedirs(os.path.join(person_dir, folder), exist_ok=True)

        print(f"Created all necessary directories in: {self.data_path}")


if __name__ == "__main__":
    try:
        # 创建 Config 实例
        config = Config()

        # 创建所有目录
        config.create_directories()

        # 打印配置项
        print("User Home Path:", config.user_home_path)
        print("Data Path:", config.data_path)
        print("Model Path:", config.model_path)
        print("3rd Party Other Faces Path:", config.third_party_other_faces_path)
        print("Persons:", config.persons)

    except ValueError as e:
        print(e)
        exit(1)