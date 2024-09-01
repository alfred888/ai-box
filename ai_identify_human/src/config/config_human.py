import os
import json
from enum import Enum


# 定义人员的枚举类,datasets 目录里面，根据不同的人，生成目录。
# 人的定义是在一个枚举类里，例如“Yafei”，“Xiaomi”，“Tiger”，“others”，每个人有自己的数据集文件夹，每个文件夹下面，有训练数据，验证数据，
# 测试数据，三个文件夹。 给我生成文件夹的程序。 以及定义人的程序。
class Person(Enum):
    YAFEI = "Yafei"
    XIAOMI = "Xiaomi"
    TIGER = "Tiger"
    OTHERS = "Others"

# 配置类，管理目录结构和路径
class Config:
    def __init__(self, config_file_path='config.json'):
        # 加载配置文件
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)

        # 从配置文件中读取 datasets_home_path，并检查是否存在
        self.datasets_home_path = config_data.get('datasets_home_path')
        if not self.datasets_home_path:
            raise ValueError("Configuration error: 'datasets_home_path' is missing in the configuration file.")

    def create_person_directories(self):
        # 遍历枚举类中的每一个人员，创建对应的目录结构
        for person in Person:
            person_dir = os.path.join(self.datasets_home_path,"humans", person.value)
            train_dir = os.path.join(person_dir, 'train')
            val_dir = os.path.join(person_dir, 'val')
            test_dir = os.path.join(person_dir, 'test')

            # 创建目录结构
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            print(f"Created directories for {person.value}:")
            print(f"  - {train_dir}")
            print(f"  - {val_dir}")
            print(f"  - {test_dir}")


# 使用示例
if __name__ == "__main__":
    try:
        # 创建 Config 实例并加载配置文件
        config = Config()

        # 创建所有目录
        config.create_person_directories()

    except ValueError as e:
        print(e)
        exit(1)