import os
from config import Config

class InitDataFolders:
    def __init__(self, config):
        self.data_path = config.data_path  # 图片数据的主路径
        self.persons = config.persons  # 三个人物名称列表
        self.third_party_other_faces_path = config.third_party_other_faces_path  # Others 的路径

    def create_directories(self):
        # 创建主目录 dataset 以及 train 和 val 文件夹
        train_dir = os.path.join(self.data_path, 'train')
        val_dir = os.path.join(self.data_path, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # 为每个目标人物在 train 和 val 下创建子目录
        for person in self.persons:
            person_train_dir = os.path.join(train_dir, person)
            person_val_dir = os.path.join(val_dir, person)

            os.makedirs(person_train_dir, exist_ok=True)
            os.makedirs(person_val_dir, exist_ok=True)

            print(f"Created directories for {person}:")
            print(f"  - {person_train_dir}")
            print(f"  - {person_val_dir}")

        # 为 Others 类别创建子目录
        others_train_dir = os.path.join(train_dir, 'Others')
        others_val_dir = os.path.join(val_dir, 'Others')

        os.makedirs(others_train_dir, exist_ok=True)
        os.makedirs(others_val_dir, exist_ok=True)

        print(f"Created directories for Others:")
        print(f"  - {others_train_dir}")
        print(f"  - {others_val_dir}")

if __name__ == "__main__":
    try:
        # 创建 Config 实例并加载配置文件
        config = Config()

        # 创建 InitDataFolders 实例并创建目录
        init_data_folders = InitDataFolders(config)
        init_data_folders.create_directories()

    except ValueError as e:
        print(e)
        exit(1)