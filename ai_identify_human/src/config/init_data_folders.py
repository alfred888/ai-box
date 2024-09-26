import os
from config import Config

class InitDataFolders:
    def __init__(self, config):
        self.data_path = config.data_path
        self.persons = config.persons

    def create_directories(self):
        for person in self.persons:
            person_dir = os.path.join(self.data_path, person)
            train_dir = os.path.join(person_dir, 'train')
            validate_dir = os.path.join(person_dir, 'validate')
            test_dir = os.path.join(person_dir, 'test')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(validate_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            print(f"Created directories for {person}:")
            print(f"  - {train_dir}")
            print(f"  - {validate_dir}")
            print(f"  - {test_dir}")
        Others_dir = os.path.join(config.third_party_data_path,config.third_party_other_faces_path)
        Others_train_dir=os.path.join(Others_dir,'train')
        Others_val_dir=os.path.join(Others_dir,'validate')
        Others_test_dir=os.path.join(Others_dir,'test')
        os.makedirs(Others_dir, exist_ok=True)
        os.makedirs(Others_train_dir, exist_ok=True)
        os.makedirs(Others_val_dir, exist_ok=True)
        os.makedirs(Others_test_dir, exist_ok=True)
        print(f" - {Others_dir}")
        print(f"  - {Others_train_dir}")
        print(f"  - {Others_val_dir}")
        print(f"  - {Others_test_dir}")

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