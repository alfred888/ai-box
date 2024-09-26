# 主要用来识别具体的人

# 目录说明，看项目创建脚本，数据集单独放置。
project_name/
│
├── data/                           # 原始数据和处理后的数据
│   ├── raw/                        # 原始、未处理的数据
│   ├── processed/                  # 预处理后、清理后的数据
│   ├── interim/                    # 临时数据，处于处理中间步骤的数据
│   └── external/                   # 从外部导入的数据集
│
├── datasets/                       # 数据集管理
│   ├── dataset1/                   # 数据集1
│   ├── dataset2/                   # 数据集2
│   └── ...                         # 更多数据集



project_name/
├── src/                            # 源代码目录
│   ├── __init__.py                 # 初始化模块
│   ├── config/                     # 配置管理
│   │   └── config.py               # 配置文件
│   │   └── config.json             # JSON 配置文件
│   ├── data_preparation/           # 数据准备、清理、加工
│   │   ├── __init__.py
│   │   ├── clean_data.py           # 数据清理脚本
│   │   ├── prepare_data.py         # 数据准备脚本
│   │   └── transform_data.py       # 数据变换、处理脚本
│   ├── training/                   # 训练相关代码
│   │   ├── __init__.py
│   │   ├── train_model.py          # 训练模型的脚本
│   │   ├── model.py                # 定义模型结构
│   │   └── utils.py                # 训练过程中使用的工具函数
│   ├── testing/                    # 测试和验证相关代码
│   │   ├── __init__.py
│   │   ├── evaluate.py             # 模型评估脚本
│   │   └── test_model.py           # 测试模型的脚本
│   ├── inferencing/                # 推理/预测代码
│   │   ├── __init__.py
│   │   ├── predict.py              # 推理脚本
│   │   └── deploy.py               # 部署模型脚本
│   ├── visualization/              # 数据和模型结果的可视化
│   │   ├── __init__.py
│   │   ├── plot_results.py         # 可视化结果的脚本
│   │   └── visualize_data.py       # 数据可视化的脚本
│   └── utils/                      # 常用的工具函数和类
│       ├── __init__.py
│       ├── logger.py               # 日志记录工具
│       └── helpers.py              # 辅助函数
│
├── notebooks/                      # Jupyter notebooks 用于实验和探索
│   ├── data_exploration.ipynb      # 数据探索的 notebook
│   ├── model_training.ipynb        # 模型训练的 notebook
│   └── ...                         # 更多 notebooks
│
├── scripts/                        # 一些常用的脚本
│   ├── run_training.sh             # 用于启动训练的 shell 脚本
│   ├── run_testing.sh              # 用于启动测试的 shell 脚本
│   └── ...                         # 更多脚本
│
├── requirements.txt                # 项目所需的 Python 包列表
├── Dockerfile                      # 用于 Docker 部署的文件
├── setup.py                        # 安装、打包相关的脚本
├── README.md                       # 项目的介绍和说明文件
└── .gitignore                      # Git忽略的文件和目录

# 数据原始图片

























