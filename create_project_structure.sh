#!/bin/bash

# 项目根目录名
PROJECT_NAME="ai_identify_human"

# 创建项目根目录
mkdir -p $PROJECT_NAME

# 创建 data 目录及其子目录
mkdir -p $PROJECT_NAME/data/raw
mkdir -p $PROJECT_NAME/data/processed
mkdir -p $PROJECT_NAME/data/interim
mkdir -p $PROJECT_NAME/data/external

# 创建 datasets 目录
mkdir -p $PROJECT_NAME/datasets/dataset1
mkdir -p $PROJECT_NAME/datasets/dataset2

# 创建 src 目录及其子目录
mkdir -p $PROJECT_NAME/src/config
mkdir -p $PROJECT_NAME/src/data_preparation
mkdir -p $PROJECT_NAME/src/training
mkdir -p $PROJECT_NAME/src/testing
mkdir -p $PROJECT_NAME/src/inferencing
mkdir -p $PROJECT_NAME/src/visualization
mkdir -p $PROJECT_NAME/src/utils

# 创建 notebooks 目录
mkdir -p $PROJECT_NAME/notebooks

# 创建 scripts 目录
mkdir -p $PROJECT_NAME/scripts

# 创建一些基础文件
touch $PROJECT_NAME/README.md
touch $PROJECT_NAME/requirements.txt
touch $PROJECT_NAME/Dockerfile
touch $PROJECT_NAME/setup.py
touch $PROJECT_NAME/.gitignore

# 在 src/config 目录中创建 config.json 文件
cat <<EOT >> $PROJECT_NAME/src/config/config.json
{
    "data_home_path": "/your/data/home/path"
}
EOT

# 在 src/data_preparation 目录中创建基础文件
touch $PROJECT_NAME/src/data_preparation/__init__.py
touch $PROJECT_NAME/src/data_preparation/clean_data.py
touch $PROJECT_NAME/src/data_preparation/prepare_data.py
touch $PROJECT_NAME/src/data_preparation/transform_data.py

# 在 src/training 目录中创建基础文件
touch $PROJECT_NAME/src/training/__init__.py
touch $PROJECT_NAME/src/training/train_model.py
touch $PROJECT_NAME/src/training/model.py
touch $PROJECT_NAME/src/training/utils.py

# 在 src/testing 目录中创建基础文件
touch $PROJECT_NAME/src/testing/__init__.py
touch $PROJECT_NAME/src/testing/evaluate.py
touch $PROJECT_NAME/src/testing/test_model.py

# 在 src/inferencing 目录中创建基础文件
touch $PROJECT_NAME/src/inferencing/__init__.py
touch $PROJECT_NAME/src/inferencing/predict.py
touch $PROJECT_NAME/src/inferencing/deploy.py

# 在 src/visualization 目录中创建基础文件
touch $PROJECT_NAME/src/visualization/__init__.py
touch $PROJECT_NAME/src/visualization/plot_results.py
touch $PROJECT_NAME/src/visualization/visualize_data.py

# 在 src/utils 目录中创建基础文件
touch $PROJECT_NAME/src/utils/__init__.py
touch $PROJECT_NAME/src/utils/logger.py
touch $PROJECT_NAME/src/utils/helpers.py

# 在 notebooks 目录中创建 notebook 文件
touch $PROJECT_NAME/notebooks/data_exploration.ipynb
touch $PROJECT_NAME/notebooks/model_training.ipynb

# 在 scripts 目录中创建脚本文件
touch $PROJECT_NAME/scripts/run_training.sh
touch $PROJECT_NAME/scripts/run_testing.sh

echo "Project structure created successfully."