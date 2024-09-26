import cv2
import os
from ..config import Config


def create_faces_folder(directory_path):
    """
    创建 faces 文件夹，用于保存截取的头像
    :param directory_path: 原始图片所在目录
    :return: faces 文件夹路径
    """
    faces_folder = os.path.join(directory_path, 'faces')
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)
    return faces_folder


def detect_and_save_faces(image_path, faces_folder, face_cascade):
    """
    检测并保存图片中的所有人脸
    :param image_path: 原始图片路径
    :param faces_folder: 保存头像的文件夹路径
    :param face_cascade: 人脸检测模型
    """
    # 读取图像
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 保存人脸
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y + h, x:x + w]
        face_filename = os.path.join(faces_folder,
                                     f'{os.path.splitext(os.path.basename(image_path))[0]}_face_{i + 1}.jpg')
        cv2.imwrite(face_filename, face)
        print(f'Saved face {i + 1} to {face_filename}')


def process_directory(directory_path):
    """
    遍历目录下所有图片，检测并保存人脸
    :param directory_path: 要处理的图片目录
    """
    # 加载预训练的人脸检测模型（Haar Cascades）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 创建 faces 文件夹
    faces_folder = create_faces_folder(directory_path)

    # 遍历目录中的所有图像文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                detect_and_save_faces(image_path, faces_folder, face_cascade)


if __name__ == "__main__":
    # 从 Config 类获取配置
    config = Config()

    # 使用 config.test_image_path 作为目录路径
    directory_path = config.test_image_path

    # 处理目录中的所有图像
    process_directory(directory_path)