import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'test.jpg'
print("Loading image...")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image loaded and converted to RGB.")

# 显示原始图像
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()
print("Displayed original image.")

# 将图像转换为灰度图像
print("Converting image to grayscale...")
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print("Image converted to grayscale.")

# 定义卷积核
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
print("Defined convolution kernels.")

# 应用卷积操作
print("Applying Sobel X convolution...")
sobel_x_image = cv2.filter2D(gray_image, -1, sobel_x)
print("Sobel X convolution applied.")

print("Applying Sobel Y convolution...")
sobel_y_image = cv2.filter2D(gray_image, -1, sobel_y)
print("Sobel Y convolution applied.")

print("Applying Prewitt X convolution...")
prewitt_x_image = cv2.filter2D(gray_image, -1, prewitt_x)
print("Prewitt X convolution applied.")

print("Applying Prewitt Y convolution...")
prewitt_y_image = cv2.filter2D(gray_image, -1, prewitt_y)
print("Prewitt Y convolution applied.")

print("Applying Laplacian convolution...")
laplacian_image = cv2.filter2D(gray_image, -1, laplacian)
print("Laplacian convolution applied.")

# 显示卷积后的图像
print("Displaying convolved images...")
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(sobel_x_image, cmap='gray')
axs[0, 0].set_title('Sobel X')
axs[0, 0].axis('off')

axs[0, 1].imshow(sobel_y_image, cmap='gray')
axs[0, 1].set_title('Sobel Y')
axs[0, 1].axis('off')

axs[0, 2].imshow(prewitt_x_image, cmap='gray')
axs[0, 2].set_title('Prewitt X')
axs[0, 2].axis('off')

axs[1, 0].imshow(prewitt_y_image, cmap='gray')
axs[1, 0].set_title('Prewitt Y')
axs[1, 0].axis('off')

axs[1, 1].imshow(laplacian_image, cmap='gray')
axs[1, 1].set_title('Laplacian')
axs[1, 1].axis('off')

plt.show()
print("Displayed all convolved images.")