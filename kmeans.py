import cv2
import numpy as np

# 读取图像
image = cv2.imread(r"D:\job\ui2.png")
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# 将图像转换为二维数组
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 定义K-means参数
k = 5  # 聚类数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
attempts = 10

# 应用K-means算法
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

# 将中心转换回8位无符号整型
centers = np.uint8(centers)

# 映射每个像素到其中心
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# 显示原始图像和分割后的图像
cv2.imwrite('kmeans.png', segmented_image)
