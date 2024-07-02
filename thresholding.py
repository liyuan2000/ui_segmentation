import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread(r"D:\job\ui2.png", cv2.IMREAD_GRAYSCALE)

# # 应用简单阈值分割
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('thresh.png',thresh1)
# 应用自适应阈值分割
thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('thresh_mean.png',thresh2)
cv2.imwrite('thresh_gaussian.png',thresh3)
# 显示结果
titles = ['Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [thresh2, thresh3]

for i in range(2):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

