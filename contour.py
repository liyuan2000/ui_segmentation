import cv2
import numpy as np

# 读取图像
image = cv2.imread(r"D:\job\ui2.png")
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用边缘检测
edges = cv2.Canny(gray, 50, 150)

# 找到轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
segmented_image = image.copy()
cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)


cv2.imwrite('contour.png', segmented_image)