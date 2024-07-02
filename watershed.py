import cv2
import numpy as np

# 读取图像
image = cv2.imread(r"D:\job\ui2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Otsu阈值法进行二值化
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 去除噪声
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 确定前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.9 * dist_transform.max(), 255, 0)

# 确定未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通组件
_, markers = cv2.connectedComponents(sure_fg)

# 为所有的标记加1，保证背景是1不是0
markers = markers + 1

# 标记未知区域为0
markers[unknown == 0] = 0

# 应用分水岭算法
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]

# 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Markers', markers)

# 等待按键输入后关闭所有窗口
cv2.imwrite('watershed.png', markers)

